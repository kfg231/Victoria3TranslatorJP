"""Core translation workflow.

Pipeline::

    discover yml files
         │
         ▼
    parse all files, collect unique source strings
         │
         ▼
    look up in cache  →  cache hits (skip API)
         │
         ▼
    batch remaining strings and dispatch via ThreadPoolExecutor
         │
         ▼
    write results back to <output>/<src>_to_<tgt>/localization/<tgt>/...

The caller (CLI or GUI) passes a ``progress_cb(done, total, message)``
that is invoked after every batch to drive the progress bar.
A ``threading.Event`` allows cooperative cancellation.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import loc_parser
from .cache_manager import TranslationCache
from .deepseek_client import DEFAULT_BASE_URL, DeepSeekClient, DeepSeekError, DeepSeekRateLimitError
from .language_config import get_language


logger = logging.getLogger(__name__)


ProgressCb = Callable[[int, int, str], None]


@dataclass
class TranslationResult:
    total_files: int = 0
    written_files: int = 0
    skipped_files: int = 0
    total_strings: int = 0
    cache_hits: int = 0
    api_strings: int = 0
    failed_batches: int = 0
    failed_files: List[str] = field(default_factory=list)
    output_dir: Optional[Path] = None


def _chunks(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def _is_already_translated(source_val: str, tgt_val: str, target_lang_key: str) -> bool:
    """Check if the string is already translated using comparison and character heuristics."""
    if source_val != tgt_val:
        return True
    
    if target_lang_key == "japanese":
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', tgt_val):
            return True
    elif target_lang_key == "simp_chinese":
        if re.search(r'[\u4E00-\u9FFF]', tgt_val):
            return True
    elif target_lang_key == "korean":
        if re.search(r'[\uAC00-\uD7A3]', tgt_val):
            return True
    elif target_lang_key == "russian":
        if re.search(r'[А-Яа-я]', tgt_val):
            return True
            
    return False


def translate_mod(
    *,
    source_lang_key: str,
    target_lang_key: str,
    source_root: Path,
    output_root: Path,
    output_name: Optional[str] = None,
    api_key: str,
    mod_context: str = "",
    model: str = "deepseek-v4-pro",
    enable_thinking: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    batch_size: int = 40,
    initial_max_workers: int = 8,
    max_retries: int = 3,
    rate_limit_policy: str = "backoff",  # "backoff" | "stop" | "ignore" | "retry"
    cache: Optional[TranslationCache] = None,
    progress_cb: Optional[ProgressCb] = None,
    stop_flag: Optional[threading.Event] = None,
    smart_skip_translated: bool = False,
) -> TranslationResult:
    """Translate every ``_l_<source>.yml`` under ``source_root``."""
    source_root = Path(source_root)
    output_root = Path(output_root)

    source_lang = dict(get_language(source_lang_key))
    source_lang["folder_key"] = source_lang_key
    target_lang = dict(get_language(target_lang_key))
    target_lang["folder_key"] = target_lang_key

    if source_lang_key == target_lang_key:
        raise ValueError("Source and target languages must differ")

    src_header_key = source_lang["key"]  # e.g. "l_english"
    tgt_header_key = target_lang["key"]  # e.g. "l_japanese"

    if output_name is None:
        output_name = f"{source_lang_key}_to_{target_lang_key}"

    out_dir = output_root / output_name
    if out_dir.exists():
        raise ValueError(
            f"Output folder already exists: {out_dir}. "
            "Please choose a different output folder name."
        )

    result = TranslationResult(output_dir=out_dir)

    # 1. discover files
    files = loc_parser.discover_yml_files(source_root, source_lang_key)
    result.total_files = len(files)
    if not files:
        logger.warning(
            "No files ending with _l_%s.yml found under %s",
            source_lang_key,
            source_root,
        )
        return result
    logger.info("Discovered %d .yml files to translate", len(files))

    # 2. parse
    parsed_files: List[Tuple[loc_parser.ParsedLocFile, Path]] = []
    all_texts: List[str] = []
    for i, (abs_path, rel_root) in enumerate(files):
        if stop_flag is not None and stop_flag.is_set():
            return result
        
        if progress_cb:
            progress_cb(i, len(files), f"解析中: {abs_path.name}")
            
        try:
            pf = loc_parser.parse_loc_file(abs_path)
        except Exception as e:
            logger.error("Failed to parse %s: %s", abs_path, e)
            result.failed_files.append(str(abs_path))
            continue
        parsed_files.append((pf, rel_root))
        for entry in pf.entries:
            all_texts.append(entry.value)

    result.total_strings = len(all_texts)
    unique_texts = list(dict.fromkeys(all_texts))  # dedupe, preserve order
    logger.info(
        "Collected %d strings (%d unique) from %d files",
        len(all_texts),
        len(unique_texts),
        len(parsed_files),
    )

    if stop_flag is not None and stop_flag.is_set():
        return result

    # 2.5 smart skip logic
    translation_map: Dict[str, str] = {}
    if smart_skip_translated:
        logger.info("Smart skip enabled. Checking for existing translated files...")
        if progress_cb:
            progress_cb(0, 1, "既存の翻訳ファイルを確認中...")
            
        skipped_count = 0
        for pf, rel_root in parsed_files:
            if stop_flag is not None and stop_flag.is_set():
                return result
                
            tgt_rel = _remap_relative_root(rel_root, source_lang_key, target_lang_key)
            tgt_filename = loc_parser.translate_filename(pf.path.name, source_lang_key, target_lang_key)
            existing_target_path = source_root / tgt_rel / tgt_filename
            
            if existing_target_path.exists():
                try:
                    tgt_pf = loc_parser.parse_loc_file(existing_target_path)
                    tgt_dict = {e.key: e.value for e in tgt_pf.entries}
                    for src_e in pf.entries:
                        if src_e.key in tgt_dict:
                            tgt_val = tgt_dict[src_e.key]
                            if _is_already_translated(src_e.value, tgt_val, target_lang_key):
                                translation_map[src_e.value] = tgt_val
                                skipped_count += 1
                except Exception as e:
                    logger.warning("Failed to parse existing target file %s: %s", existing_target_path, e)
        if skipped_count > 0:
            logger.info("Smart skip: preserved %d already translated strings from existing files.", skipped_count)

    # 3. cache lookup
    if cache is not None and unique_texts:
        if progress_cb:
            progress_cb(0, 1, "キャッシュを確認中...")
        cached = cache.get_many(
            unique_texts,
            source_lang=source_lang_key,
            target_lang=target_lang_key,
            model=model,
        )

        translation_map.update(cached)
        result.cache_hits = len(cached)
        logger.info("Cache hits: %d / %d", len(cached), len(unique_texts))

    pending = [t for t in unique_texts if t not in translation_map]
    result.api_strings = len(pending)

    # 3.5 Track which files have been written incrementally
    written_incrementally = set()  # track already-written file paths to skip in step 5
    pending_files = list(parsed_files)
    
    # 4. DeepSeek calls
    if pending:
        client = DeepSeekClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            max_retries=max_retries,
            rate_limit_policy=rate_limit_policy,
        )

        batches = _chunks(pending, batch_size)
        done_batches = 0
        total_batches = len(batches)
        current_workers = initial_max_workers
        rate_limit_hits = 0
        logger.info(
            "Dispatching %d batches × up to %d strings to DeepSeek (workers=%d)",
            total_batches,
            batch_size,
            current_workers,
        )

        # Use at most one worker per batch; cap with user-supplied max_workers.
        workers = max(1, min(current_workers, total_batches))

        def _work(batch: List[str]) -> Tuple[List[str], List[str]]:
            def _translate_whole_with_policy(texts: List[str]) -> List[str]:
                if rate_limit_policy != "retry":
                    return client.translate_batch(
                        texts,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        mod_context=mod_context,
                    )

                retry_round = 0
                while True:
                    try:
                        return client.translate_batch(
                            texts,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            mod_context=mod_context,
                        )
                    except DeepSeekRateLimitError as e:
                        retry_round += 1
                        if retry_round > max_retries:
                            raise
                        wait_s = min(15 * retry_round, 120)
                        logger.warning(
                            "429 detected for batch size=%d, retry policy waiting %ds (round %d/%d): %s",
                            len(texts),
                            wait_s,
                            retry_round,
                            max_retries,
                            e,
                        )
                        time.sleep(wait_s)

            def _translate_with_fallback(texts: List[str]) -> List[str]:
                try:
                    return _translate_whole_with_policy(texts)
                except DeepSeekRateLimitError:
                    # Preserve existing semantics for rate limit handling.
                    raise
                except DeepSeekError as e:
                    # Parse/format failures: contain blast radius to this batch only.
                    if len(texts) == 1:
                        logger.warning(
                            "Batch fallback to source text for 1 string after DeepSeekError: %s",
                            e,
                        )
                        return [texts[0]]

                    mid = len(texts) // 2
                    left = texts[:mid]
                    right = texts[mid:]
                    logger.warning(
                        "Batch failed (size=%d). Splitting into %d + %d for fallback recovery: %s",
                        len(texts),
                        len(left),
                        len(right),
                        e,
                    )
                    return _translate_with_fallback(left) + _translate_with_fallback(right)

            return batch, _translate_with_fallback(batch)

        # Create executor with initial worker count
        with ThreadPoolExecutor(max_workers=current_workers) as pool:
            futures = {pool.submit(_work, b): b for b in batches}
            try:
                pending_futures = set(futures.keys())
                while pending_futures:
                    if stop_flag is not None and stop_flag.is_set():
                        logger.warning(
                            "Cancellation requested — aborting remaining batches"
                        )
                        for f in pending_futures:
                            f.cancel()
                        break

                    done, pending_futures = wait(
                        pending_futures,
                        timeout=0.3,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        continue

                    for future in done:
                        batch = futures[future]
                        try:
                            src_list, translated_list = future.result()
                        except DeepSeekRateLimitError as e:
                            result.failed_batches += 1
                            rate_limit_hits += 1
                            if rate_limit_policy == "backoff" and current_workers > 1:
                                old = current_workers
                                current_workers = max(1, current_workers // 2)
                                logger.warning(
                                    "Rate limit hit #%d, reducing workers from %d to %d",
                                    rate_limit_hits,
                                    old,
                                    current_workers,
                                )
                            elif rate_limit_policy == "stop":
                                logger.error(
                                    "Rate limit hit, stopping translation per policy"
                                )
                                for f in pending_futures:
                                    f.cancel()
                                pending_futures.clear()
                                break
                            # Fallback only for this failed batch.
                            for s in batch:
                                translation_map[s] = s
                            logger.error("Batch rate-limited and failed: %s", e)
                        except DeepSeekError as e:
                            result.failed_batches += 1
                            # Fallback only for this failed batch.
                            for s in batch:
                                translation_map[s] = s
                            logger.error("Batch failed: %s", e)
                        except Exception as e:  # noqa: BLE001
                            result.failed_batches += 1
                            # Unexpected errors are also isolated to this batch.
                            for s in batch:
                                translation_map[s] = s
                            logger.exception("Unexpected batch error: %s", e)
                        else:
                            for s, t in zip(src_list, translated_list):
                                translation_map[s] = t
                            if cache is not None:
                                cache.put_many(
                                    zip(src_list, translated_list),
                                    source_lang=source_lang_key,
                                    target_lang=target_lang_key,
                                    model=model,
                                )

                            # Incremental writing: Check if any file can be written now
                            still_pending = []
                            for pf, rel_root in pending_files:
                                if stop_flag is not None and stop_flag.is_set():
                                    still_pending.append((pf, rel_root))
                                    continue
                                if all(e.value in translation_map for e in pf.entries):
                                    try:
                                        _write_one_file(pf, rel_root, translation_map, out_dir, tgt_header_key, source_lang_key, target_lang_key, result)
                                        written_incrementally.add(str(pf.path))
                                    except Exception as e:
                                        logger.error("Failed to write %s incrementally: %s", pf.path, e)
                                        result.failed_files.append(str(pf.path))
                                else:
                                    still_pending.append((pf, rel_root))
                            pending_files = still_pending

                        done_batches += 1
                        if progress_cb is not None:
                            # 進捗は 0%〜90% までをバッチ処理に割り当て
                            pct_progress = (done_batches / total_batches) * 90.0
                            progress_cb(
                                int(pct_progress * 10),  # 仮のdone/total (内部的に%)
                                1000,
                                f"翻訳バッチ {done_batches}/{total_batches} 完了 (書込: {result.written_files}/{result.total_files})",
                            )
            finally:
                # ensure pool shutdown on cancel
                pool.shutdown(wait=False, cancel_futures=True)

        if stop_flag is not None and stop_flag.is_set():
            logger.warning("Translation cancelled by user")
            return result

        missing_count = sum(1 for text in pending if text not in translation_map)
        if missing_count > 0 and rate_limit_policy != "ignore":
            raise DeepSeekError(
                f"{missing_count} strings remain untranslated due to API failures. "
                f"Current policy={rate_limit_policy}. "
                "Some batches may have crashed unexpectedly before fallback completed."
            )

    # 5. write remaining output (skip files already written incrementally)
    if stop_flag is not None and stop_flag.is_set():
        logger.warning("Translation cancelled — skipping file output")
        return result

    remaining_files = [(pf, rel_root) for pf, rel_root in parsed_files if str(pf.path) not in written_incrementally]
    if remaining_files:
        logger.info("Writing remaining %d translated files under %s (skipping %d already written)", len(remaining_files), out_dir, len(written_incrementally))
        total_remaining = len(remaining_files)
        for idx, (pf, rel_root) in enumerate(remaining_files):
            try:
                # Build translations aligned to entries order
                translations = [translation_map.get(e.value, e.value) for e in pf.entries]
                new_lines = loc_parser.rebuild_lines(pf, tgt_header_key, translations)

                # Output path: mirror the relative folder layout but swap the
                # top-level language folder from <source_lang_key> -> <target_lang_key>.
                out_rel = _remap_relative_root(rel_root, source_lang_key, target_lang_key)
                out_filename = loc_parser.translate_filename(
                    pf.path.name, source_lang_key, target_lang_key
                )
                out_path = out_dir / out_rel / out_filename

                loc_parser.save_loc_file(out_path, new_lines)
                result.written_files += 1
                if progress_cb is not None:
                    # 進捗 90%〜100% をファイル書き込みに割り当て
                    pct_progress = 90.0 + ((idx + 1) / total_remaining) * 10.0
                    progress_cb(
                        int(pct_progress * 10),
                        1000,
                        f"書き込み中 {result.written_files}/{result.total_files} ファイル完了",
                    )
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to write %s: %s", pf.path, e)
                result.failed_files.append(str(pf.path))
    else:
        logger.info("All files already written incrementally; skipping bulk write phase.")
        if progress_cb is not None:
            progress_cb(1000, 1000, f"完了: {result.total_files} ファイル書き込み済み")

    logger.info(
        "Done. %d files written, %d skipped, %d cache hits, %d API strings",
        result.written_files,
        result.skipped_files,
        result.cache_hits,
        result.api_strings,
    )
    return result


def _remap_relative_root(rel_root: Path, src_key: str, tgt_key: str) -> Path:
    """Replace a path segment equal to ``src_key`` with ``tgt_key``.

    This turns ``english/CWN·Events`` → ``japanese/CWN·Events`` and
    ``replace/english`` → ``replace/japanese``.
    """
    parts = list(rel_root.parts)
    parts = [tgt_key if p == src_key else p for p in parts]
    return Path(*parts) if parts else Path(".")


def _write_one_file(
    pf: loc_parser.ParsedLocFile,
    rel_root: Path,
    translation_map: Dict[str, str],
    out_dir: Path,
    tgt_header_key: str,
    source_lang_key: str,
    target_lang_key: str,
    result: TranslationResult,
) -> None:
    # Build translations aligned to entries order
    translations = [translation_map.get(e.value, e.value) for e in pf.entries]
    new_lines = loc_parser.rebuild_lines(pf, tgt_header_key, translations)

    # Output path remap
    out_rel = _remap_relative_root(rel_root, source_lang_key, target_lang_key)
    out_filename = loc_parser.translate_filename(
        pf.path.name, source_lang_key, target_lang_key
    )
    out_path = out_dir / out_rel / out_filename

    loc_parser.save_loc_file(out_path, new_lines)
    result.written_files += 1
