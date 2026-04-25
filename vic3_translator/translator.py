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
from collections import deque
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


class _SplitBatchError(Exception):
    """Signal that a failed batch should be re-queued as smaller batches."""

    def __init__(self, batch: List[str], cause: Exception):
        super().__init__(str(cause))
        self.batch = batch
        self.cause = cause


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
    error_retries: int = 1,
    rate_limit_policy: str = "backoff",  # "backoff" | "stop" | "ignore" | "retry"
    error_policy: str = "retry",  # "stop" | "ignore" | "retry"
    cache: Optional[TranslationCache] = None,
    progress_cb: Optional[ProgressCb] = None,
    stop_flag: Optional[threading.Event] = None,
    smart_skip_translated: bool = False,
    reasoning_effort: str = "none",
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

    if progress_cb is not None:
        if pending:
            progress_cb(0, len(pending), f"キャッシュ確認完了: {result.cache_hits}件ヒット, {len(pending)}件をAPI翻訳中")
        else:
            progress_cb(1, 1, f"全件キャッシュヒット ({result.cache_hits}件)")

    # 3.5 Pre-compute per-file remaining counts for O(1) readiness checks.
    # Instead of scanning all entries of every pending file after each batch
    # (which becomes O(files × entries × batches) — the root cause of the
    # 90%-slowdown), we track how many *untranslated* unique values each file
    # still needs.  When a batch finishes we decrement counters for the
    # newly-translated strings and write files whose counter reaches zero.
    written_incrementally: set = set()

    # Map: source_text -> list of file indices that need it
    _text_to_file_indices: Dict[str, List[int]] = {}
    # Per-file remaining count of untranslated unique values
    _file_remaining: List[int] = []

    for fi, (pf, _rel) in enumerate(parsed_files):
        unique_values_for_file = set()
        for entry in pf.entries:
            if entry.value not in translation_map:
                unique_values_for_file.add(entry.value)
        _file_remaining.append(len(unique_values_for_file))
        for v in unique_values_for_file:
            _text_to_file_indices.setdefault(v, []).append(fi)

    def _mark_translated_and_write(newly_translated: List[str]) -> None:
        """Decrement remaining counts and write ready files."""
        ready_indices: List[int] = []
        for src_text in newly_translated:
            for fi in _text_to_file_indices.pop(src_text, []):
                _file_remaining[fi] -= 1
                if _file_remaining[fi] == 0:
                    ready_indices.append(fi)

        for fi in ready_indices:
            if str(parsed_files[fi][0].path) in written_incrementally:
                continue
            if stop_flag is not None and stop_flag.is_set():
                continue
            pf, rel_root = parsed_files[fi]
            try:
                _write_one_file(pf, rel_root, translation_map, out_dir, tgt_header_key, source_lang_key, target_lang_key, result)
                written_incrementally.add(str(pf.path))
            except Exception as e:
                logger.error("Failed to write %s incrementally: %s", pf.path, e)
                result.failed_files.append(str(pf.path))

    translated_unique_strings = sum(1 for t in unique_texts if t in translation_map)
    total_unique_strings = len(unique_texts)

    # Write any files that are already fully satisfied by cache / smart-skip
    initial_ready = [fi for fi in range(len(parsed_files)) if _file_remaining[fi] == 0]
    if initial_ready:
        for fi in initial_ready:
            pf, rel_root = parsed_files[fi]
            if stop_flag is not None and stop_flag.is_set():
                break
            try:
                _write_one_file(pf, rel_root, translation_map, out_dir, tgt_header_key, source_lang_key, target_lang_key, result)
                written_incrementally.add(str(pf.path))
            except Exception as e:
                logger.error("Failed to write %s incrementally: %s", pf.path, e)
                result.failed_files.append(str(pf.path))
            
            if progress_cb is not None:
                progress_cb(
                    translated_unique_strings,
                    total_unique_strings,
                    f"キャッシュから出力中... (ファイル出力: {result.written_files}/{result.total_files})"
                )

    # 4. DeepSeek calls
    if pending:
        client = DeepSeekClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            max_retries=max_retries,
            error_retries=error_retries,
            rate_limit_policy=rate_limit_policy,
            reasoning_effort=reasoning_effort,
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

        def _split_batch(batch: List[str]) -> Tuple[List[str], List[str]]:
            mid = len(batch) // 2
            return batch[:mid], batch[mid:]

        def _work(batch: List[str]) -> Tuple[List[str], List[str]]:
            def _translate_whole_with_policy(texts: List[str], is_fallback: bool = False) -> List[str]:
                if rate_limit_policy != "retry":
                    return client.translate_batch(
                        texts,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        mod_context=mod_context,
                        max_retries=1 if is_fallback else None,
                        error_retries=1 if is_fallback else None,
                    )

                retry_round = 0
                while True:
                    try:
                        return client.translate_batch(
                            texts,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            mod_context=mod_context,
                            max_retries=1 if is_fallback else None,
                            error_retries=1 if is_fallback else None,
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

            def _translate_with_fallback(texts: List[str], is_fallback: bool = False) -> List[str]:
                try:
                    return _translate_whole_with_policy(texts, is_fallback=is_fallback)
                except DeepSeekRateLimitError:
                    # Preserve existing semantics for rate limit handling.
                    raise
                except DeepSeekError as e:
                    if error_policy != "retry":
                        raise  # Bubble up to handle according to error_policy

                    # Keep fallback work in the shared queue so freed worker slots
                    # can immediately pick up the split batches.
                    if len(texts) == 1:
                        logger.warning(
                            "Batch fallback to source text for 1 string after DeepSeekError: %s",
                            e,
                        )
                        return [texts[0]]

                    left, right = _split_batch(texts)
                    logger.warning(
                        "Batch failed (size=%d). Splitting into %d + %d for fallback recovery: %s",
                        len(texts),
                        len(left),
                        len(right),
                        e,
                    )
                    raise _SplitBatchError(texts, e) from e

            return batch, _translate_with_fallback(batch)

        pending_batches = deque(batches)

        # Keep only up to current_workers batches active at a time and refill
        # slots as soon as one completes.
        with ThreadPoolExecutor(max_workers=current_workers) as pool:
            active_futures = {}

            def _submit_available() -> None:
                while pending_batches and len(active_futures) < current_workers:
                    batch = pending_batches.popleft()
                    future = pool.submit(_work, batch)
                    active_futures[future] = batch

            _submit_available()
            try:
                while active_futures or pending_batches:
                    if stop_flag is not None and stop_flag.is_set():
                        logger.warning(
                            "Cancellation requested — aborting remaining batches"
                        )
                        for f in active_futures:
                            f.cancel()
                        active_futures.clear()
                        pending_batches.clear()
                        break

                    if not active_futures:
                        _submit_available()
                        if not active_futures:
                            break

                    done, _ = wait(
                        set(active_futures.keys()),
                        timeout=0.3,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        continue

                    for future in done:
                        batch = active_futures.pop(future)
                        newly_translated: List[str] = []
                        try:
                            src_list, translated_list = future.result()
                        except _SplitBatchError as e:
                            left, right = _split_batch(e.batch)
                            pending_batches.appendleft(right)
                            pending_batches.appendleft(left)
                            total_batches += 1
                            logger.info(
                                "Re-queued split batch into %d + %d items (queued=%d, active=%d, target_workers=%d)",
                                len(left),
                                len(right),
                                len(pending_batches),
                                len(active_futures),
                                current_workers,
                            )
                            continue
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
                                for f in active_futures:
                                    f.cancel()
                                active_futures.clear()
                                pending_batches.clear()
                                break
                            # Fallback only for this failed batch.
                            for s in batch:
                                translation_map[s] = s
                                newly_translated.append(s)
                            logger.error("Batch rate-limited and failed: %s", e)
                        except DeepSeekError as e:
                            result.failed_batches += 1
                            if error_policy == "stop":
                                logger.error(
                                    "Error hit, stopping translation per policy"
                                )
                                for f in active_futures:
                                    f.cancel()
                                active_futures.clear()
                                pending_batches.clear()
                                break
                            # Fallback only for this failed batch.
                            for s in batch:
                                translation_map[s] = s
                                newly_translated.append(s)
                            logger.error("Batch failed (policy=%s): %s", error_policy, e)
                        except Exception as e:  # noqa: BLE001
                            result.failed_batches += 1
                            # Unexpected errors are also isolated to this batch.
                            for s in batch:
                                translation_map[s] = s
                                newly_translated.append(s)
                            logger.exception("Unexpected batch error: %s", e)
                        else:
                            for s, t in zip(src_list, translated_list):
                                translation_map[s] = t
                                newly_translated.append(s)
                            if cache is not None:
                                cache.put_many(
                                    zip(src_list, translated_list),
                                    source_lang=source_lang_key,
                                    target_lang=target_lang_key,
                                    model=model,
                                )

                        # Efficiently update remaining counts and write ready files
                        if newly_translated:
                            translated_unique_strings += len(newly_translated)
                            _mark_translated_and_write(newly_translated)

                        done_batches += 1
                        if progress_cb is not None:
                            progress_cb(
                                translated_unique_strings,
                                total_unique_strings,
                                f"翻訳バッチ {done_batches}/{total_batches} 完了 "
                                f"({translated_unique_strings}/{total_unique_strings}文字列, "
                                f"ファイル出力: {result.written_files}/{result.total_files})",
                            )
                    _submit_available()
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
                    # Keep progress at 100% during write phase to avoid resetting the bar
                    progress_cb(
                        total_unique_strings, total_unique_strings,
                        f"最終処理中... (ファイル出力: {result.written_files}/{result.total_files})"
                    )
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to write %s: %s", pf.path, e)
                result.failed_files.append(str(pf.path))
    else:
        logger.info("All files already written incrementally; skipping bulk write phase.")
        if progress_cb is not None:
            progress_cb(total_unique_strings, total_unique_strings, f"完了: {result.total_files} ファイル出力済み")

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
