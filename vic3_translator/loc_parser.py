"""Parser / writer for Victoria 3 localization ``.yml`` files.

The format is *not* standard YAML; it looks like::

    l_english:
     key.name:0 "Some translated string with [SCOPE.Func] and $concept$"
     another_key:1 "Multi-word    value. With #bold Formatting#! tags"

Rules implemented here:

* The file is encoded as **UTF-8 with BOM** (``utf-8-sig``).
* The first non-empty line is the language header (``l_english:`` etc.).
* Each entry line is ``<indent><key>:<version?> "<value>"``.
    (both ``key:0 "..."`` and ``key: "..."`` are accepted)
* Lines that do not match (comments starting with ``#``, blank lines,
  malformed entries) are kept verbatim so the output preserves the
  exact layout / comments / ordering of the source.
* Inside ``<value>`` the game engine recognises ``\\"`` as an escaped
  quote.  We preserve it literally during round-trip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union


# Matches an entry line. Groups: (indent, key, version?, value)
# Supports both `key:0 "..."` and `key: "..."` styles.
# Handles escaped quotes \" and trailing comments # ...
ENTRY_RE = re.compile(
    r'^(\s*)([A-Za-z0-9_.\-]+):\s*(?:([0-9]+)\s*)?"((?:[^"\\]|\\.)*)"(?:\s*#.*)?$'
)

# The language header: "l_english:" on its own line (possibly with BOM stripped)
HEADER_RE = re.compile(r"^\s*(l_[a-z_]+)\s*:\s*$")

# Fallback parser for format-variant entry lines.
# Accepts broad key patterns and then extracts version/value in code.
ENTRY_FALLBACK_RE = re.compile(r'^(\s*)([^\s:#][^:]*)\s*:\s*(.*)$')
VALUE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass
class LocEntry:
    """A single translatable entry."""

    line_index: int  # 0-based index into the original lines list
    indent: str
    key: str
    version: str | None  # keep as string to preserve ``0``/``1`` etc.
    value: str  # raw value (without surrounding quotes)


@dataclass
class ParsedLocFile:
    """Result of :func:`parse_loc_file`."""

    path: Path
    header: str  # e.g. "l_english"
    original_lines: List[str] = field(default_factory=list)
    entries: List[LocEntry] = field(default_factory=list)


def parse_loc_file(path: Union[str, Path]) -> ParsedLocFile:
    """Parse a Victoria 3 localization .yml file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    # splitlines() drops the trailing empty string, preserve line endings manually
    lines = text.splitlines()

    header = ""
    entries: List[LocEntry] = []

    for idx, line in enumerate(lines):
        if not header:
            m = HEADER_RE.match(line)
            if m:
                header = m.group(1)
                continue

        m = ENTRY_RE.match(line)
        if m:
            indent, key, version, value = m.groups()
            entries.append(
                LocEntry(
                    line_index=idx,
                    indent=indent or " ",
                    key=key,
                    version=version,
                    value=value,
                )
            )
            continue

        # Tolerant fallback:
        # If the line has <key>: ..., try to extract optional numeric version
        # and quoted value even when spacing/format differs from canonical style.
        fm = ENTRY_FALLBACK_RE.match(line)
        if fm:
            indent, key, rest = fm.groups()
            rest = rest.strip()

            version: str | None = None
            vm = re.match(r"^([0-9]+)\b\s*(.*)$", rest)
            if vm:
                version = vm.group(1)
                rest = vm.group(2).lstrip()

            q = VALUE_RE.search(rest)
            if q:
                value = q.group(1)
                entries.append(
                    LocEntry(
                        line_index=idx,
                        indent=indent or " ",
                        key=key.strip(),
                        version=version,
                        value=value,
                    )
                )

    if not header:
        # Fall back to filename-derived header if missing
        header = "l_english"

    return ParsedLocFile(
        path=path,
        header=header,
        original_lines=lines,
        entries=entries,
    )


def rebuild_lines(
    parsed: ParsedLocFile,
    target_header: str,
    translations: List[str],
) -> List[str]:
    """Return a new list of lines where entries have been replaced with
    translated values and the language header swapped to ``target_header``.

    ``translations`` must be aligned with ``parsed.entries``.
    Unmatched entries fall back to the original value.
    """
    if len(translations) != len(parsed.entries):
        raise ValueError(
            f"translations length {len(translations)} does not match "
            f"entries length {len(parsed.entries)}"
        )

    # Build a map: original line_index -> replacement line
    replacement: dict[int, str] = {}
    for entry, translated in zip(parsed.entries, translations):
        # Escape any bare double-quote inside the translation
        safe = _escape_value(translated)
        version_part = f":{entry.version}" if entry.version is not None else ":"
        new_line = f'{entry.indent}{entry.key}{version_part} "{safe}"'
        replacement[entry.line_index] = new_line

    out: List[str] = []
    header_replaced = False
    for idx, line in enumerate(parsed.original_lines):
        if not header_replaced:
            m = HEADER_RE.match(line)
            if m:
                # Preserve any leading whitespace before the header
                leading_ws = line[: m.start(1)]
                out.append(f"{leading_ws}{target_header}:")
                header_replaced = True
                continue

        if idx in replacement:
            out.append(replacement[idx])
        else:
            out.append(line)

    if not header_replaced:
        # File had no detectable header; prepend one.
        out.insert(0, f"{target_header}:")

    return out


def save_loc_file(
    out_path: Union[str, Path],
    lines: List[str],
    newline: str = "\n",
) -> None:
    """Write ``lines`` to ``out_path`` using UTF-8 with BOM (utf-8-sig).

    Adds a trailing newline to mimic Paradox's own output.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = newline.join(lines)
    if not content.endswith(newline):
        content += newline
    out_path.write_text(content, encoding="utf-8-sig", newline=newline)


def translate_filename(filename: str, src_key: str, tgt_key: str) -> str:
    """Replace ``_l_<src>.yml`` suffix with ``_l_<tgt>.yml``.

    If the source suffix is not found, return the filename unchanged.
    """
    src_suffix = f"_l_{src_key}.yml"
    tgt_suffix = f"_l_{tgt_key}.yml"
    if filename.endswith(src_suffix):
        return filename[: -len(src_suffix)] + tgt_suffix
    return filename


def _escape_value(value: str) -> str:
    """Escape bare ``"`` characters inside a loc value.

    Paradox's own files already use ``\\"``; keep already-escaped quotes
    intact.
    """
    out: list[str] = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            # Preserve any escape sequence (\", \n, \t, \\ ...)
            out.append(value[i : i + 2])
            i += 2
            continue
        if ch == '"':
            out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def discover_yml_files(
    source_root: Path,
    source_lang_key: str,
) -> List[Tuple[Path, Path]]:
    """Walk ``source_root`` and return a list of ``(absolute_path, rel_root)``.

    ``rel_root`` is the path of the file's parent folder *relative to*
    ``source_root``, so the output writer can reconstruct the same
    directory layout under a different root.

    Only files whose name ends with ``_l_<source_lang_key>.yml`` are returned.
    Both direct ``localization/<lang>/`` and ``localization/replace/<lang>/``
    layouts are supported.
    """
    results: List[Tuple[Path, Path]] = []
    if not source_root.exists():
        return results

    suffix = f"_l_{source_lang_key}.yml"
    for path in source_root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        rel_root = path.parent.relative_to(source_root)
        results.append((path, rel_root))

    return results
