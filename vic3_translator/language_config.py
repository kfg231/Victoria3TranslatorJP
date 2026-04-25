"""Victoria 3 supported languages.

Each entry maps the *folder / key name* used by Victoria 3
(e.g. `english`, `simp_chinese`) to metadata used by the translator.

- ``code``:     BCP-47 style code used in prompts / glossary matching
- ``key``:      the full header key used in .yml files (``l_english:`` etc.)
- ``display``:  localized name shown in the GUI dropdown
- ``name_en``:  English name, fed to the LLM prompt
"""

from __future__ import annotations

from typing import Dict


LANGUAGES: Dict[str, Dict[str, str]] = {
    "english": {
        "code": "en",
        "key": "l_english",
        "display": "English",
        "name_en": "English",
    },
    "simp_chinese": {
        "code": "zh-CN",
        "key": "l_simp_chinese",
        "display": "简体中文 (Simplified Chinese)",
        "name_en": "Simplified Chinese",
    },
    "japanese": {
        "code": "ja",
        "key": "l_japanese",
        "display": "日本語 (Japanese)",
        "name_en": "Japanese",
    },
    "korean": {
        "code": "ko",
        "key": "l_korean",
        "display": "한국어 (Korean)",
        "name_en": "Korean",
    },
    "french": {
        "code": "fr",
        "key": "l_french",
        "display": "Français (French)",
        "name_en": "French",
    },
    "german": {
        "code": "de",
        "key": "l_german",
        "display": "Deutsch (German)",
        "name_en": "German",
    },
    "spanish": {
        "code": "es",
        "key": "l_spanish",
        "display": "Español (Spanish)",
        "name_en": "Spanish",
    },
    "russian": {
        "code": "ru",
        "key": "l_russian",
        "display": "Русский (Russian)",
        "name_en": "Russian",
    },
    "polish": {
        "code": "pl",
        "key": "l_polish",
        "display": "Polski (Polish)",
        "name_en": "Polish",
    },
    "braz_por": {
        "code": "pt-BR",
        "key": "l_braz_por",
        "display": "Português do Brasil",
        "name_en": "Brazilian Portuguese",
    },
    "turkish": {
        "code": "tr",
        "key": "l_turkish",
        "display": "Türkçe (Turkish)",
        "name_en": "Turkish",
    },
}


def get_language(name: str) -> Dict[str, str]:
    """Return language info by folder-name key. Raises KeyError if unknown."""
    return LANGUAGES[name]


def list_display_names() -> list[tuple[str, str]]:
    """Return [(folder_key, display_name), ...] for GUI dropdowns."""
    return [(k, v["display"]) for k, v in LANGUAGES.items()]
