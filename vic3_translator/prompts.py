"""Prompt templates used when calling DeepSeek for Vic3 translation."""

from __future__ import annotations

import json
from typing import Dict, List


SYSTEM_PROMPT = (
    "You are a professional translator specialised in Paradox Interactive's "
    "Victoria 3 game localization. You translate mod text from "
    "{source_lang_name} to {target_lang_name}. You MUST translate the text "
    "and NEVER return the original source text as the translation. "
    "You strictly preserve all game scripting syntax and produce only "
    "the JSON output requested."
)


# The main batch user-prompt. ``numbered_list`` is a JSON array of strings
# (the source texts). The model must answer with a JSON array of the same
# length containing the translations in the same order.
USER_PROMPT = """\
You will receive a JSON array of {count} Victoria 3 localization strings in \
{source_lang_name}. Translate every string into {target_lang_name}.

MOD CONTEXT: {mod_context}

CRITICAL RULES — read carefully:
1. DO NOT translate or alter any of the following tokens. Copy them verbatim:
   - Scope / function calls in square brackets, e.g. [ROOT.GetCountry.GetName],
     [SCOPE.sCountry('target_country_scope').GetName]
   - Concept references between dollar signs, e.g. $concept_war$, $concept_country$
   - Formatting tags such as #bold, #italic, #indent_newline:2, #! (closing tag),
     #R, #G, #B, #tooltip, #tooltippable, #weak, #variable etc.
   - Escape sequences: \\n, \\t, \\", \\\\
   - Curly-brace variables such as {{TARGET_COUNTRY}}, {{0|%}}
2. Translate ONLY the natural-language parts that sit between those tokens.
3. Preserve the original placement of every token — do not move them around,
   do not add extra spaces that weren't there.
4. Preserve leading/trailing spaces exactly as they appear.
5. Use punctuation conventions native to {target_lang_name}
   (e.g. full-width 「」 and 、 for Japanese, 。for periods).
6. Do NOT add explanations, pinyin/romaji, or any text outside the JSON.
7. DO NOT copy source sentences unchanged unless the source is a proper noun,
    acronym, pure number, or pure token sequence.
8. If target language is Japanese, output must be natural Japanese prose.
    For full sentences, Japanese grammar particles (e.g. は, が, を, に, で, の)
    should appear where appropriate. Returning Simplified Chinese sentences as-is
    is invalid.

OUTPUT FORMAT — this is mandatory:
Respond with a single JSON array of exactly {count} strings, in the same order
as the input. No markdown, no code fences, no commentary. Example:
  ["translated 1","translated 2","translated 3"]

SOURCE STRINGS (JSON array):
{numbered_list}
"""


def build_user_prompt(
    texts: List[str],
    source_lang: Dict[str, str],
    target_lang: Dict[str, str],
    mod_context: str,
) -> str:
    """Build the user prompt for a batch of texts."""
    numbered_list = json.dumps(texts, ensure_ascii=False)
    return USER_PROMPT.format(
        count=len(texts),
        source_lang_name=source_lang.get(
            "name_en", source_lang.get("display", "English")
        ),
        target_lang_name=target_lang.get(
            "name_en", target_lang.get("display", "Japanese")
        ),
        mod_context=mod_context or "(generic Victoria 3 mod)",
        numbered_list=numbered_list,
    )


def build_system_prompt(
    source_lang: Dict[str, str],
    target_lang: Dict[str, str],
) -> str:
    return SYSTEM_PROMPT.format(
        source_lang_name=source_lang.get("name_en", "English"),
        target_lang_name=target_lang.get("name_en", "Japanese"),
    )
