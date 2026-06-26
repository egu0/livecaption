"""Language tag normalization for user-facing CLI options.

ASR models expect exact prompt keys such as ``en-US``. Translation prompts work better
with natural language names such as ``Simplified Chinese``. The CLI accepts both compact
language tags and English language names, so this module bridges those forms to the
model-facing values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")

_PRIMARY_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_DEFAULT_TAGS = {
    "de": "de-de",
    "en": "en-us",
    "es": "es-es",
    "fr": "fr-fr",
    "it": "it-it",
    "ja": "ja-jp",
    "ko": "ko-kr",
    "pt": "pt-pt",
    "zh": "zh-cn",
}

_TAG_PROMPT_NAMES = {
    "pt-br": "Brazilian Portuguese",
    "pt-pt": "European Portuguese",
    "zh-cn": "Simplified Chinese",
    "zh-hans": "Simplified Chinese",
    "zh-sg": "Simplified Chinese",
    "zh-hant": "Traditional Chinese",
    "zh-hk": "Traditional Chinese",
    "zh-mo": "Traditional Chinese",
    "zh-tw": "Traditional Chinese",
}


def _name_key(value: str) -> str:
    return " ".join(value.replace("-", " ").replace("_", " ").split()).casefold()


_NAME_TO_TAG = {
    _name_key(name): _DEFAULT_TAGS.get(code, code)
    for code, name in _PRIMARY_LANGUAGE_NAMES.items()
}
_NAME_TO_TAG.update(
    {
        "brazilian portuguese": "pt-BR",
        "european portuguese": "pt-PT",
        "mandarin": "zh-CN",
        "simplified chinese": "zh-CN",
        "traditional chinese": "zh-TW",
    }
)

_SUPPORTED_TARGET_CODES = set(_PRIMARY_LANGUAGE_NAMES) | {
    code.split("-", 1)[0] for code in _TAG_PROMPT_NAMES
}


@dataclass(frozen=True)
class TargetLanguage:
    """Normalized target language.

    ``code`` is the user-facing language tag. ``prompt_name`` is what gets inserted into
    the translation prompt.
    """

    code: str
    prompt_name: str


def _require_value(value: str, option_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{option_name} cannot be empty")
    return stripped


def canonicalize_language_tag(value: str) -> str:
    """Return the CLI spelling for a compact language tag.

    This deliberately handles the common CLI forms rather than trying to implement the
    full BCP-47 grammar. CLI-facing language tags are lowercase so examples and defaults
    are easy to type and compare.
    """

    raw = value.strip()
    if not _LANGUAGE_TAG_RE.fullmatch(raw):
        return raw
    return "-".join(part.lower() for part in re.split("[-_]", raw))


def _tag_prompt_name(code: str) -> str:
    key = code.casefold()
    if key in _TAG_PROMPT_NAMES:
        return _TAG_PROMPT_NAMES[key]
    primary = key.split("-", 1)[0]
    return _PRIMARY_LANGUAGE_NAMES.get(primary, code)


def normalize_target_language(value: str) -> TargetLanguage:
    """Normalize ``--target-lang`` from either a language tag or an English name."""

    raw = _require_value(value, "--target-lang")
    name_key = _name_key(raw)
    tag = _NAME_TO_TAG.get(name_key)
    if tag is None and not _LANGUAGE_TAG_RE.fullmatch(raw):
        raise ValueError(
            f"Target language '{raw}' is not a valid language tag. "
            "Use a language tag or English name, e.g. zh-cn / Chinese, ja-jp / Japanese."
        )
    tag = tag or raw
    code = canonicalize_language_tag(tag)
    primary = code.split("-", 1)[0]
    if primary not in _SUPPORTED_TARGET_CODES:
        raise ValueError(
            f"Target language '{raw}' is not recognized. "
            "Use a supported language tag or English name, e.g. zh-cn / Chinese, "
            "ja-jp / Japanese, or de-de / German."
        )
    return TargetLanguage(code=code, prompt_name=_tag_prompt_name(code))


def _case_insensitive_match(candidate: str, supported: Iterable[str]) -> str | None:
    lowered = candidate.casefold()
    for item in supported:
        if item.casefold() == lowered:
            return item
    return None


def normalize_asr_language(value: str, supported: Iterable[str] | None = None) -> str:
    """Normalize ``--asr-lang`` to an exact model prompt key when possible."""

    raw = _require_value(value, "--asr-lang")
    if raw.casefold() == "auto":
        if supported is None:
            return "auto"
        return _case_insensitive_match("auto", supported) or "auto"

    tag = _NAME_TO_TAG.get(_name_key(raw), raw)
    code = canonicalize_language_tag(tag)
    if supported is None:
        return code

    supported_list = list(supported)
    match = _case_insensitive_match(code, supported_list)
    if match is not None:
        return match

    primary = code.split("-", 1)[0].casefold()
    primary_matches = [
        item
        for item in supported_list
        if item.casefold() == primary or item.casefold().startswith(f"{primary}-")
    ]
    if len(primary_matches) == 1:
        return primary_matches[0]
    return code
