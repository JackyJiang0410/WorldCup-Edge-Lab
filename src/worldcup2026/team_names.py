from __future__ import annotations

import re
import unicodedata

from .constants import TEAM_ALIASES


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def team_key(value: str | None) -> str:
    if value is None:
        return ""
    text = strip_accents(str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_team_name(value: str | None) -> str:
    raw = "" if value is None else str(value).strip()
    key = team_key(raw)
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    return raw


def is_prohibited_trade_team(value: str | None) -> bool:
    return team_key(normalize_team_name(value)) in {"argentina", "spain"}

