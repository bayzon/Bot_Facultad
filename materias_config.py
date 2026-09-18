#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
materias_config.py — Whitelist and parsing for WhatsApp Bot Facultad.

Active materias (only 3):
  - Redes de Datos
  - Gestion de Calidad (canon with tilde: Gestion de Calidad)
  - Ingenieria de Software (canon with tilde)

Parsing: `MATERIA: pregunta` split on first colon.
Validation: case-insensitive, accent-insensitive, whitespace-collapsed alias resolution.
"""

from __future__ import annotations

import re
import unicodedata

# Strict regex: everything before first colon is materia, after is question (trimmed)
MATERIA_REGEX = r"^([^:]+):\s*(.+)$"

# Compiled for reuse
_MATERIA_PATTERN = re.compile(MATERIA_REGEX)

# Canonical whitelist — only 3 active materias.
# Official exact names (with tildes where required).
MATERIAS_WHITELIST: list[str] = [
    "Redes de Datos",
    "Gestión de Calidad",
    "Ingeniería de Software",
]


def _normalize(text: str) -> str:
    """Lower, strip accents, collapse whitespace. Pure helper."""
    if not isinstance(text, str):
        return ""
    lowered = text.strip().lower()
    # Remove diacritics via NFD
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return collapsed


# Raw aliases (human-readable) -> canonical. Keys are normalized at build time.
_RAW_ALIASES: dict[str, str] = {
    # Redes de Datos
    "redes": "Redes de Datos",
    "red": "Redes de Datos",
    "redes de datos": "Redes de Datos",
    "redes datos": "Redes de Datos",
    "rd": "Redes de Datos",
    # Gestion de Calidad
    "gestion de calidad": "Gestión de Calidad",
    "gestión de calidad": "Gestión de Calidad",
    "calidad": "Gestión de Calidad",
    "gestion calidad": "Gestión de Calidad",
    "gestión calidad": "Gestión de Calidad",
    "gc": "Gestión de Calidad",
    # Ingenieria de Software
    "ingenieria de software": "Ingeniería de Software",
    "ingeniería de software": "Ingeniería de Software",
    "ingenieria software": "Ingeniería de Software",
    "ingeniería software": "Ingeniería de Software",
    "ing software": "Ingeniería de Software",
    "software": "Ingeniería de Software",
    "is": "Ingeniería de Software",
    # Backward-compat alias (legacy folder/code ISW)
    "isw": "Ingeniería de Software",
}

# Normalized alias map: normalized key -> canonical
_ALIASES: dict[str, str] = {}
for _raw_key, _canon in _RAW_ALIASES.items():
    _ALIASES[_normalize(_raw_key)] = _canon

# Build lower-case canonical lookup for O(1) validation (normalized too)
_CANONICAL_LOWER: dict[str, str] = {_normalize(c): c for c in MATERIAS_WHITELIST}


def resolve_materia(alias: str) -> str | None:
    """
    Pure helper: resolve user alias to canonical materia or None.

    Normalization: lower, accent-stripped (unicodedata), whitespace-collapsed.
    """
    if not isinstance(alias, str):
        return None
    key = _normalize(alias)
    if not key:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    if key in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[key]
    return None


def parse_message(text: str) -> tuple[str, str] | None:
    """
    Parse `MATERIA: pregunta` (split on first colon).

    Returns (materia_raw, question) trimmed, or None if no colon / empty parts.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    m = _MATERIA_PATTERN.match(stripped)
    if not m:
        return None
    materia_raw = m.group(1).strip()
    question = m.group(2).strip()
    if not materia_raw or not question:
        return None
    return (materia_raw, question)


def validate_materia(raw: str) -> str | None:
    """
    Validate raw materia against whitelist via alias resolution.

    Returns canonical whitelist entry or None if unknown.
    """
    return resolve_materia(raw)


def help_text() -> str:
    """
    Help reply for missing-colon case (byte-exact contract).

    Returns the exact missing-colon message required by WhatsApp interface.
    """
    return "❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?"


def unknown_materia_text(raw_materia: str) -> str:
    """Byte-exact unknown-materia message preserving user input as written."""
    return f'❌ No encontré la materia "{raw_materia.strip()}".'
