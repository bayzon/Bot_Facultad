#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for materias_config.py — 3 active materias, aliases, normalization."""

import materias_config
from materias_config import help_text, parse_message, resolve_materia, validate_materia


# --- parse_message ---------------------------------------------------------

def test_parse_valid_basic():
    assert parse_message("Redes de Datos: ¿Qué es una VLAN?") == ("Redes de Datos", "¿Qué es una VLAN?")


def test_parse_trims_spaces():
    assert parse_message("  redes:   ¿Qué es una VLAN?  ") == ("redes", "¿Qué es una VLAN?")


def test_parse_preserves_colon_in_question():
    assert parse_message("Redes: teorema: VLAN?") == ("Redes", "teorema: VLAN?")


def test_parse_no_colon_returns_none():
    assert parse_message("hola sin colon") is None


def test_parse_empty_materia_returns_none():
    assert parse_message(": pregunta") is None


def test_parse_empty_question_returns_none():
    assert parse_message("Redes:") is None
    assert parse_message("Redes:   ") is None


def test_parse_empty_string_returns_none():
    assert parse_message("") is None
    assert parse_message("   ") is None


def test_parse_non_string_returns_none():
    assert parse_message(None) is None  # type: ignore
    assert parse_message(123) is None  # type: ignore


def test_parse_regex_constant():
    assert materias_config.MATERIA_REGEX == r"^([^:]+):\s*(.+)$"


# --- whitelist: only 3 active ------------------------------------------------

def test_whitelist_only_three_active():
    assert materias_config.MATERIAS_WHITELIST == [
        "Redes de Datos",
        "Gestión de Calidad",
        "Ingeniería de Software",
    ]


# --- validate_materia / resolve_materia -------------------------------------

def test_validate_redes_aliases():
    assert validate_materia("redes") == "Redes de Datos"
    assert validate_materia("red") == "Redes de Datos"
    assert validate_materia("Redes de Datos") == "Redes de Datos"
    assert validate_materia("redes datos") == "Redes de Datos"
    assert validate_materia("rd") == "Redes de Datos"
    assert validate_materia("  REDES  ") == "Redes de Datos"


def test_validate_calidad_aliases():
    assert validate_materia("gestion de calidad") == "Gestión de Calidad"
    assert validate_materia("gestión de calidad") == "Gestión de Calidad"
    assert validate_materia("calidad") == "Gestión de Calidad"
    assert validate_materia("gestion calidad") == "Gestión de Calidad"
    assert validate_materia("gestión calidad") == "Gestión de Calidad"
    assert validate_materia("gc") == "Gestión de Calidad"
    assert validate_materia("CALIDAD") == "Gestión de Calidad"


def test_validate_software_aliases():
    assert validate_materia("ingenieria de software") == "Ingeniería de Software"
    assert validate_materia("ingeniería de software") == "Ingeniería de Software"
    assert validate_materia("ingenieria software") == "Ingeniería de Software"
    assert validate_materia("ingeniería software") == "Ingeniería de Software"
    assert validate_materia("ing software") == "Ingeniería de Software"
    assert validate_materia("software") == "Ingeniería de Software"
    assert validate_materia("is") == "Ingeniería de Software"


def test_validate_normalization_accents_spaces():
    # Accent-insensitive + whitespace-collapsed
    assert validate_materia("  gestión   de   calidad  ") == "Gestión de Calidad"
    assert validate_materia("INGENIERÍA DE SOFTWARE") == "Ingeniería de Software"
    assert resolve_materia("Redes") == "Redes de Datos"


def test_validate_unknown_returns_none():
    assert validate_materia("Matemática Superior") is None
    assert validate_materia("AM2") is None
    assert validate_materia("Fisica") is None
    assert validate_materia("Fisica2") is None
    assert validate_materia("Desconocida") is None
    assert validate_materia("") is None
    assert validate_materia("   ") is None


def test_validate_non_string_returns_none():
    assert validate_materia(None) is None  # type: ignore
    assert validate_materia(123) is None  # type: ignore
    assert resolve_materia(None) is None  # type: ignore


# --- help_text: byte-exact missing-colon message ----------------------------

def test_help_text_missing_colon_exact():
    txt = help_text()
    assert txt == "❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?"


def test_unknown_materia_text_exact():
    txt = materias_config.unknown_materia_text("Matemática Superior")
    assert txt == '❌ No encontré la materia "Matemática Superior".'
