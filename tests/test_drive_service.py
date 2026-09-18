#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for drive_docs_service.py — exact block format, private docs, dedup."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import drive_docs_service
from drive_docs_service import (
    FALLBACK_DISPLAY,
    RAG_FALLBACK_PHRASE,
    _format_block,
    append_entry,
    get_or_create_doc,
)


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    fake_cache = tmp_path / "materias_docs.json"
    monkeypatch.setattr(drive_docs_service, "CACHE_PATH", fake_cache)
    drive_docs_service.reset_cache_for_tests()
    drive_docs_service.clear_dedup_for_tests()
    if fake_cache.exists():
        fake_cache.unlink()
    drive_docs_service._drive_service = None
    drive_docs_service._docs_service = None
    drive_docs_service._credentials = None
    yield
    drive_docs_service.reset_cache_for_tests()
    drive_docs_service.clear_dedup_for_tests()
    drive_docs_service._drive_service = None
    drive_docs_service._docs_service = None
    drive_docs_service._credentials = None


# --- exact block format ----------------------------------------------------

def test_format_block_exact_with_fuentes():
    block = _format_block(
        "2026-09-10T12:00:00+00:00",
        "¿Qué es una VLAN?",
        "Una red virtual.",
        [{"file": "redes.pdf", "page": "12"}, {"file": "otro.pdf", "page": ""}],
        numero=3,
    )
    assert block.startswith("--------------------------------------------------\n\nPregunta 3\n")
    assert "¿Qué es una VLAN?" in block
    assert "Respuesta:\n\nUna red virtual." in block
    assert "Fuentes utilizadas:" in block
    assert "- redes.pdf — página 12" in block
    # No page -> only file, never invent
    assert "- otro.pdf" in block
    assert "página " not in [l for l in block.splitlines() if l.strip() == "- otro.pdf"]
    assert "Fecha:\n" in block
    # DD/MM/YYYY HH:MM format
    import re

    assert re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", block)
    assert block.rstrip().endswith("--------------------------------------------------")


def test_format_block_fallback_saved():
    block = _format_block(
        "2026-09-10T12:00:00+00:00",
        "inexistente",
        RAG_FALLBACK_PHRASE,
        [],
        numero=1,
    )
    # Fallback IS saved in Docs verbatim
    assert RAG_FALLBACK_PHRASE in block
    assert FALLBACK_DISPLAY in block  # same value per new contract
    assert "Pregunta 1" in block
    assert "- Sin fuentes registradas" in block


def test_format_block_empty_respuesta_maps_to_fallback():
    block = _format_block("2026-09-10T12:00:00+00:00", "q?", "", [], numero=2)
    assert FALLBACK_DISPLAY in block
    assert "Pregunta 2" in block


def test_format_block_no_page_never_invents():
    block = _format_block("2026-09-10T12:00:00+00:00", "q?", "a", [{"file": "x.pdf", "page": "?"}], numero=1)
    assert "- x.pdf" in block
    # Must not contain invented page number
    for line in block.splitlines():
        if line.strip() == "- x.pdf":
            assert "página" not in line


# --- cache -----------------------------------------------------------------

def test_cache_save_and_load_cycle():
    from drive_docs_service import _load_cache, _save_cache

    data = {"Redes de Datos": "doc123", "Gestión de Calidad": "doc456"}
    _save_cache(data)
    cache_path = drive_docs_service.CACHE_PATH
    assert cache_path.exists()
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert raw == data
    drive_docs_service.reset_cache_for_tests()
    loaded = _load_cache()
    assert loaded == data


def test_cache_empty_file_seed():
    cache_path = drive_docs_service.CACHE_PATH
    cache_path.write_text("{}", encoding="utf-8")
    drive_docs_service.reset_cache_for_tests()
    loaded = drive_docs_service._load_cache()
    assert loaded == {}
    cache_path.unlink()
    drive_docs_service.reset_cache_for_tests()
    loaded2 = drive_docs_service._load_cache()
    assert loaded2 == {}


def test_cache_handles_invalid_json():
    cache_path = drive_docs_service.CACHE_PATH
    cache_path.write_text("{ not json", encoding="utf-8")
    drive_docs_service.reset_cache_for_tests()
    loaded = drive_docs_service._load_cache()
    assert loaded == {}


def test_cache_file_exists_gitignored():
    gitignore = Path(".gitignore")
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        assert "materias_docs.json" in content


# --- get_or_create_doc: private, exact title -------------------------------

def test_get_or_create_doc_cache_hit_no_api_called(monkeypatch):
    from drive_docs_service import _save_cache

    _save_cache({"Redes de Datos": "doc_cached_123"})
    drive_docs_service.reset_cache_for_tests()
    mock_build = MagicMock()
    monkeypatch.setattr("googleapiclient.discovery.build", mock_build)
    result = get_or_create_doc("Redes de Datos")
    assert result == "doc_cached_123"
    mock_build.assert_not_called()


def test_get_or_create_doc_creates_preguntas_title_and_private(monkeypatch):
    fake_creds = MagicMock()
    monkeypatch.setattr(drive_docs_service, "_get_credentials", lambda: fake_creds)
    monkeypatch.setattr(drive_docs_service, "_get_folder_id", lambda: "folderXYZ")

    mock_drive = MagicMock()
    mock_files = MagicMock()
    mock_create = MagicMock()
    mock_create.execute.return_value = {"id": "newdoc999"}
    mock_files.create.return_value = mock_create
    mock_drive.files.return_value = mock_files
    # permissions mock to assert NEVER called
    mock_perms = MagicMock()
    mock_drive.permissions.return_value = mock_perms

    with patch("googleapiclient.discovery.build", return_value=mock_drive):
        result = get_or_create_doc("Redes de Datos")
        assert result == "newdoc999"
        args, kwargs = mock_files.create.call_args
        body = kwargs.get("body") or (args[0] if args else {})
        assert body["name"] == "Redes de Datos - Preguntas"
        assert body["parents"] == ["folderXYZ"]
        assert body["mimeType"] == "application/vnd.google-apps.document"
        # Docs never public
        mock_perms.create.assert_not_called()
        assert not mock_drive.permissions.called
        cached = json.loads(drive_docs_service.CACHE_PATH.read_text(encoding="utf-8"))
        assert cached["Redes de Datos"] == "newdoc999"


def test_docs_never_public_all_materias(monkeypatch):
    """All 3 canon titles end with ' - Preguntas' and never call permissions."""
    for canon in ["Redes de Datos", "Gestión de Calidad", "Ingeniería de Software"]:
        drive_docs_service.reset_cache_for_tests()
        drive_docs_service._drive_service = None
        drive_docs_service._docs_service = None
        drive_docs_service._credentials = None
        # Clear cache file
        try:
            drive_docs_service.CACHE_PATH.unlink()
        except Exception:
            pass
        fake_creds = MagicMock()
        monkeypatch.setattr(drive_docs_service, "_get_credentials", lambda: fake_creds)
        monkeypatch.setattr(drive_docs_service, "_get_folder_id", lambda: "folderXYZ")
        mock_drive = MagicMock()
        mock_files = MagicMock()
        mock_create = MagicMock()
        mock_create.execute.return_value = {"id": f"id-{canon}"}
        mock_files.create.return_value = mock_create
        mock_drive.files.return_value = mock_files
        mock_perms = MagicMock()
        mock_drive.permissions.return_value = mock_perms
        with patch("googleapiclient.discovery.build", return_value=mock_drive):
            result = get_or_create_doc(canon)
            assert result == f"id-{canon}"
            body = mock_files.create.call_args[1].get("body")
            assert body["name"] == f"{canon} - Preguntas"
            mock_perms.create.assert_not_called()


# --- append_entry ----------------------------------------------------------

def _mock_docs_service_for_append(end_index=50, existing_text=""):
    mock_docs = MagicMock()
    mock_get = MagicMock()
    # Build minimal Docs structure with endIndex + optional text for numbering
    content = [{"endIndex": end_index}]
    if existing_text:
        # Simulate paragraph elements containing existing_text
        content = [
            {
                "endIndex": end_index,
                "paragraph": {"elements": [{"textRun": {"content": existing_text}}]},
            }
        ]
    mock_get.execute.return_value = {"body": {"content": content}}
    mock_docs.documents.return_value.get.return_value = mock_get
    mock_batch = MagicMock()
    mock_batch.execute.return_value = {}
    mock_docs.documents.return_value.batchUpdate.return_value = mock_batch
    return mock_docs, mock_batch


def test_append_entry_success_exact_format(monkeypatch):
    monkeypatch.setattr(drive_docs_service, "get_or_create_doc", lambda m: "doc123")
    mock_docs, mock_batch = _mock_docs_service_for_append(end_index=50)
    monkeypatch.setattr(drive_docs_service, "_get_docs_service", lambda: mock_docs)

    ok = append_entry(
        materia="Redes de Datos",
        fecha_iso="2026-09-10T12:00:00+00:00",
        pregunta="¿Qué es una VLAN?",
        respuesta="Una red virtual.",
        fuentes=[{"file": "redes.pdf", "page": "12"}],
        message_id="wamid.test1",
    )
    assert ok is True
    assert mock_batch.execute.called
    call_kwargs = mock_docs.documents.return_value.batchUpdate.call_args[1]
    assert call_kwargs["documentId"] == "doc123"
    requests = call_kwargs["body"]["requests"]
    assert requests[0]["insertText"]["location"]["index"] == 49
    text = requests[0]["insertText"]["text"]
    assert text.startswith("--------------------------------------------------")
    assert "Pregunta 1" in text
    assert "¿Qué es una VLAN?" in text
    assert "Respuesta:\n\nUna red virtual." in text
    assert "- redes.pdf — página 12" in text
    assert "Fecha:\n" in text


def test_append_entry_numbering_auto(monkeypatch):
    monkeypatch.setattr(drive_docs_service, "get_or_create_doc", lambda m: "doc123")
    existing = "Pregunta 1\n\nQ1\n\nPregunta 2\n\nQ2\n"
    mock_docs, mock_batch = _mock_docs_service_for_append(end_index=100, existing_text=existing)
    monkeypatch.setattr(drive_docs_service, "_get_docs_service", lambda: mock_docs)

    ok = append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q3?", "A3", [], "wamid.num1")
    assert ok is True
    text = mock_docs.documents.return_value.batchUpdate.call_args[1]["body"]["requests"][0]["insertText"]["text"]
    assert "Pregunta 3" in text


def test_append_entry_dedup_blocks_second_write(monkeypatch):
    monkeypatch.setattr(drive_docs_service, "get_or_create_doc", lambda m: "doc123")
    mock_docs, mock_batch = _mock_docs_service_for_append(end_index=100)
    monkeypatch.setattr(drive_docs_service, "_get_docs_service", lambda: mock_docs)

    mid = "wamid.dup1"
    ok1 = append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q?", "A", [], mid)
    assert ok1 is True
    assert mock_batch.execute.call_count == 1

    ok2 = append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q?", "A", [], mid)
    assert ok2 is True
    assert mock_batch.execute.call_count == 1


def test_append_entry_failure_does_not_mark_dedup(monkeypatch):
    monkeypatch.setattr(drive_docs_service, "get_or_create_doc", lambda m: "doc123")
    mock_docs = MagicMock()
    mock_get = MagicMock()
    mock_get.execute.return_value = {"body": {"content": [{"endIndex": 10}]}}
    mock_docs.documents.return_value.get.return_value = mock_get
    mock_batch = MagicMock()
    mock_batch.execute.side_effect = Exception("Drive down")
    mock_docs.documents.return_value.batchUpdate.return_value = mock_batch
    monkeypatch.setattr(drive_docs_service, "_get_docs_service", lambda: mock_docs)

    mid = "wamid.fail1"
    ok = append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q?", "A", [], mid)
    assert ok is False
    # Must NOT be marked, so retry is allowed (peek False)
    assert drive_docs_service.is_already_processed(mid) is False


def test_append_entry_fallback_saved_with_block(monkeypatch):
    monkeypatch.setattr(drive_docs_service, "get_or_create_doc", lambda m: "docX")
    mock_docs, mock_batch = _mock_docs_service_for_append(end_index=10)
    monkeypatch.setattr(drive_docs_service, "_get_docs_service", lambda: mock_docs)

    ok = append_entry(
        materia="Redes de Datos",
        fecha_iso="2026-09-10T12:00:00+00:00",
        pregunta="inexistente?",
        respuesta=RAG_FALLBACK_PHRASE,
        fuentes=[],
        message_id="wamid.fallback1",
    )
    assert ok is True
    text = mock_docs.documents.return_value.batchUpdate.call_args[1]["body"]["requests"][0]["insertText"]["text"]
    assert RAG_FALLBACK_PHRASE in text
    assert "Fuentes utilizadas:" in text


# --- locks -----------------------------------------------------------------

def test_per_materia_locks_distinct():
    lock_a = drive_docs_service._get_materia_lock("Redes de Datos")
    lock_b = drive_docs_service._get_materia_lock("Gestión de Calidad")
    lock_a2 = drive_docs_service._get_materia_lock("Redes de Datos")
    assert lock_a is not lock_b
    assert lock_a is lock_a2
    assert isinstance(lock_a, type(threading.Lock()))


def test_lru_eviction_drive_dedup():
    original_max = drive_docs_service._DEDUP_MAX
    drive_docs_service._DEDUP_MAX = 5
    try:
        drive_docs_service.clear_dedup_for_tests()
        for i in range(5):
            assert drive_docs_service._is_duplicate(f"id{i}") is False
        assert drive_docs_service._is_duplicate("id5") is False
        assert drive_docs_service._is_duplicate("id0") is False
        assert drive_docs_service._is_duplicate("id2") is True
    finally:
        drive_docs_service._DEDUP_MAX = original_max
        drive_docs_service.clear_dedup_for_tests()
