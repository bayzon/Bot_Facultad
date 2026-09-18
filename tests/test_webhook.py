#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integration tests for webhook_server.py — interface-only WhatsApp contract."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import webhook_server
from webhook_server import app, clear_lru_for_tests

try:
    import rag_core

    FALLBACK = rag_core.FALLBACK_PHRASE
except Exception:
    FALLBACK = "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad."

SUCCESS = "✅ Pregunta guardada y respondida"
MISSING_COLON = "❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?"
DOCS_FAIL = "❌ No pude guardar la pregunta. Intentá nuevamente."


@pytest.fixture
def client():
    clear_lru_for_tests()
    try:
        import drive_docs_service

        drive_docs_service.clear_dedup_for_tests()
    except Exception:
        pass
    with TestClient(app) as c:
        yield c
    clear_lru_for_tests()


@pytest.fixture(autouse=True)
def env_verify_token(monkeypatch):
    monkeypatch.setenv("VERIFY_TOKEN", "test_verify_token_pr1_foundation")
    yield


def _whatsapp_payload(message_id="wamid.test123", from_number="5491123456789", body="Redes de Datos: ¿Qué es una VLAN?", msg_type="text"):
    msg = {
        "from": from_number,
        "id": message_id,
        "timestamp": "1234567890",
        "type": msg_type,
    }
    if msg_type == "text":
        msg["text"] = {"body": body}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "entry1",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "5491123456789", "phone_number_id": "123456"},
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": from_number}],
                            "messages": [msg],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


# --- GET /webhook ------------------------------------------------------------

def test_get_verify_valid_returns_challenge(client):
    resp = client.get("/webhook?hub.mode=subscribe&hub.verify_token=test_verify_token_pr1_foundation&hub.challenge=12345")
    assert resp.status_code == 200
    assert resp.text == "12345"


def test_get_verify_invalid_token_rejected(client):
    resp = client.get("/webhook?hub.mode=subscribe&hub.verify_token=wrong_token&hub.challenge=12345")
    assert resp.status_code == 403


# --- POST: success exact + never academic ------------------------------------

def test_post_success_exact_and_never_academic(client):
    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = ("Una red virtual es...", [{"file": "redes.pdf", "page": "12"}])
    mock_rag.FALLBACK_PHRASE = FALLBACK
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = True
    mock_drive.is_already_processed.return_value = False
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True

    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        payload = _whatsapp_payload(message_id="wamid.ok1", body="Redes de Datos: ¿Qué es una VLAN?")
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        assert mock_drive.append_entry.called
        assert mock_whatsapp.send_text.called
        body = mock_whatsapp.send_text.call_args[0][1]
        assert body == SUCCESS
        # Never academic
        assert "Una red virtual" not in body
        assert "redes.pdf" not in body
        assert "página" not in body
        assert "http" not in body.lower()


def test_post_fallback_still_success_not_fallback_in_whatsapp(client):
    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = (FALLBACK, [])
    mock_rag.FALLBACK_PHRASE = FALLBACK
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = True
    mock_drive.is_already_processed.return_value = False
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True

    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        payload = _whatsapp_payload(message_id="wamid.fb1", body="Redes de Datos: pregunta inexistente xyz")
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        # Docs still saves fallback Q+A
        assert mock_drive.append_entry.called
        call = mock_drive.append_entry.call_args
        kwargs = call[1] if call[1] else {}
        respuesta = kwargs.get("respuesta", call[0][3] if len(call[0]) > 3 else "")
        assert respuesta == FALLBACK
        # WhatsApp still success, never fallback text
        body = mock_whatsapp.send_text.call_args[0][1]
        assert body == SUCCESS
        assert FALLBACK not in body


def test_post_docs_fail_sends_error_never_success(client):
    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = ("Respuesta", [{"file": "a.pdf", "page": "1"}])
    mock_rag.FALLBACK_PHRASE = FALLBACK
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = False
    mock_drive.is_already_processed.return_value = False
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True

    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        payload = _whatsapp_payload(message_id="wamid.fail1", body="Redes de Datos: ¿Qué es una VLAN?")
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        body = mock_whatsapp.send_text.call_args[0][1]
        assert body == DOCS_FAIL
        assert body != SUCCESS


def test_post_invalid_payload_200_skip(client):
    mock_rag = MagicMock()
    mock_drive = MagicMock()
    mock_whatsapp = MagicMock()
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        resp = client.post("/webhook", json={})
        assert resp.status_code == 200
        time.sleep(0.05)
        assert not mock_drive.append_entry.called


# --- Dedup -------------------------------------------------------------------

def test_duplicate_message_id_single_write_lru(client):
    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = ("Respuesta", [{"file": "a.pdf", "page": "1"}])
    mock_rag.FALLBACK_PHRASE = FALLBACK
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = True
    mock_drive.is_already_processed.return_value = False
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True

    dup_id = "wamid.duplicate123"
    payload = _whatsapp_payload(message_id=dup_id, body="Redes de Datos: ¿Qué es una VLAN?")

    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        resp1 = client.post("/webhook", json=payload)
        assert resp1.json()["enqueued"] == 1
        time.sleep(0.05)
        resp2 = client.post("/webhook", json=payload)
        assert resp2.json()["enqueued"] == 0
        time.sleep(0.05)
        assert mock_drive.append_entry.call_count == 1


def test_lru_eviction_1000_to_1001(client):
    from webhook_server import _seen_ids, _seen_lock, LRU_MAX

    assert LRU_MAX == 1000
    clear_lru_for_tests()
    with _seen_lock:
        for i in range(1000):
            _seen_ids[f"id_{i}"] = None
        assert len(_seen_ids) == 1000
    from webhook_server import _is_duplicate_lru

    assert _is_duplicate_lru("id_1000") is False
    with _seen_lock:
        assert "id_0" not in _seen_ids
        assert len(_seen_ids) == 1000
    assert _is_duplicate_lru("id_0") is False
    assert _is_duplicate_lru("id_0") is True


# --- Validation: unknown / missing colon / non-text --------------------------

def test_unknown_materia_exact_no_rag_drive(client):
    mock_rag = MagicMock()
    mock_drive = MagicMock()
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True

    payload = _whatsapp_payload(message_id="wamid.unknown1", body="Matemática Superior: ¿Qué es?")
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        # No RAG/Docs (check both legacy query and new qfm not called)
        assert not mock_rag.query_for_materia.called
        assert not mock_drive.append_entry.called
        assert mock_whatsapp.send_text.called
        assert mock_whatsapp.send_text.call_args[0][1] == '❌ No encontré la materia "Matemática Superior".'


def test_extra_materia_am2_rejected(client):
    mock_rag = MagicMock()
    mock_drive = MagicMock()
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True
    payload = _whatsapp_payload(message_id="wamid.extra1", body="AM2: Rolle?")
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        assert not mock_drive.append_entry.called
        assert mock_whatsapp.send_text.call_args[0][1] == '❌ No encontré la materia "AM2".'


def test_no_colon_exact_help_no_rag_drive(client):
    mock_rag = MagicMock()
    mock_drive = MagicMock()
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True
    payload = _whatsapp_payload(message_id="wamid.nocolon1", body="hola sin colon")
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        assert not mock_drive.append_entry.called
        assert mock_whatsapp.send_text.call_args[0][1] == MISSING_COLON


def test_non_text_ignored_without_rag_docs(client):
    mock_rag = MagicMock()
    mock_drive = MagicMock()
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True
    payload = _whatsapp_payload(message_id="wamid.img1", body="", msg_type="image")
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        assert not mock_drive.append_entry.called
        # Help without RAG/Docs
        assert mock_whatsapp.send_text.called


def test_alias_redes_routes_to_canon(client):
    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = ("R", [{"file": "f.pdf", "page": "1"}])
    mock_rag.FALLBACK_PHRASE = FALLBACK
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = True
    mock_drive.is_already_processed.return_value = False
    mock_whatsapp = MagicMock()
    mock_whatsapp.send_text.return_value = True

    payload = _whatsapp_payload(message_id="wamid.alias1", body="redes: ¿Qué es una VLAN?")
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_whatsapp):
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        time.sleep(0.05)
        assert mock_rag.query_for_materia.called
        assert mock_rag.query_for_materia.call_args[0][0] == "Redes de Datos"
        assert mock_rag.query_for_materia.call_args[0][1] == "¿Qué es una VLAN?"


# --- Health ------------------------------------------------------------------

def test_health_200_when_loaded(monkeypatch, client):
    monkeypatch.setattr(webhook_server, "_query_engine", MagicMock())
    monkeypatch.setattr(webhook_server, "_index_error", None)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_503_when_missing_chroma(monkeypatch, client):
    monkeypatch.setattr(webhook_server, "_query_engine", None)
    monkeypatch.setattr(webhook_server, "_index_error", "503: indice no disponible — ejecutar con --reindex")
    resp = client.get("/health")
    assert resp.status_code == 503
