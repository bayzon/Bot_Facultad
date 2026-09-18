#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for whatsapp_client.py — httpx mock Client success/failure/timeout/400.
"""

import httpx
from unittest.mock import MagicMock, patch

import whatsapp_client
from whatsapp_client import GRAPH_BASE, send_text


def _mock_response(status_code=200, json_data=None, text="ok"):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.text = text
    if json_data is not None:
        mock_resp.json.return_value = json_data
    else:
        mock_resp.json.side_effect = Exception("no json")
    return mock_resp


def _mock_client_context(mock_response=None, side_effect=None):
    """Return a mocked httpx.Client context manager that yields a client with post()."""
    mock_client = MagicMock()
    if side_effect:
        mock_client.post.side_effect = side_effect
    else:
        mock_client.post.return_value = mock_response
    # Context manager: __enter__ returns mock_client, __exit__ None
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_client
    mock_ctx.__exit__.return_value = False
    # Patch httpx.Client to return mock_ctx
    return mock_ctx, mock_client


def test_send_text_success_calls_correct_url_and_payload(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TEST_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    fake_resp = _mock_response(
        200, json_data={"messages": [{"id": "wamid.test123"}]}, text='{"messages":[{"id":"wamid.test123"}]}'
    )
    mock_ctx, mock_client = _mock_client_context(fake_resp)
    with patch("whatsapp_client.httpx.Client", return_value=mock_ctx):
        ok = send_text("5491123456789", "Hola desde test")
        assert ok is True
        # Verify URL
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        url = args[0] if args else kwargs.get("url")
        assert url == f"{GRAPH_BASE}/123456789012345/messages"
        # Verify headers contain Bearer token
        headers = kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer EAA_TEST_TOKEN"
        assert headers.get("Content-Type") == "application/json"
        # Verify JSON payload
        payload = kwargs.get("json", {})
        assert payload["messaging_product"] == "whatsapp"
        assert payload["to"] == "5491123456789"
        assert payload["type"] == "text"
        assert payload["text"]["body"] == "Hola desde test"


def test_send_text_missing_token_returns_false(monkeypatch):
    monkeypatch.delenv("WHATSAPP_TOKEN", raising=False)
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    # Should not even create client — return False quickly
    with patch("whatsapp_client.httpx.Client") as mock_cls:
        ok = send_text("5491123456789", "Hola")
        assert ok is False
        mock_cls.assert_not_called()


def test_send_text_missing_phone_id_returns_false(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TOKEN")
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    with patch("whatsapp_client.httpx.Client") as mock_cls:
        ok = send_text("5491123456789", "Hola")
        assert ok is False
        mock_cls.assert_not_called()


def test_send_text_empty_to_returns_false(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    with patch("whatsapp_client.httpx.Client") as mock_cls:
        assert send_text("", "Hola") is False
        assert send_text("   ", "Hola") is False
        assert send_text(None, "Hola") is False  # type: ignore
        mock_cls.assert_not_called()


def test_send_text_empty_body_returns_false(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    with patch("whatsapp_client.httpx.Client") as mock_cls:
        assert send_text("5491123456789", "") is False
        assert send_text("5491123456789", "   ") is False
        mock_cls.assert_not_called()


def test_send_text_timeout_returns_false(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    mock_ctx, mock_client = _mock_client_context(side_effect=httpx.TimeoutException("timeout"))
    # Need to patch Client to return ctx that raises on post
    with patch("whatsapp_client.httpx.Client", return_value=mock_ctx):
        ok = send_text("5491123456789", "Hola con timeout")
        assert ok is False


def test_send_text_network_error_returns_false(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    mock_ctx, _ = _mock_client_context(side_effect=httpx.NetworkError("network down"))
    with patch("whatsapp_client.httpx.Client", return_value=mock_ctx):
        ok = send_text("5491123456789", "Hola network")
        assert ok is False


def test_send_text_400_error_returns_false(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    fake_resp = _mock_response(400, json_data={"error": {"message": "Bad Request"}}, text="Bad Request")
    mock_ctx, _ = _mock_client_context(fake_resp)
    with patch("whatsapp_client.httpx.Client", return_value=mock_ctx):
        ok = send_text("5491123456789", "Hola 400")
        assert ok is False


def test_send_text_cleans_to_format(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "EAA_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789012345")
    fake_resp = _mock_response(200, json_data={"messages": [{"id": "wamid.x"}]}, text="ok")
    mock_ctx, mock_client = _mock_client_context(fake_resp)
    with patch("whatsapp_client.httpx.Client", return_value=mock_ctx):
        ok = send_text("+54 911 1234-5678", "Hola limpieza")
        assert ok is True
        payload = mock_client.post.call_args[1]["json"]
        # Should strip +, spaces, dashes
        assert payload["to"] == "5491112345678"
        # Also test with plus and spaces variant
        mock_client.post.reset_mock()
        ok2 = send_text("5491123456789", "Hola2")
        assert ok2 is True
        assert mock_client.post.call_args[1]["json"]["to"] == "5491123456789"


def test_send_text_override_token_phone_id(monkeypatch):
    # Ensure override args bypass env
    monkeypatch.setenv("WHATSAPP_TOKEN", "ENV_TOKEN")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "ENV_ID")
    fake_resp = _mock_response(200, json_data={"messages": [{"id": "wamid.y"}]}, text="ok")
    mock_ctx, mock_client = _mock_client_context(fake_resp)
    with patch("whatsapp_client.httpx.Client", return_value=mock_ctx):
        ok = send_text("5491123456789", "Hola override", token="OVERRIDE_TOKEN", phone_id="999999")
        assert ok is True
        assert mock_client.post.call_args[0][0] == f"{GRAPH_BASE}/999999/messages"
        assert mock_client.post.call_args[1]["headers"]["Authorization"] == "Bearer OVERRIDE_TOKEN"
