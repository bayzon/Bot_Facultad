#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
whatsapp_client.py — WhatsApp Cloud API sender for Bot Facultad.

Sends text messages via Meta Graph API:
  POST https://graph.facebook.com/{WHATSAPP_GRAPH_VERSION}/{phone_number_id}/messages
  (default v23.0, configurable via WHATSAPP_GRAPH_VERSION env — horariumTest parity)

Env contract (from .env):
  WHATSAPP_TOKEN            — Bearer token (EAA...)
  WHATSAPP_PHONE_NUMBER_ID  — Phone number ID for the URL path
  WHATSAPP_GRAPH_VERSION    — Graph API version (default v23.0)

Design: stateless helper, sync httpx.Client, returns bool, logs on error.
No exceptions propagate to caller — webhook's BackgroundTasks must not crash
on send failure.

Usage:
  from whatsapp_client import send_text
  ok = send_text("5491123456789", "Hola desde el bot")
"""

from __future__ import annotations

import os
import logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import httpx
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "httpx not installed — pip install httpx (see requirements.txt)"
    ) from e

logger = logging.getLogger(__name__)

GRAPH_VERSION = os.getenv("WHATSAPP_GRAPH_VERSION", "v23.0").strip() or "v23.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"
TIMEOUT_SECONDS = 10.0


def _get_graph_base() -> str:
    """Dynamic GRAPH_BASE — reads WHATSAPP_GRAPH_VERSION env on each call (horariumTest v23.0 parity)."""
    ver = os.getenv("WHATSAPP_GRAPH_VERSION", "v23.0").strip() or "v23.0"
    return f"https://graph.facebook.com/{ver}"


def _get_token_and_phone_id(
    token: str | None = None,
    phone_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve token/phone_id from args or env. Returns (token, phone_id)."""
    t = token if token is not None else os.getenv("WHATSAPP_TOKEN", "").strip()
    pid = (
        phone_id
        if phone_id is not None
        else os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    )
    # Normalize empty to None for caller checks
    if not t:
        t = None  # type: ignore
    if not pid:
        pid = None  # type: ignore
    return t, pid  # type: ignore


def send_text(
    to: str,
    body: str,
    token: str | None = None,
    phone_id: str | None = None,
) -> bool:
    """
    Send a WhatsApp text message via Cloud API.

    Args:
        to: Recipient in WhatsApp format, digits with country code
            (e.g. "5491123456789"). The Graph API accepts with or without '+';
            we strip leading '+' and spaces.
        body: Message text (1..4096 chars recommended). Empty is rejected.
        token: Optional override for WHATSAPP_TOKEN (for tests).
        phone_id: Optional override for WHATSAPP_PHONE_NUMBER_ID (for tests).

    Returns:
        True on 2xx from Graph API, False otherwise (logged).

    Contract:
        - Uses httpx POST {GRAPH_BASE}/{phone_id}/messages
        - Header Authorization: Bearer {token}
        - JSON: {messaging_product:"whatsapp", to, type:"text", text:{body}}
        - Errors are logged, never raised.
    """
    # Fail fast on bad input — recoverable, return False
    if not isinstance(to, str) or not to.strip():
        logger.warning("[whatsapp] send_text: empty/invalid 'to' — skip")
        return False
    if not isinstance(body, str) or not body.strip():
        logger.warning("[whatsapp] send_text: empty body — skip")
        return False

    clean_to = to.strip().lstrip("+").replace(" ", "").replace("-", "")
    if not clean_to:
        logger.warning("[whatsapp] send_text: 'to' became empty after clean — skip")
        return False

    resolved_token, resolved_phone_id = _get_token_and_phone_id(token, phone_id)

    if not resolved_token:
        logger.error(
            "[whatsapp] WHATSAPP_TOKEN missing — set WHATSAPP_TOKEN in .env "
            "(or pass token=). Message not sent to %s",
            clean_to,
        )
        return False
    if not resolved_phone_id:
        logger.error(
            "[whatsapp] WHATSAPP_PHONE_NUMBER_ID missing — set "
            "WHATSAPP_PHONE_NUMBER_ID in .env (or pass phone_id=). "
            "Message not sent to %s",
            clean_to,
        )
        return False

    url = f"{_get_graph_base()}/{resolved_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {resolved_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": clean_to,
        "type": "text",
        "text": {"body": body},
    }

    try:
        # Sync client per design (not Async). Use context manager for cleanup.
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            resp = client.post(url, headers=headers, json=payload)
        if 200 <= resp.status_code < 300:
            logger.info(
                "[whatsapp] sent to %s (status=%s, message_id=%s)",
                clean_to,
                resp.status_code,
                _extract_message_id(resp),
            )
            return True
        # Non-2xx — log body for debugging (token is not in body)
        logger.error(
            "[whatsapp] Graph API error to=%s status=%s body=%.500s url=%s",
            clean_to,
            resp.status_code,
            resp.text,
            url,
        )
        return False
    except httpx.TimeoutException as e:
        logger.error("[whatsapp] timeout sending to %s: %s", clean_to, e)
        return False
    except httpx.NetworkError as e:
        logger.error("[whatsapp] network error to %s: %s", clean_to, e)
        return False
    except httpx.HTTPError as e:
        # Catch-all for httpx (covers RequestError, etc.)
        logger.error("[whatsapp] httpx error to %s: %s", clean_to, e)
        return False
    except Exception as e:  # pragma: no cover — defensive
        logger.exception("[whatsapp] unexpected error to %s: %s", clean_to, e)
        return False


def _extract_message_id(resp: httpx.Response) -> str:
    """Best-effort extract wamid message id from Graph response JSON."""
    try:
        data = resp.json()
        # Graph returns {"messages":[{"id":"wamid...."}]}
        msgs = data.get("messages") or []
        if msgs and isinstance(msgs[0], dict):
            return str(msgs[0].get("id", ""))
        return ""
    except Exception:
        return ""


__all__ = ["send_text", "GRAPH_BASE", "GRAPH_VERSION"]
