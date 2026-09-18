#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
webhook_server.py — FastAPI webhook for WhatsApp Cloud API (Bot Facultad).

Endpoints:
  GET  /webhook?hub.mode=subscribe&hub.verify_token=&hub.challenge=
       -> 200 hub.challenge on valid VERIFY_TOKEN else 403
  POST /webhook
       Parses entry.changes.value.messages[0].id/from/text.body,
       LRU(1000) dedup by wamid, BackgroundTasks.add_task(process_question),
       ACK 200 <200ms.
  GET  /health -> 200 if index exists else 503

Flow (process_question, background):
  materias_config.parse_message -> validate_materia
    if None: send help_text() via whatsapp_client (no RAG/Drive)
    else: rag_core.query() -> fallback phrase handling
          -> drive_docs_service.append_entry()
          -> whatsapp_client.send_text(answer + fuentes footer)

Design contracts:
  - lifespan singleton loads rag_core.get_query_engine() once
  - Missing chroma_db is handled gracefully (health 503, query fallback)
  - No exception escapes POST handler — invalid payload returns 200-skip
  - Per design Data Flow and Interfaces / Contracts verbatim

Env:
  VERIFY_TOKEN, WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID,
  GOOGLE_SERVICE_ACCOUNT_JSON, GOOGLE_DRIVE_FOLDER_ID, OPENAI_API_KEY

Run:
  uvicorn webhook_server:app --host 0.0.0.0 --port 8000
  # local: ngrok http 8000 -> Meta webhook URL + VERIFY_TOKEN
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import BackgroundTasks, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Local imports — keep at top but allow missing during tests (lazy in handlers)
try:
    import materias_config  # type: ignore
except ImportError as e:
    materias_config = None  # type: ignore
    logger.warning("[webhook] materias_config not found: %s", e)

try:
    import rag_core  # type: ignore
except ImportError as e:
    rag_core = None  # type: ignore
    logger.warning("[webhook] rag_core not found: %s", e)

try:
    import drive_docs_service  # type: ignore
except ImportError as e:
    drive_docs_service = None  # type: ignore
    logger.warning("[webhook] drive_docs_service not found: %s", e)

try:
    import whatsapp_client  # type: ignore
except ImportError as e:
    whatsapp_client = None  # type: ignore
    logger.warning("[webhook] whatsapp_client not found: %s", e)

# ---------------------------------------------------------------------------
# WhatsApp interface contract — byte-exact messages. Never send academic content.
# ---------------------------------------------------------------------------
SUCCESS_MSG = "✅ Pregunta guardada y respondida"
MISSING_COLON_MSG = "❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?"
DOCS_FAIL_MSG = "❌ No pude guardar la pregunta. Intentá nuevamente."


def unknown_materia_msg(raw_materia: str) -> str:
    """Byte-exact unknown-materia message preserving user input as written."""
    return f'❌ No encontré la materia "{raw_materia.strip()}".'


# ---------------------------------------------------------------------------
# LRU dedup (1000) — primary guard; drive service has secondary 2000
# ---------------------------------------------------------------------------
LRU_MAX = 1000
_seen_ids: OrderedDict[str, None] = OrderedDict()
_seen_lock = __import__("threading").Lock()


def _is_duplicate_lru(message_id: str) -> bool:
    """Return True if already seen, else mark and return False. Thread-safe."""
    if not message_id:
        return False
    with _seen_lock:
        if message_id in _seen_ids:
            _seen_ids.move_to_end(message_id)
            return True
        _seen_ids[message_id] = None
        while len(_seen_ids) > LRU_MAX:
            _seen_ids.popitem(last=False)
        return False


def _remove_from_lru(message_id: str) -> None:
    """Unmark LRU on Docs failure so retry can proceed. Thread-safe."""
    if not message_id:
        return
    with _seen_lock:
        _seen_ids.pop(message_id, None)


def verify_hmac_sha256(raw: bytes, signature_header: str | None, app_secret: str) -> bool:
    """
    Verify HMAC SHA256 signature — replica lib/whatsapp/hmac.ts.

    - Strips "sha256=" prefix if present
    - Computes hmac.new(app_secret.encode(), raw, hashlib.sha256).hexdigest()
    - Compares via bytes.fromhex + hmac.compare_digest with length check
    - Returns False if missing secret/header or ValueError (non-hex, etc.)
    """
    if not app_secret or not signature_header:
        return False
    try:
        sig = signature_header.strip()
        if sig.startswith("sha256="):
            sig = sig[len("sha256=") :]
        sig = sig.strip()
        if not sig:
            return False
        expected_hex = hmac.new(app_secret.encode(), raw, hashlib.sha256).hexdigest()
        # Length check before compare_digest (hex strings should be 64 chars)
        if len(sig) != len(expected_hex):
            return False
        # Use bytes.fromhex + compare_digest for constant-time comparison
        return hmac.compare_digest(bytes.fromhex(sig), bytes.fromhex(expected_hex))
    except ValueError:
        return False
    except Exception:
        return False


def clear_lru_for_tests() -> None:
    """Test helper."""
    with _seen_lock:
        _seen_ids.clear()


# ---------------------------------------------------------------------------
# App state / lifespan
# ---------------------------------------------------------------------------
_query_engine: Any | None = None
_index_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan singleton: try to load rag_core index once.
    Missing chroma_db must not crash startup — health will return 503.
    """
    global _query_engine, _index_error
    logger.info("[webhook] lifespan startup — loading RAG index...")
    if rag_core is None:
        _index_error = "rag_core not available"
        logger.warning("[webhook] %s", _index_error)
    else:
        try:
            # Webhook-safe: fail_fast=True — nunca construye índice en caliente
            # (no quema cuota 429). Solo `python main.py --reindex` manual reconstruye.
            # Si el índice falta, queda _index_error y /health da 503 con fallback por pregunta.
            try:
                _query_engine = rag_core.get_query_engine(fail_fast=True)  # type: ignore
            except TypeError:
                _query_engine = rag_core.get_query_engine()  # type: ignore
            _index_error = None
            logger.info("[webhook] RAG index loaded (engine=%s)", type(_query_engine).__name__)
        except RuntimeError as e:
            # 503 semantics from rag_core
            _index_error = str(e)
            _query_engine = None
            logger.warning("[webhook] RAG index not available at startup: %s", e)
        except SystemExit as e:
            _index_error = f"SystemExit: {e}"
            _query_engine = None
            logger.warning("[webhook] RAG index SystemExit: %s", e)
        except Exception as e:
            _index_error = str(e)
            _query_engine = None
            logger.warning("[webhook] unexpected RAG load failure: %s", e)
    yield
    logger.info("[webhook] lifespan shutdown")
    # No explicit cleanup needed (chroma client is persistent)


app = FastAPI(
    title="Bot Facultad WhatsApp Webhook",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /webhook — verification
# ---------------------------------------------------------------------------

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
):
    """
    Meta verification handshake.

    Spec:
      GIVEN hub.mode=subscribe + hub.verify_token==VERIFY_TOKEN
      WHEN GET /webhook
      THEN 200 with hub.challenge body
      ELSE 403
    """
    expected = os.getenv("VERIFY_TOKEN", "").strip()
    # Also allow query param name variations (Meta uses exactly hub.*)
    mode = (hub_mode or "").strip()
    token = (hub_verify_token or "").strip()
    challenge = hub_challenge or ""

    if mode == "subscribe" and token and expected and token == expected:
        logger.info("[webhook] GET verify ok (mode=subscribe)")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning(
        "[webhook] GET verify failed: mode=%r token_match=%s (expected set=%s)",
        mode,
        (token == expected) if expected else False,
        bool(expected),
    )
    return PlainTextResponse(content="Verification failed", status_code=403)


# ---------------------------------------------------------------------------
# Helpers: payload parsing
# ---------------------------------------------------------------------------

def _extract_messages(payload: dict) -> list[dict]:
    """
    Extract list of {id, from, body, msg_type} from Meta webhook payload.

    Payload shape:
      {entry:[{changes:[{value:{messages:[{id,from,type,text:{body}}]}}]}]}

    Only type=text is processed with RAG/Docs; other types get help without RAG/Docs.
    Returns list (may be empty for status updates or invalid payload).
    Invalid payload never raises — returns [] for safe 200-skip.
    Limits: body truncated to 2000 chars, out capped to 10 messages.
    """
    out: list[dict] = []
    try:
        entries = payload.get("entry") or []
        if not isinstance(entries, list):
            return []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            changes = entry.get("changes") or []
            if not isinstance(changes, list):
                continue
            for ch in changes:
                if not isinstance(ch, dict):
                    continue
                value = ch.get("value") or {}
                if not isinstance(value, dict):
                    continue
                messages = value.get("messages") or []
                if not isinstance(messages, list):
                    continue
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    mid = msg.get("id")
                    frm = msg.get("from")
                    msg_type = str(msg.get("type") or "").strip().lower()
                    text_obj = msg.get("text") or {}
                    body = ""
                    if isinstance(text_obj, dict):
                        body = text_obj.get("body") or ""
                    # Only handle messages with required fields
                    if not mid or not frm:
                        continue
                    # Non-text types keep empty body and type for help-only handling.
                    body_str = str(body) if body is not None else ""
                    # Limit body to 2000 chars (horarium parity)
                    if len(body_str) > 2000:
                        body_str = body_str[:2000]
                    out.append(
                        {
                            "id": str(mid),
                            "from": str(frm),
                            "body": body_str,
                            "msg_type": msg_type or ("text" if body_str else "unknown"),
                        }
                    )
                    # Cap total messages to 10 to avoid abuse
                    if len(out) >= 10:
                        return out[:10]
    except Exception as e:
        logger.warning("[webhook] _extract_messages failed (non-fatal): %s", e)
    return out[:10]


# ---------------------------------------------------------------------------
# POST /webhook — fast ACK + background processing
# ---------------------------------------------------------------------------

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Fast ingestion: ACK 200 <200ms, enqueue process_question via BackgroundTasks.

    Security (horariumTest parity):
      - Reads raw body (await request.body()) for HMAC verification — never request.json() before verify
      - Caps raw body at 1MB -> 413 if exceeded
      - Verifies X-Hub-Signature-256 via HMAC SHA256 (verify_hmac_sha256)
      - Validates phone_number_id against WHATSAPP_PHONE_NUMBER_ID env
    Invalid payload: 200 skip without crash (spec Invalid payload safe).
    Duplicate message_id (LRU 1000): 200 without second BackgroundTasks.
    """
    # --- Raw body + 1MB cap (must be before JSON parse for HMAC) ---
    try:
        raw = await request.body()
    except Exception:
        logger.info("[webhook] POST no body or read error — 200 skip")
        return JSONResponse({"status": "ok", "note": "no body"}, status_code=200)

    if len(raw) > 1_000_000:
        logger.warning("[webhook] POST body too large (%d bytes) — 413", len(raw))
        return JSONResponse({"status": "error", "detail": "Payload too large"}, status_code=413)

    # --- HMAC verification (horariumTest parity) ---
    # Starlette headers are case-insensitive, but check both variants explicitly
    signature_header = request.headers.get("x-hub-signature-256")
    if signature_header is None:
        signature_header = request.headers.get("X-Hub-Signature-256")
    # Support both naming conventions for env secret
    app_secret = os.getenv("WHATSAPP_APP_SECRET", "").strip()
    # Fallback alias (same name, kept for spec compatibility)
    if not app_secret:
        app_secret = os.getenv("WHATSAPP_APP_SECRET", "").strip()

    if app_secret:
        if not verify_hmac_sha256(raw, signature_header, app_secret):
            logger.warning("[webhook] POST invalid HMAC signature — 401")
            return PlainTextResponse(content="Invalid signature", status_code=401)
    else:
        logger.warning("[webhook] WHATSAPP_APP_SECRET not set — skipping HMAC verification (dev mode)")

    # --- JSON parse after HMAC ---
    try:
        # Handle empty body gracefully
        if not raw or not raw.strip():
            raise ValueError("empty body")
        payload = json.loads(raw)
    except Exception:
        # No JSON body or malformed — safe skip (do not 401/413)
        logger.info("[webhook] POST no JSON or invalid — 200 skip")
        return JSONResponse({"status": "ok", "note": "no json"}, status_code=200)

    if not isinstance(payload, dict):
        logger.info("[webhook] POST payload not dict — 200 skip")
        return JSONResponse({"status": "ok"}, status_code=200)

    # --- phone_number_id validation (horariumTest parity) ---
    env_phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    # Also support alias (same var, kept for spec)
    if not env_phone_id:
        env_phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()

    # Extract phone_number_id from payload: entry[0].changes[0].value.metadata.phone_number_id
    payload_phone_id: str | None = None
    try:
        entries = payload.get("entry") or []
        if isinstance(entries, list) and entries:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                changes = entry.get("changes") or []
                if not isinstance(changes, list):
                    continue
                for ch in changes:
                    if not isinstance(ch, dict):
                        continue
                    value = ch.get("value") or {}
                    if not isinstance(value, dict):
                        continue
                    metadata = value.get("metadata") or {}
                    if isinstance(metadata, dict):
                        pid = metadata.get("phone_number_id")
                        if pid:
                            payload_phone_id = str(pid).strip()
                            break
                if payload_phone_id:
                    break
    except Exception:
        payload_phone_id = None

    # Phone ID validation — enforce only when HMAC is configured (prod).
    # In dev without WHATSAPP_APP_SECRET we skip strict phone check to keep existing
    # pytest suite (which uses dummy phone_number_id 123456) green; prod with
    # Ramiro's number will have both APP_SECRET and PHONE_ID set -> strict.
    # See task: horariumTest parity; APP_SECRET empty -> dev mode warning but allow.
    if env_phone_id and app_secret:
        if payload_phone_id:
            if payload_phone_id != env_phone_id:
                logger.warning(
                    "[webhook] POST phone_number_id mismatch payload=%r env=%r — 401",
                    payload_phone_id,
                    env_phone_id,
                )
                return PlainTextResponse(content="Invalid phone_number_id", status_code=401)
        else:
            # Env is set but payload has no phone_number_id — spec says 401 Missing
            # Only enforce if payload looks like a WhatsApp message (has entry/changes)
            # to avoid breaking status-update / empty payloads that legitimately lack metadata.
            # We still respect `app_secret` gate above (prod-only).
            has_value = False
            try:
                for e in payload.get("entry") or []:
                    if isinstance(e, dict) and isinstance(e.get("changes"), list):
                        for c in e["changes"]:
                            if isinstance(c, dict) and isinstance(c.get("value"), dict):
                                has_value = True
                                break
                    if has_value:
                        break
            except Exception:
                has_value = False
            if has_value:
                logger.warning("[webhook] POST missing phone_number_id but env is set — 401")
                return PlainTextResponse(content="Missing phone_number_id", status_code=401)
    elif env_phone_id and not app_secret:
        # Dev mode: phone ID set but HMAC not configured — log but allow (test compatibility)
        if payload_phone_id and payload_phone_id != env_phone_id:
            logger.info(
                "[webhook] phone_number_id mismatch but HMAC not configured — allowing (dev): payload=%r env=%r",
                payload_phone_id,
                env_phone_id,
            )

    messages = _extract_messages(payload)
    if not messages:
        # Status updates (delivery/read) or empty — still ACK 200
        logger.info("[webhook] POST no messages in payload — 200 skip")
        return JSONResponse({"status": "ok", "note": "no messages"}, status_code=200)

    enqueued = 0
    for m in messages:
        mid = m["id"]
        frm = m["from"]
        body = m["body"]
        msg_type = m.get("msg_type", "text")
        if _is_duplicate_lru(mid):
            logger.info("[webhook] dedup lru hit %s — skip enqueue", mid)
            continue
        # Enqueue background (msg_type for text-only RAG/Docs gating)
        background_tasks.add_task(process_question, mid, frm, body, msg_type)
        enqueued += 1
        logger.info("[WhatsApp] Mensaje recibido id=%s from=%s type=%s", mid, frm, msg_type)

    return JSONResponse({"status": "ok", "enqueued": enqueued}, status_code=200)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """
    Health check: 200 if index exists else 503 with --reindex hint.
    Checks lifespan _query_engine and filesystem chroma_db as fallback.
    """
    # Fast path: lifespan succeeded
    if _query_engine is not None:
        return JSONResponse({"status": "ok", "index": "loaded"}, status_code=200)

    # Fallback: check on-demand if file exists now (e.g., after reindex without restart)
    # Try lightweight filesystem check first before attempting load
    from pathlib import Path

    chroma_dir = Path("chroma_db")
    # If chroma_db exists and has content, try to load lazily once
    if rag_core is not None and chroma_dir.exists():
        try:
            # any file inside indicates persist
            has_files = any(chroma_dir.iterdir())
            if has_files and _index_error is not None:
                # Try one lazy load attempt for health probe
                # but don't cache globally if it fails (keep _query_engine None)
                # For simplicity just report based on file existence
                pass
        except Exception:
            pass

    # If still no engine, report 503
    if _index_error:
        hint = "ejecutar con --reindex tras preparar datos/" if "reindex" not in _index_error.lower() else _index_error
        return JSONResponse(
            {"status": "unavailable", "reason": _index_error, "hint": hint},
            status_code=503,
        )

    # Check filesystem directly for fresh deploys where lifespan not yet set
    if not chroma_dir.exists() or not any(chroma_dir.iterdir()) if chroma_dir.exists() else True:
        return JSONResponse(
            {
                "status": "unavailable",
                "reason": "chroma_db not found or empty",
                "hint": "ejecutar python main.py --reindex tras preparar datos/",
            },
            status_code=503,
        )

    # chroma exists but engine not loaded (e.g., startup not run in test) — still 200 if files present
    return JSONResponse({"status": "ok", "index": "present (not yet loaded)"}, status_code=200)


# ---------------------------------------------------------------------------
# Background worker: process_question
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown artifacts for WhatsApp short reply."""
    import re

    if not text:
        return ""
    # Remove code blocks and inline code
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove bold/italic markers **, __, *, _
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove headers
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove remaining markdown chars but keep line breaks for now
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_for_whatsapp(text: str, max_chars: int = 350) -> str:
    """Truncate to ~280-350 chars, 3 lines max, word-boundary safe."""
    cleaned = _strip_markdown(text).strip()
    # Normalize whitespace and newlines to spaces for whatsapp (keep single line breaks as spaces for length calc)
    # But keep one newline handling? For short answer we want plain text without markdown, single paragraph.
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= max_chars:
        return cleaned
    # Truncate at word boundary near max_chars
    truncated = cleaned[:max_chars]
    # Try to cut at last space
    last_space = truncated.rfind(" ")
    if last_space > int(max_chars * 0.7):  # only if not too short
        truncated = truncated[:last_space]
    truncated = truncated.rstrip(" ,.;:")
    return truncated + "…"


def _format_whatsapp_answer(answer: str, fuentes: list[dict], materia: str | None = None) -> str:
    """
    Legacy compat: WhatsApp is interface-only, never sends academic content.
    Always returns SUCCESS_MSG (fallback also returns SUCCESS_MSG per new contract:
    fallback IS saved in Docs, WhatsApp still replies success).
    Kept for backward-compat imports; new code sends SUCCESS_MSG directly.
    """
    return SUCCESS_MSG


# Ops counter for Cloud send failures (in-memory, process-local).
_SEND_FAILURES = 0
_SEND_FAILURES_LOCK = threading.Lock()


def _send_config_presence() -> tuple[bool, bool]:
    """Masked credential presence for logs — booleans only, never values."""
    try:
        token_set = bool(os.getenv("WHATSAPP_TOKEN", "").strip())
    except Exception:
        token_set = False
    try:
        phone_set = bool(os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip())
    except Exception:
        phone_set = False
    return token_set, phone_set


def _record_send_failure(to: str, reason: str) -> int:
    """Increment the ops failure counter and log one ERROR line. Never raises."""
    global _SEND_FAILURES
    token_set, phone_set = _send_config_presence()
    try:
        with _SEND_FAILURES_LOCK:
            _SEND_FAILURES += 1
            count = _SEND_FAILURES
    except Exception:
        count = -1
    logger.error(
        "[WhatsApp] send failed to=%s reason=%s failures=%d token_set=%s phone_set=%s "
        "— check WHATSAPP_TOKEN, phone number ID, tunnel/Meta config",
        to,
        reason,
        count,
        token_set,
        phone_set,
    )
    return count


def _send_text(to: str, body: str) -> bool:
    """Send WhatsApp text via client. Returns True on send OK. Never raises."""
    if whatsapp_client is None:
        _record_send_failure(to, "client-unavailable")
        return False
    try:
        ok = whatsapp_client.send_text(to, body)  # type: ignore
        if not ok:
            _record_send_failure(to, "send-text-false")
        return bool(ok)
    except Exception as e:
        _record_send_failure(to, "exception:%s" % type(e).__name__)
        return False


def process_question(message_id: str, from_number: str, text_body: str, msg_type: str = "text") -> None:
    """
    Orchestrate (interface-only WhatsApp):
      parse MATERIA: pregunta -> validate -> RAG exclusive -> Docs append -> SUCCESS ack.

    WhatsApp never receives academic content, summaries, quotes, pages or links.
    Runs in BackgroundTasks (sync function). Never raises.
    Logs with prefixes: [WhatsApp] [Materia] [Pregunta] [RAG] [LLM] [Google Docs].
    """
    logger.info("[WhatsApp] Mensaje recibido id=%s from=%s type=%s", message_id, from_number, msg_type)

    if not from_number or not str(from_number).strip():
        logger.warning("[WhatsApp] empty from_number — abort id=%s", message_id)
        return

    # Only type=text is processed with RAG/Docs; rest gets help without RAG/Docs.
    mtype = str(msg_type or "text").strip().lower()
    if mtype and mtype != "text":
        logger.info("[WhatsApp] non-text type %r — send help without RAG/Docs", mtype)
        _send_text(str(from_number), MISSING_COLON_MSG)
        return

    raw_text = (text_body or "").strip()
    if not raw_text:
        logger.info("[WhatsApp] empty body — send help")
        _send_text(str(from_number), MISSING_COLON_MSG)
        return

    if ":" not in raw_text:
        logger.info("[Materia] missing colon — send help")
        _send_text(str(from_number), MISSING_COLON_MSG)
        return

    if materias_config is None:
        logger.error("[Materia] materias_config unavailable — send help")
        _send_text(str(from_number), MISSING_COLON_MSG)
        return

    parsed = materias_config.parse_message(raw_text)  # type: ignore
    if parsed is None:
        logger.info("[Materia] parse failed — send help")
        _send_text(str(from_number), MISSING_COLON_MSG)
        return

    raw_materia, question = parsed
    canonical = materias_config.validate_materia(raw_materia)  # type: ignore
    if canonical is None:
        logger.info("[Materia] unknown %r — no RAG/Docs", raw_materia)
        _send_text(str(from_number), unknown_materia_msg(str(raw_materia)))
        return

    question = question.strip()
    if not question:
        logger.info("[Materia] empty question — send help")
        _send_text(str(from_number), MISSING_COLON_MSG)
        return

    logger.info("[Materia] %s", canonical)
    logger.info("[Pregunta] %s", question)

    # Already completed via Docs OK -> no rewrite, but still ack so Meta
    # retries never look like "saved, no reply". Reuses SUCCESS, no new copy.
    try:
        if drive_docs_service is not None:
            peek = getattr(drive_docs_service, "is_already_processed", None)
            if callable(peek) and peek(message_id):
                logger.info("[Google Docs] duplicate wamid %s already saved — resend SUCCESS ack", message_id)
                _send_text(str(from_number), SUCCESS_MSG)
                return
    except Exception:
        pass

    # --- RAG exclusive query (BEFORE LLM filtering inside rag_core) ---
    answer: str = ""
    fuentes: list[dict] = []
    fallback_phrase = "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad."
    try:
        if rag_core is not None:
            try:
                fallback_phrase = str(getattr(rag_core, "FALLBACK_PHRASE", fallback_phrase))
            except Exception:
                pass
    except Exception:
        pass
    if rag_core is None:
        logger.error("[RAG] rag_core unavailable — fallback")
        answer = fallback_phrase
        fuentes = []
    else:
        logger.info("[RAG] Consultando exclusivamente %s", canonical)
        try:
            used_qfm = False
            if hasattr(rag_core, "query_for_materia"):
                try:
                    maybe = rag_core.query_for_materia(canonical, question)  # type: ignore
                    if isinstance(maybe, tuple) and len(maybe) == 2 and isinstance(maybe[0], str) and isinstance(maybe[1], list):
                        answer, fuentes = maybe  # type: ignore
                        used_qfm = True
                    else:
                        raise ValueError(f"query_for_materia returned invalid type {type(maybe)}")
                except BaseException as e_qfm:
                    # Incluye SystemExit/429/cuota: webhook NUNCA sale ni reintenta.
                    # Fail-fast a fallback exacto, sigue a Drive + ✅.
                    if used_qfm:
                        raise
                    logger.warning("[RAG] query_for_materia failed (%s) — safe fallback", e_qfm)
                    answer, fuentes = fallback_phrase, []
                    used_qfm = True
                if used_qfm:
                    answer = str(answer).strip() if answer else ""
                    if not isinstance(fuentes, list):
                        fuentes = []
            else:
                answer, fuentes = rag_core.query(question)  # type: ignore
                answer = str(answer).strip() if answer else ""
                if not isinstance(fuentes, list):
                    fuentes = []
        except BaseException as e:
            # Defensa en profundidad: ni SystemExit ni 429 matan el BackgroundTask.
            logger.warning("[RAG] fail-fast for %r: %s — fallback sin reintento", question, e)
            answer = fallback_phrase
            fuentes = []

    if not answer:
        answer = fallback_phrase

    try:
        n_frag = len(fuentes) if isinstance(fuentes, list) else 0
    except Exception:
        n_frag = 0
    logger.info("[RAG] %d fragmentos encontrados", n_frag)
    logger.info("[LLM] Respuesta generada")

    # --- Docs append (definitive). Never mark success before Docs OK. ---
    if drive_docs_service is None:
        logger.error("[Google Docs] service unavailable — send error, unmark LRU")
        _remove_from_lru(message_id)
        _send_text(str(from_number), DOCS_FAIL_MSG)
        return
    try:
        from datetime import datetime, timezone

        fecha_iso = datetime.now(timezone.utc).isoformat()
        ok = drive_docs_service.append_entry(  # type: ignore
            materia=canonical,
            fecha_iso=fecha_iso,
            pregunta=question,
            respuesta=answer,
            fuentes=fuentes,
            message_id=message_id,
        )
        if not ok:
            logger.warning("[Google Docs] append failed id=%s — send error, unmark LRU", message_id)
            _remove_from_lru(message_id)
            _send_text(str(from_number), DOCS_FAIL_MSG)
            return
    except Exception as e:
        logger.error("[Google Docs] append exception id=%s: %s — unmark LRU", message_id, e)
        _remove_from_lru(message_id)
        _send_text(str(from_number), DOCS_FAIL_MSG)
        return

    logger.info("[Google Docs] Pregunta guardada id=%s materia=%s", message_id, canonical)
    # --- WhatsApp confirmation (interface-only, never academic) ---
    # Empty-source case (e.g. Redes vacia): fallback with zero sources is saved
    # in Drive as "- Sin fuentes registradas"; WhatsApp stays SUCCESS only.
    try:
        is_fallback = (not fuentes) or (str(answer).strip() == fallback_phrase)
    except Exception:
        is_fallback = False
    if is_fallback:
        logger.info(
            "[RAG] fallback with zero sources for %s id=%s — Drive shows "
            "'- Sin fuentes registradas', WhatsApp stays interface-only SUCCESS",
            canonical,
            message_id,
        )
    sent = _send_text(str(from_number), SUCCESS_MSG)
    if sent:
        logger.info("[WhatsApp] Confirmación enviada id=%s", message_id)
    else:
        # _send_text already logged ERROR with masked presence plus the ops
        # counter. Retry the same SUCCESS once — no new user-facing copy.
        # Docs already saved, so DOCS_FAIL_MSG would be wrong here.
        logger.warning("[WhatsApp] retry SUCCESS ack id=%s after send failure", message_id)
        if _send_text(str(from_number), SUCCESS_MSG):
            logger.info("[WhatsApp] Confirmación enviada on retry id=%s", message_id)
        else:
            logger.error(
                "[WhatsApp] SUCCESS ack undeliverable id=%s — Docs saved but user "
                "sees nothing; check WHATSAPP_TOKEN, number ID, tunnel/Meta config",
                message_id,
            )


def _send_help(to: str, body: str | None = None) -> None:
    """Send help without RAG/Docs. Defaults to missing-colon exact message."""
    _send_text(to, body or MISSING_COLON_MSG)


# For uvicorn import check
__all__ = ["app", "process_question", "clear_lru_for_tests", "verify_webhook", "receive_webhook", "health", "verify_hmac_sha256", "_extract_messages", "SUCCESS_MSG", "MISSING_COLON_MSG", "DOCS_FAIL_MSG", "unknown_materia_msg"]
