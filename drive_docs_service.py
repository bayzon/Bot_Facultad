#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
drive_docs_service.py — Per-materia Google Docs accumulation for Bot Facultad.

Handles:
  - Service Account auth (google.oauth2.service_account) with scopes
    https://www.googleapis.com/auth/drive
    https://www.googleapis.com/auth/documents
  - File cache materias_docs.json { materia_canonical: docId } with thread-safe
    load/save and per-materia Lock dict.
  - get_or_create_doc(materia) auto-creates "MATERIA - Bot Facultad" in folder
    GOOGLE_DRIVE_FOLDER_ID, grants Editor share (best-effort), caches id.
  - append_entry(materia, fecha_iso, pregunta, respuesta, fuentes, message_id)
    via docs.documents.batchUpdate insertText at endIndex-1, per-materia
    threading.Lock, message_id dedup (thread-safe set).

Templated block (append_entry):
  Fecha: {iso}
  Pregunta: {q}
  Respuesta: {a}
  Fuentes: {src p. N}
  ---
  For fallback (no evidence) respuesta is normalized to
  "Pregunta sin respuesta en bibliografía" and Fuentes is "N/A".

Env contract (.env):
  GOOGLE_SERVICE_ACCOUNT_JSON — path to SA JSON file (e.g. ./bot-facultad-sa.json)
  GOOGLE_DRIVE_FOLDER_ID      — Drive folder id where docs are created

Concurrency: per-materia Lock avoids interleaved batchUpdate at endIndex;
global cache Lock protects file I/O; dedup set prevents double append on
Meta retry (webhook also does LRU 1000, this is second layer).

Errors: never raises to webhook caller — logs and returns gracefully.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]

CACHE_PATH = Path("materias_docs.json")
# Exact fallback saved in Docs (same as LLM contract). WhatsApp never shows it.
FALLBACK_DISPLAY = "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad."
# Legacy fallback for detection (old index may still return it).
RAG_FALLBACK_PHRASE = (
    "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad."
)
LEGACY_RAG_FALLBACK_PHRASE = (
    "No encontré información suficiente en los documentos para responder esa pregunta."
)
# Frase usuario normalizada (sin tilde) y variante con tilde para detección
USER_FALLBACK_PHRASE = "No se encontro una respuesta en la fuente."
USER_FALLBACK_PHRASE_ACCENT = "No se encontró una respuesta en la fuente."

# Locks & dedup
_cache_lock = threading.Lock()
_materia_locks: dict[str, threading.Lock] = {}
_materia_locks_lock = threading.Lock()  # protects _materia_locks dict
_dedup_lock = threading.Lock()
# Use OrderedDict as bounded dedup set (LRU eviction 2000) — intra-module only
# webhook has primary LRU 1000, this is secondary layer for background retries
_DEDUP_MAX = 2000
_processed_ids: OrderedDict[str, None] = OrderedDict()

# In-memory cache mirror (loaded lazily)
_cache: dict[str, str] | None = None

# Google service singletons (lazy)
_drive_service: Any | None = None
_docs_service: Any | None = None
_credentials: Any | None = None


# ---------------------------------------------------------------------------
# Helpers: cache I/O
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, str]:
    """Thread-safe load of materias_docs.json into memory (cached)."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            return dict(_cache)
        path = CACHE_PATH
        if not path.exists():
            _cache = {}
            return {}
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                _cache = {}
                return {}
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning(
                    "[drive] cache %s is not a dict — resetting to {}", path
                )
                _cache = {}
                return {}
            # Normalize to str->str
            normalized = {str(k): str(v) for k, v in data.items()}
            _cache = normalized
            return dict(_cache)
        except json.JSONDecodeError as e:
            logger.warning("[drive] cache JSON invalid %s: %s — resetting", path, e)
            _cache = {}
            return {}
        except Exception as e:
            logger.warning("[drive] failed to read cache %s: %s", path, e)
            _cache = {}
            return {}


def _save_cache(data: dict[str, str]) -> None:
    """Thread-safe atomic save of cache to disk."""
    global _cache
    with _cache_lock:
        _cache = dict(data)
        tmp = CACHE_PATH.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp.replace(CACHE_PATH)
            logger.info("[drive] cache saved %s (%d entries)", CACHE_PATH, len(data))
        except Exception as e:
            logger.error("[drive] failed to save cache %s: %s", CACHE_PATH, e)


def _get_materia_lock(materia: str) -> threading.Lock:
    """Return per-materia Lock, creating if needed (thread-safe)."""
    key = materia.strip()
    with _materia_locks_lock:
        if key not in _materia_locks:
            _materia_locks[key] = threading.Lock()
        return _materia_locks[key]


def _is_duplicate(message_id: str) -> bool:
    """
    Check message_id dedup set. Returns True if already seen (duplicate).
    Otherwise marks as seen and returns False. Thread-safe with LRU eviction.
    """
    if not message_id:
        return False
    with _dedup_lock:
        if message_id in _processed_ids:
            # Move to end (LRU touch)
            _processed_ids.move_to_end(message_id)
            return True
        _processed_ids[message_id] = None
        # Evict oldest if over cap
        while len(_processed_ids) > _DEDUP_MAX:
            _processed_ids.popitem(last=False)
        return False


def _mark_duplicate(message_id: str) -> None:
    """Explicitly mark id without check (used after successful append)."""
    if not message_id:
        return
    with _dedup_lock:
        _processed_ids[message_id] = None
        _processed_ids.move_to_end(message_id)
        while len(_processed_ids) > _DEDUP_MAX:
            _processed_ids.popitem(last=False)


def is_already_processed(message_id: str) -> bool:
    """Peek dedup set without marking. True if already completed via Docs OK."""
    if not message_id:
        return False
    with _dedup_lock:
        return message_id in _processed_ids


def clear_dedup_for_tests() -> None:
    """Test helper: clear dedup set and materia locks."""
    with _dedup_lock:
        _processed_ids.clear()
    with _materia_locks_lock:
        _materia_locks.clear()


def reset_cache_for_tests() -> None:
    """Test helper: clear in-memory cache (does not delete file)."""
    global _cache
    with _cache_lock:
        _cache = None


# ---------------------------------------------------------------------------
# Google auth
# ---------------------------------------------------------------------------

def _get_credentials():
    """
    Load Service Account credentials from GOOGLE_SERVICE_ACCOUNT_JSON.

    Returns google.oauth2.service_account.Credentials or None on failure.
    Caches singleton after first success.
    """
    global _credentials
    if _credentials is not None:
        return _credentials

    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_path:
        logger.warning(
            "[drive] GOOGLE_SERVICE_ACCOUNT_JSON not set — Drive/Docs disabled "
            "(set path to SA JSON in .env, e.g. ./bot-facultad-sa.json)"
        )
        return None

    p = Path(sa_path)
    # Resolve relative to project cwd
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        logger.error("[drive] SA JSON file not found: %s (env=%s)", p, sa_path)
        return None

    try:
        from google.oauth2.service_account import Credentials  # type: ignore

        creds = Credentials.from_service_account_file(str(p), scopes=SCOPES)
        _credentials = creds
        logger.info("[drive] SA credentials loaded from %s", p)
        return creds
    except ImportError as e:
        logger.error(
            "[drive] google-auth not installed (pip install google-auth): %s", e
        )
        return None
    except Exception as e:
        logger.error("[drive] failed to load SA credentials from %s: %s", p, e)
        return None


def _get_drive_service():
    """Lazy Drive v3 service (cached). Returns None on failure."""
    global _drive_service
    if _drive_service is not None:
        return _drive_service
    creds = _get_credentials()
    if creds is None:
        return None
    try:
        from googleapiclient.discovery import build  # type: ignore

        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        _drive_service = svc
        return svc
    except ImportError as e:
        logger.error(
            "[drive] google-api-python-client not installed "
            "(pip install google-api-python-client): %s",
            e,
        )
        return None
    except Exception as e:
        logger.error("[drive] failed to build drive service: %s", e)
        return None


def _get_docs_service():
    """Lazy Docs v1 service (cached). Returns None on failure."""
    global _docs_service
    if _docs_service is not None:
        return _docs_service
    creds = _get_credentials()
    if creds is None:
        return None
    try:
        from googleapiclient.discovery import build  # type: ignore

        svc = build("docs", "v1", credentials=creds, cache_discovery=False)
        _docs_service = svc
        return svc
    except ImportError as e:
        logger.error(
            "[drive] google-api-python-client not installed: %s", e
        )
        return None
    except Exception as e:
        logger.error("[drive] failed to build docs service: %s", e)
        return None


def _get_folder_id() -> str | None:
    fid = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    if not fid or "TEST" in fid or "dummy" in fid.lower():
        # Allow TEST dummy for tests — still return it so mocked Drive can assert
        # but log warning when it looks like dummy and file cache will be used
        if not fid:
            logger.warning("[drive] GOOGLE_DRIVE_FOLDER_ID not set — doc create will fail")
            return None
    return fid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _find_existing_doc(drive: Any, folder_id: str, title: str) -> str | None:
    """
    Search GOOGLE_DRIVE_FOLDER_ID for an exact-name non-trashed Doc.

    Returns doc id on hit, None on miss or lookup failure (caller falls
    through to create). Never raises.
    """
    try:
        safe_title = title.replace("'", "\\'")
        query = f"name='{safe_title}' and '{folder_id}' in parents and trashed=false"
        result = (
            drive.files()
            .list(
                q=query,
                supportsAllDrives=True,
                fields="files(id,name)",
                pageSize=10,
            )
            .execute()
        )
        files = result.get("files", []) if isinstance(result, dict) else []
        if isinstance(files, list) and files:
            first = files[0] if isinstance(files[0], dict) else {}
            doc_id = first.get("id")
            if doc_id and isinstance(doc_id, str) and doc_id.strip():
                logger.info("[drive] found existing doc '%s' -> %s", title, doc_id)
                return doc_id.strip()
        return None
    except Exception as e:
        logger.warning("[drive] doc lookup failed for '%s': %s — will try create", title, e)
        return None


def get_or_create_doc(materia: str) -> str | None:
    """
    Return Google Doc id for materia, auto-creating "MATERIA - Preguntas".

    Cache-aware: checks materias_docs.json first (memory + disk).
    On miss, creates via Drive API in GOOGLE_DRIVE_FOLDER_ID folder.
    Docs stay private (no anyone/writer permission).

    Args:
        materia: Canonical materia name (e.g. "Redes de Datos"). Whitespace trimmed.

    Returns:
        Doc id string, or None on failure (logged).
    """
    if not isinstance(materia, str) or not materia.strip():
        logger.warning("[drive] get_or_create_doc: empty materia")
        return None
    canonical = materia.strip()

    # Check cache
    cache = _load_cache()
    if canonical in cache and cache[canonical].strip():
        logger.info("[drive] cache hit for '%s' -> %s", canonical, cache[canonical])
        return cache[canonical]

    # Need to create — require services
    drive = _get_drive_service()
    if drive is None:
        logger.error(
            "[drive] cannot create doc for '%s' — drive service unavailable", canonical
        )
        return None

    folder_id = _get_folder_id()
    if not folder_id:
        logger.error("[drive] cannot create doc — GOOGLE_DRIVE_FOLDER_ID missing")
        return None

    title = f"{canonical} - Preguntas"

    # Reuse pre-created Doc shared with the SA (avoids files.create 403
    # storageQuotaExceeded). Only create when no exact-name hit exists.
    existing_id = _find_existing_doc(drive, folder_id, title)
    if existing_id:
        cache[canonical] = existing_id
        _save_cache(cache)
        return existing_id

    body = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id],
    }
    try:
        # Use fields=id to minimize response
        created = drive.files().create(body=body, fields="id").execute()
        doc_id = created.get("id") if isinstance(created, dict) else None
        if not doc_id:
            # Some mock returns object with .get
            try:
                doc_id = created.get("id")  # type: ignore
            except Exception:
                doc_id = None
        if not doc_id:
            logger.error("[drive] create returned no id for '%s': %r", canonical, created)
            return None
        logger.info("[drive] created doc '%s' -> %s in folder %s", title, doc_id, folder_id)

        # Docs stay private — no anyone/writer permission (removed per spec).

        # Cache write
        cache[canonical] = doc_id
        _save_cache(cache)
        return doc_id

    except Exception as e:
        logger.error("[drive] failed to create doc '%s': %s", title, e)
        return None


def _format_fecha(fecha_iso: str) -> str:
    """Convert ISO datetime to DD/MM/YYYY HH:MM. Falls back to now on parse failure."""
    from datetime import datetime, timezone

    try:
        raw = (fecha_iso or "").strip()
        if raw:
            # Handle trailing Z
            iso = raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            # Convert to local for display; keep naive as-is
            try:
                if dt.tzinfo is not None:
                    dt = dt.astimezone()
            except Exception:
                pass
            return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        pass
    try:
        return datetime.now().astimezone().strftime("%d/%m/%Y %H:%M")
    except Exception:
        from datetime import datetime as _dt

        return _dt.now().strftime("%d/%m/%Y %H:%M")


def _format_fuentes_lines(fuentes: list[dict] | None) -> str:
    """Format fuentes lines: `- {archivo} — página {p}` or `- {archivo}` if no page."""
    if not fuentes:
        return "- Sin fuentes registradas"
    lines: list[str] = []
    for f in fuentes:
        try:
            archivo = str(f.get("file") or f.get("archivo") or "?").strip() or "?"
        except Exception:
            archivo = "?"
        try:
            page_raw = f.get("page")
            page = str(page_raw).strip() if page_raw is not None else ""
        except Exception:
            page = ""
        if not page or page == "?" or page.lower() in ("n/a", "na", "sin pagina", "sin página", "-"):
            lines.append(f"- {archivo}")
        else:
            lines.append(f"- {archivo} — página {page}")
    return "\n".join(lines) if lines else "- Sin fuentes registradas"


def _extract_doc_text(doc: dict) -> str:
    """Extract plain text from Docs API get() response for Pregunta-N counting."""
    try:
        body = doc.get("body", {}) if isinstance(doc, dict) else {}
        content = body.get("content", []) if isinstance(body, dict) else []
        parts: list[str] = []
        for el in content:
            if not isinstance(el, dict):
                continue
            para = el.get("paragraph")
            if not isinstance(para, dict):
                continue
            for elem in para.get("elements", []) or []:
                if not isinstance(elem, dict):
                    continue
                tr = elem.get("textRun")
                if isinstance(tr, dict) and isinstance(tr.get("content"), str):
                    parts.append(tr["content"])
        return "".join(parts)
    except Exception:
        return ""


def _count_existing_questions(doc: dict | None) -> int:
    """Count existing `Pregunta N` blocks in doc text for auto-numbering."""
    import re as _re

    if not doc or not isinstance(doc, dict):
        return 0
    text = _extract_doc_text(doc)
    if not text:
        return 0
    return len(_re.findall(r"^Pregunta\s+\d+", text, flags=_re.MULTILINE))


def _format_block(
    fecha_iso: str,
    pregunta: str,
    respuesta: str,
    fuentes: list[dict],
    numero: int = 1,
) -> str:
    """
    Build exact append block (byte-exact contract):

      --------------------------------------------------\\n\\nPregunta {N}\\n\\n{pregunta}\\n\\nRespuesta:\\n\\n{respuesta}\\n\\nFuentes utilizadas:\\n- {archivo} — página {p}\\n\\nFecha:\\n{DD/MM/YYYY HH:MM}\\n\\n--------------------------------------------------

    - numero defaults to 1 for backward-compat direct calls; append_entry computes N.
    - respuesta empty -> FALLBACK_DISPLAY (fallback IS saved in Docs).
    - fuentes empty -> "- Sin fuentes registradas" (never invent pages).
    """
    # Normalize respuesta for no-info case — alinea con contrato WhatsApp ultra simple
    # FALLBACK_DISPLAY ahora es "No se encontro una respuesta en la fuente." (requerimiento final)
    # Se detectan variantes con/sin acento y frases rag_core para robustez.
    raw = (respuesta or "").strip()
    display_respuesta = raw if raw else FALLBACK_DISPLAY
    try:
        n = int(numero) if int(numero) >= 1 else 1
    except Exception:
        n = 1
    fuentes_block = _format_fuentes_lines(fuentes)
    fecha_fmt = _format_fecha(fecha_iso)
    pregunta_clean = (pregunta or "").strip()
    sep = "--------------------------------------------------"
    block = (
        f"{sep}\n"
        f"\n"
        f"Pregunta {n}\n"
        f"\n"
        f"{pregunta_clean}\n"
        f"\n"
        f"Respuesta:\n"
        f"\n"
        f"{display_respuesta}\n"
        f"\n"
        f"Fuentes utilizadas:\n"
        f"{fuentes_block}\n"
        f"\n"
        f"Fecha:\n"
        f"{fecha_fmt}\n"
        f"\n"
        f"{sep}"
    )
    # Docs batchUpdate needs trailing newline to avoid glued appends
    return block + "\n"


def append_entry(
    materia: str,
    fecha_iso: str,
    pregunta: str,
    respuesta: str,
    fuentes: list[dict] | None,
    message_id: str,
) -> bool:
    """
    Append a Q&A block to the per-materia Google Doc.

    Thread-safe: per-materia Lock serializes batchUpdate at endIndex.
    Idempotent: message_id dedup (thread-safe bounded set) — duplicate call
    returns True without second write.

    Args:
        materia: Canonical materia (e.g. "AM2")
        fecha_iso: ISO datetime string
        pregunta: Question text
        respuesta: Answer text (or fallback phrase)
        fuentes: List of {file, page} dicts (may be None/empty for fallback)
        message_id: WhatsApp wamid for dedup

    Returns:
        True on success or duplicate (no extra write needed), False on failure.
    """
    if not materia or not materia.strip():
        logger.warning("[drive] append_entry: empty materia — skip")
        return False
    if not pregunta or not pregunta.strip():
        logger.warning("[drive] append_entry: empty pregunta — skip")
        return False
    if not fecha_iso or not fecha_iso.strip():
        # Generate fallback ISO if caller forgot
        from datetime import datetime, timezone

        fecha_iso = datetime.now(timezone.utc).isoformat()

    canonical = materia.strip()
    fuentes_list: list[dict] = list(fuentes) if fuentes else []

    # Dedup peek (no marking yet): if already completed via Docs OK, skip without rewrite.
    if message_id and is_already_processed(message_id):
        logger.info(
            "[drive] dedup: message_id %s already processed — skip append for %s",
            message_id,
            canonical,
        )
        return True

    # Per-materia lock
    lock = _get_materia_lock(canonical)
    with lock:
        # Re-check inside lock (race-safe peek, still no marking).
        if message_id and is_already_processed(message_id):
            logger.info(
                "[drive] dedup (inside lock): message_id %s already processed — skip",
                message_id,
            )
            return True

        doc_id = get_or_create_doc(canonical)
        if not doc_id:
            logger.error("[drive] append_entry: no doc id for '%s' — abort", canonical)
            # Do NOT mark dedup on failure so retry can proceed.
            return False

        docs = _get_docs_service()
        if docs is None:
            logger.error("[drive] append_entry: docs service unavailable")
            return False

        try:
            # Fetch doc for endIndex + auto-numbering (count Pregunta N blocks).
            # docs.documents().get(documentId=docId).execute()
            doc = docs.documents().get(documentId=doc_id).execute()  # type: ignore
            body = doc.get("body", {}) if isinstance(doc, dict) else {}
            content = body.get("content", []) if isinstance(body, dict) else []
            end_index = 1
            if content and isinstance(content, list):
                try:
                    last = content[-1]
                    if isinstance(last, dict):
                        end_index = int(last.get("endIndex", 1))
                    else:
                        end_index = int(getattr(last, "endIndex", 1))
                except Exception:
                    end_index = 1
            # Docs API uses 1-based index; insertion at endIndex-1 appends before final newline
            insert_index = max(1, end_index - 1)

            try:
                existing = _count_existing_questions(doc if isinstance(doc, dict) else None)
            except Exception:
                existing = 0
            numero = existing + 1
            block = _format_block(fecha_iso, pregunta, respuesta, fuentes_list, numero=numero)

            requests = [
                {
                    "insertText": {
                        "location": {"index": insert_index},
                        "text": block,
                    }
                }
            ]
            docs.documents().batchUpdate(  # type: ignore
                documentId=doc_id, body={"requests": requests}
            ).execute()
            logger.info(
                "[drive] appended to '%s' doc %s at %d (%d chars, id=%s)",
                canonical,
                doc_id,
                insert_index,
                len(block),
                message_id or "-",
            )
            # Mark definitive ONLY after Docs success.
            if message_id:
                _mark_duplicate(message_id)
            return True

        except Exception as e:
            logger.error(
                "[drive] batchUpdate failed for '%s' doc %s (id=%s): %s",
                canonical,
                doc_id,
                message_id or "-",
                e,
            )
            # Do NOT mark dedup on failure so retry can proceed.
            return False


__all__ = [
    "SCOPES",
    "CACHE_PATH",
    "FALLBACK_DISPLAY",
    "RAG_FALLBACK_PHRASE",
    "LEGACY_RAG_FALLBACK_PHRASE",
    "get_or_create_doc",
    "append_entry",
    "is_already_processed",
    "clear_dedup_for_tests",
    "reset_cache_for_tests",
]
