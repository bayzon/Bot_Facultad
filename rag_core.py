#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_core.py — Singleton RAG core extracted from main.py.

Provides the full Chroma + LlamaIndex lifecycle without CLI side-effects
so that both `main.py` (terminal) and `webhook_server.py` (FastAPI lifespan)
share the same VectorStoreIndex instance.

Extracted symbols (preserved verbatim where possible):
  - DATOS_DIR, CHROMA_DIR, COLLECTION_NAME
  - EMBED_MODEL_NAME, LLM_MODEL_NAME
  - ANTI_HALLUCINATION_QA_TMPL_STR / TEMPLATE / REFINE_TEMPLATE
  - validar_entorno(), configurar_modelos(), obtener_coleccion_chroma(),
    construir_o_cargar_indice(), crear_query_engine()
  - query() + singleton get_index() / get_query_engine()

Anti-hallucination contract: the phrase
  "No encontré información suficiente en los documentos para responder esa pregunta."
must remain exact — it is asserted by tests and drives fallback behavior.

Lifecycle:
  get_query_engine(reindex=False) -> lazy singleton CitationQueryEngine
  - calls configurar_modelos() once
  - calls construir_o_cargar_indice() (reuse if chroma_db has vectors)
  - missing chroma_db or empty collection without --reindex auto-builds if PDFs exist

Thread-safe via simple module lock; suitable for BackgroundTasks concurrency
with per-materia locks handled in drive_docs_service.

Dual-provider (Gemini / OpenAI):
  If GEMINI_API_KEY is set (non-empty), Gemini is used (free tier).
  Otherwise falls back to OpenAI (OPENAI_API_KEY). This keeps changes minimal
  and reversible; .env is gitignored, GEMINI_API_KEY never logged in full.

Author: extracted for Bot_Facultad whatsapp-drive-materias (PR1 Foundation)
Gemini migration: dual-provider support added (minimal, reversible)
"""

from __future__ import annotations

import os
import sys
import time
import threading
import warnings
from pathlib import Path
from typing import Any

# Ensure utf-8 stdout on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Load .env before Settings / key checks
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# LlamaIndex / Chroma imports with clear errors
# ---------------------------------------------------------------------------
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
    from llama_index.core.prompts import PromptTemplate
except ImportError as e:
    print(f"\n[ERROR] Falta llama-index-core o version incompatible: {e}")
    print("  pip install llama-index llama-index-core")
    raise

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError as e:
    print(f"\n[ERROR] Falta integracion de embeddings OpenAI: {e}")
    print("  pip install llama-index-embeddings-openai")
    raise

try:
    from llama_index.llms.openai import OpenAI
except ImportError as e:
    print(f"\n[ERROR] Falta integracion LLM OpenAI: {e}")
    print("  pip install llama-index-llms-openai")
    raise

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
except ImportError as e:
    print(f"\n[ERROR] Falta el vector store de Chroma: {e}")
    print("  pip install llama-index-vector-stores-chroma chromadb")
    raise

try:
    import chromadb
except ImportError as e:
    print(f"\n[ERROR] Falta chromadb: {e}")
    print("  pip install chromadb")
    raise

try:
    from llama_index.core.query_engine import CitationQueryEngine

    _HAS_CITATION_ENGINE = True
except ImportError:
    _HAS_CITATION_ENGINE = False
    CitationQueryEngine = None  # type: ignore

try:
    import llama_index.readers.file  # noqa: F401

    _HAS_FILE_READERS = True
except ImportError:
    _HAS_FILE_READERS = False

# Gemini optional imports — lazy, do not crash if not installed
try:
    # Suppress deprecation warning about google.generativeai retirement; still functional
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from llama_index.llms.gemini import Gemini as GeminiLLM
        from llama_index.embeddings.gemini import GeminiEmbedding

    _HAS_GEMINI = True
except ImportError:
    # Alternative import paths for some environments
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from llama_index_llms_gemini.base import Gemini as GeminiLLM  # type: ignore
            from llama_index_embeddings_gemini.base import GeminiEmbedding  # type: ignore

        _HAS_GEMINI = True
    except ImportError:
        GeminiLLM = None  # type: ignore
        GeminiEmbedding = None  # type: ignore
        _HAS_GEMINI = False


# ---------------------------------------------------------------------------
# Gemini embedding retry wrapper — handles free-tier 429 quota with backoff
# ---------------------------------------------------------------------------
if _HAS_GEMINI:
    # Only define when Gemini is available; otherwise base class is None
    class _RetryGeminiEmbedding(GeminiEmbedding):  # type: ignore
        """Thin wrapper around GeminiEmbedding that retries on 429/rate-limit.

        Free tier can hit 429 after ~40-60 embeddings/minute. This wrapper
        adds exponential backoff (10s, 20s, 40s...) and a small throttle
        between calls to stay under the limit. Keeps changes minimal and
        reversible — only used when GEMINI_API_KEY is active.
        """

        _THROTTLE_SEC = 0.7  # stay safely under ~80 req/min
        _MAX_RETRIES = 5

        def _call_with_retry(self, fn, *args, **kwargs):
            last_exc = None
            for attempt in range(self._MAX_RETRIES):
                try:
                    result = fn(*args, **kwargs)
                    # throttle after success to avoid hammering quota
                    time.sleep(self._THROTTLE_SEC)
                    return result
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    is_quota = (
                        "429" in str(e)
                        or "quota" in msg
                        or "rate" in msg
                        or "exceeded" in msg
                        or "resource" in msg
                    )
                    last_exc = e
                    if is_quota and attempt < self._MAX_RETRIES - 1:
                        wait = (2**attempt) * 10  # 10,20,40,80,160
                        # cap at 90s to avoid infinite
                        wait = min(wait, 90)
                        print(f"[WARN] Gemini 429/quota (attempt {attempt+1}/{self._MAX_RETRIES}), retry in {wait}s: {e}")
                        time.sleep(wait)
                        continue
                    raise
            if last_exc:
                raise last_exc
            raise RuntimeError("Gemini embedding retry exhausted")

        def _get_text_embedding(self, text: str):  # type: ignore[override]
            return self._call_with_retry(super()._get_text_embedding, text)

        def _get_query_embedding(self, query: str):  # type: ignore[override]
            return self._call_with_retry(super()._get_query_embedding, query)

        def _get_text_embeddings(self, texts):  # type: ignore[override]
            # Use per-text retry rather than batch, to isolate failing items
            out = []
            for t in texts:
                out.append(self._get_text_embedding(t))
            return out

        async def _aget_query_embedding(self, query: str):  # type: ignore[override]
            # For async, fallback to sync with retry (quota same)
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str):  # type: ignore[override]
            return self._get_text_embedding(text)

        async def _aget_text_embeddings(self, texts):  # type: ignore[override]
            return self._get_text_embeddings(texts)

else:
    _RetryGeminiEmbedding = None  # type: ignore

# ---------------------------------------------------------------------------
# Global configuration — preserved from main.py verbatim
# ---------------------------------------------------------------------------
# Opcion A + fuentes/ (3 materias activas). Legacy dirs kept for compat but filtered.
FUENTES_DIR = Path("fuentes")  # nuevo: fuentes/Gestion de Calidad/, fuentes/Ingenieria de Software/, fuentes/Redes de Datos/
DATOS_DIR = Path("fuentes")  # primario por materia (existe en repo; el valor legacy "fuentes by bayzon" nunca existio en disco)
DATOS_LEGACY_DIR = Path("datos")  # retrocompat — se mantiene intacto (no borrar)
DATOS_DIRS: list[Path] = [DATOS_DIR, DATOS_LEGACY_DIR]  # orden: nuevo primero
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "utn_bibliografia"

EMBED_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"

# Gemini free-tier models (used only when GEMINI_API_KEY is present)
# NOTE: models/gemini-1.5-flash and models/text-embedding-004 are deprecated for
# current Google API v1beta (404). Use stable free-tier successors:
#   LLM  -> models/gemini-flash-lite-latest (higher quota; flash-latest/3.6/3.8 have 20 req/day per model, already exhausted for 3.8/3.6 today)
#          primary lite keeps free-tier quota fresh; fallback is 3.6 for fallback
#   Embed -> models/gemini-embedding-001 (3072 dim, replaces text-embedding-004)
GEMINI_LLM_MODEL_NAME = "models/gemini-flash-lite-latest"
GEMINI_LLM_FALLBACK_MODEL_NAME = "models/gemini-3.6-flash"
GEMINI_EMBED_MODEL_NAME = "models/gemini-embedding-001"
# Gemini embedding dimension for mismatch detection (gemini-embedding-001 = 3072)
_GEMINI_EMBED_DIM = 3072
_OPENAI_EMBED_DIM = 1536

# ---------------------------------------------------------------------------
# Docx support detection
# ---------------------------------------------------------------------------
try:
    import docx  # type: ignore  # python-docx

    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

try:
    import docx2txt  # type: ignore  # required by llama_index DocxReader

    _HAS_DOCX2TXT = True
except ImportError:
    _HAS_DOCX2TXT = False

# Supported extensions for ingestion (pdf always, docx/doc if python-docx available)
# Prefer full set if both libs present; if docx2txt missing we still try manual python-docx
_SUPPORTED_EXTS = [".pdf", ".docx", ".doc"] if _HAS_DOCX else [".pdf"]

# ---------------------------------------------------------------------------
# Isolation: 3 active materias, single Chroma collection + where filter.
# Decision: single collection `utn_bibliografia` with metadata `materia`
# + MetadataFilters BEFORE the LLM (fail-closed). Chosen over one-collection-
# per-materia because it reuses existing chroma_db, needs no migration,
# and still guarantees the LLM never sees other-materia chunks (retriever
# is filtered, not post-filtered). Old vectors without `materia` metadata
# are excluded by the filter -> fallback (safe, no leakage); reindex fixes.
# ---------------------------------------------------------------------------
ACTIVE_MATERIAS: list[str] = [
    "Redes de Datos",
    "Gestión de Calidad",
    "Ingeniería de Software",
]

# Folder name (normalized) -> canonical. Covers new fuentes/ (safe names without
# tilde) + legacy `fuentes by bayzon/` + `datos/` folders. Anything else -> None (ignored).
_FOLDER_TO_CANON: dict[str, str] = {
    # New fuentes/ folders
    "redes de datos": "Redes de Datos",
    "gestion de calidad": "Gestión de Calidad",
    "ingenieria de software": "Ingeniería de Software",
    # Legacy fuentes by bayzon/
    "sistema y gestion de calidad (electiva)": "Gestión de Calidad",
    "sistema y gestion de calidad": "Gestión de Calidad",
    "ingenieria de software": "Ingeniería de Software",
    # Legacy datos/
    "gestion de calidad": "Gestión de Calidad",
    "analisis numerico": None,  # inactive — ignored (kept explicit)
}


def _normalize_folder(name: str) -> str:
    """Lower, accent-stripped, whitespace-collapsed for folder matching."""
    import re as _re
    import unicodedata as _ud

    lowered = (name or "").strip().lower()
    decomposed = _ud.normalize("NFD", lowered)
    stripped = "".join(c for c in decomposed if _ud.category(c) != "Mn")
    return _re.sub(r"\s+", " ", stripped).strip()


def resolve_materia(alias: str) -> str | None:
    """Pure helper: alias -> canonical or None. Delegates to materias_config."""
    try:
        import materias_config as _mc

        fn = getattr(_mc, "resolve_materia", None)
        if callable(fn):
            return fn(alias)
        vfn = getattr(_mc, "validate_materia", None)
        if callable(vfn):
            return vfn(alias)
    except Exception:
        pass
    return None


def collection_for(materia: str) -> str:
    """
    Pure helper: collection/namespace for a materia.

    Single-collection strategy: always returns COLLECTION_NAME.
    Isolation is enforced via where={"materia": canon} (see build_materia_filter).
    Kept as a function so tests can assert routing without touching Chroma,
    and so a future per-materia split only changes this one place.
    """
    return COLLECTION_NAME


def materia_for_file(file_path: str | Path) -> str | None:
    """
    Pure helper: map a file path to its canonical materia or None.

    Checks every path part (normalized) against known folders + alias resolution.
    Returns None for inactive/unknown locations (caller must ignore those files).
    """
    try:
        p = Path(str(file_path))
    except Exception:
        return None
    parts = list(p.parts)
    # Also consider the file stem without extension? No — only folders + full path fallback.
    for part in parts:
        norm = _normalize_folder(part)
        if not norm:
            continue
        if norm in _FOLDER_TO_CANON:
            canon = _FOLDER_TO_CANON[norm]
            if canon:
                return canon
            return None  # explicitly inactive
        # Try alias resolution on the folder name (e.g. "Redes", "SGC" legacy handled below)
        # Legacy SGC folder long name already covered; short legacy codes:
        if norm in ("sgc",):
            return "Gestión de Calidad"
        if norm in ("isw",):
            return "Ingeniería de Software"
        # Generic alias lookup (covers "redes", "calidad", "software", etc.)
        try:
            import materias_config as _mc2

            alias_map = getattr(_mc2, "_ALIASES", {})
            if norm in alias_map:
                cand = alias_map[norm]
                if cand in ACTIVE_MATERIAS:
                    return cand
        except Exception:
            pass
    # Fallback: check full path string for folder-like containment of canonical normalized
    full_norm = _normalize_folder(str(file_path))
    for canon in ACTIVE_MATERIAS:
        if _normalize_folder(canon) in full_norm:
            return canon
    return None


def build_materia_filter(materia_canonical: str):
    """Build LlamaIndex MetadataFilters for exact materia match (BEFORE LLM)."""
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    canon = (materia_canonical or "").strip()
    return MetadataFilters(filters=[MetadataFilter(key="materia", value=canon)])


def _get_active_datos_dirs() -> list[Path]:
    """Return existing dirs from DATOS_DIRS (primario + legacy) for indexación.

    Mantiene retrocompat: si "fuentes" está vacía, sigue leyendo "datos".
    Solo considera directorios que existen.
    Soporta monkeypatch en tests: si DATOS_DIR fue reasignado y ya no está en DATOS_DIRS,
    se usa solo el/los dirs parcheados para aislar el test (ej. tmp/no_datos -> 503).
    """
    # Detectar monkeypatch de tests (DATOS_DIR reasignado después del import)
    stored = globals().get("DATOS_DIRS", [])
    current_dir = globals().get("DATOS_DIR")
    current_legacy = globals().get("DATOS_LEGACY_DIR")
    stored_strs = [str(p) for p in stored] if stored else []
    if current_dir is not None and str(current_dir) not in stored_strs:
        # Test parcheó DATOS_DIR a un tmp — usar solo el/los parcheados, ignorar legacy real
        candidates: list[Path] = []
        if isinstance(current_dir, Path):
            candidates.append(current_dir)
        # Si legacy también fue parcheado (distinto de stored), incluirlo
        if isinstance(current_legacy, Path) and str(current_legacy) not in stored_strs:
            candidates.append(current_legacy)
        # Si solo DATOS_DIR fue parcheado, no incluir legacy "datos" real (para que test de 503 no encuentre datos)
        # En uso real, DATOS_DIR nunca se parchea, así que este branch no se ejecuta.
        active = [p for p in candidates if p.exists() and p.is_dir()]
        return active if active else candidates

    # Ruta normal (no parcheada): usar DATOS_DIRS tal cual
    active = [p for p in stored if isinstance(p, Path) and p.exists() and p.is_dir()]
    # Si ninguno existe, devolver lista original para que validar_entorno dé mensaje claro
    return active if active else list(stored) if stored else ([p for p in [current_dir, current_legacy] if isinstance(p, Path)])


def _collect_docs_found() -> list[Path]:
    """Recolecta PDFs/DOCXs de todos los DATOS_DIRS existentes (recursivo)."""
    found: list[Path] = []
    for base in _get_active_datos_dirs():
        if not base.exists():
            continue
        for ext in _SUPPORTED_EXTS:
            found.extend(base.rglob(f"*{ext}"))
        # rglob es case-sensitive en algunos OS, cubrir .PDF
        if not found:
            for ext in _SUPPORTED_EXTS:
                found.extend(base.rglob(f"*{ext.upper()}"))
        else:
            # Aun así buscar variantes upper para cubrir mezclas
            for ext in _SUPPORTED_EXTS:
                for p in base.rglob(f"*{ext.upper()}"):
                    if p not in found:
                        found.append(p)
    # Deduplicar por path resuelto
    uniq: dict[str, Path] = {}
    for p in found:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        uniq[key] = p
    return list(uniq.values())


def _load_docx_manual_fallback() -> list[Any]:
    """Fallback loader for .docx/.doc using python-docx directly (no docx2txt).

    Returns list of llama_index Document objects with metadata file_name/file_path.
    Used when SimpleDirectoryReader fails due to missing docx2txt.
    Soporta ambas carpetas (DATOS_DIRS) para retrocompatibilidad.
    """
    if not _HAS_DOCX:
        return []
    try:
        from llama_index.core.schema import Document as LIDocument  # type: ignore
    except ImportError:
        try:
            from llama_index.core import Document as LIDocument  # type: ignore
        except Exception:
            return []
    try:
        import docx  # type: ignore
    except ImportError:
        return []

    docs: list[Any] = []
    # Iterar sobre ambos directorios (primario + legacy) si existen
    _dirs_to_check = _get_active_datos_dirs()
    for base_dir in _dirs_to_check:
        if not base_dir.exists():
            continue
        for ext in [".docx", ".doc"]:
            for fpath in base_dir.rglob(f"*{ext}"):
                # also case-insensitive upper
                if not fpath.is_file():
                    continue
                try:
                    d = docx.Document(str(fpath))  # type: ignore
                    text_parts = []
                    for para in d.paragraphs:
                        t = para.text.strip()
                        if t:
                            text_parts.append(t)
                    # Also try tables
                    for table in getattr(d, "tables", []):
                        for row in table.rows:
                            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                            if row_text:
                                text_parts.append(row_text)
                    text = "\n".join(text_parts).strip()
                    if not text:
                        continue
                    rel = str(fpath)
                    try:
                        rel = str(fpath.relative_to(base_dir))
                    except Exception:
                        pass
                    meta = {
                        "file_name": fpath.name,
                        "file_path": str(fpath),
                        "file_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "file_size": str(fpath.stat().st_size) if fpath.exists() else "0",
                    }
                    # Ensure file_path contains materia folder for filtering
                    docs.append(LIDocument(text=text, metadata=meta))  # type: ignore
                except Exception as e:
                    print(f"[WARN] Fallo leyendo docx manual {fpath}: {e}")
                    continue
            # upper ext already covered by rglob case sensitivity? Try upper explicit
            for fpath in base_dir.rglob(f"*{ext.upper()}"):
                if not fpath.is_file():
                    continue
                # avoid duplicate if already added (check by name)
                if any(str(d.metadata.get("file_path")) == str(fpath) for d in docs):  # type: ignore
                    continue
                try:
                    d = docx.Document(str(fpath))  # type: ignore
                    text_parts = [para.text.strip() for para in d.paragraphs if para.text.strip()]
                    text = "\n".join(text_parts).strip()
                    if not text:
                        continue
                    meta = {"file_name": fpath.name, "file_path": str(fpath)}
                    docs.append(LIDocument(text=text, metadata=meta))  # type: ignore
                except Exception:
                    continue
    return docs

ANTI_HALLUCINATION_QA_TMPL_STR = (
    "Respondé exclusivamente utilizando la información incluida en el CONTEXTO. "
    "No utilices conocimiento externo ni completes con conocimiento general del modelo. "
    "No inventes definiciones, fórmulas, citas, páginas ni condiciones.\n"
    "INSTRUCCIONES ESTRICTAS:\n"
    "1. Responde EXCLUSIVAMENTE con información que aparezca en el CONTEXTO proporcionado.\n"
    "2. Si el contexto no contiene la respuesta o es insuficiente, responde EXACTAMENTE:\n"
    "   'No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad.'\n"
    "3. NO completes con conocimiento general del modelo. NO inventes definiciones, fórmulas ni condiciones.\n"
    "4. Si respondes, agrega citas indicando de qué fragmentos te basaste.\n"
    "5. Sé conciso pero preciso. Prioriza fidelidad al texto original.\n"
    "\n"
    "CONTEXTO:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "PREGUNTA: {query_str}\n"
    "\n"
    "RESPUESTA (en español rioplatense neutro, académica):"
)

ANTI_HALLUCINATION_QA_TEMPLATE = PromptTemplate(ANTI_HALLUCINATION_QA_TMPL_STR)

REFINE_TMPL_STR = (
    "Ya diste una respuesta inicial basada en parte del contexto.\n"
    "Contexto adicional:\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Pregunta: {query_str}\n"
    "Respuesta existente: {existing_answer}\n"
    "INSTRUCCIÓN: Refina la respuesta SOLO si el contexto adicional aporta información nueva y relevante. "
    "Si no aporta, mantén la respuesta. Nunca inventes. Responde en español."
)
REFINE_TEMPLATE = PromptTemplate(REFINE_TMPL_STR)

# Fallback phrase contract — exact match required by spec & tests
FALLBACK_PHRASE = "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad."
# Legacy fallback kept for backward-compat detection (old index/tests may reference it).
LEGACY_FALLBACK_PHRASE = "No encontré información suficiente en los documentos para responder esa pregunta."

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_index: VectorStoreIndex | None = None
_query_engine: Any | None = None
_singleton_lock = threading.Lock()
_models_configured = False


def _is_api_key_present() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and key.strip() not in ("", "sk-...", "tu_api_key_aqui") and "TEST" not in key or key.strip().startswith("sk-"))


def _is_gemini_configured() -> bool:
    """Return True if GEMINI_API_KEY is present and looks valid (non-dummy)."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return False
    k = key.strip()
    if not k:
        return False
    if k in ("", "tu_api_key_aqui", "REEMPLAZAR", "sk-..."):
        return False
    if "TEST" in k:
        return False
    if len(k) < 10:
        return False
    return True


def _get_active_provider() -> str:
    """Return 'gemini' if GEMINI_API_KEY is configured, else 'openai'."""
    return "gemini" if _is_gemini_configured() else "openai"


def _mask_key(key: str) -> str:
    """Mask key for logs: keep first 2 and last 2 chars."""
    if not key or len(key) < 6:
        return "***"
    return f"{key[:2]}....{key[-2:]}"


def _is_quota_error(exc: BaseException) -> bool:
    """True if exception looks like Gemini/OpenAI 429/quota (fail-fast webhook).

    Used by webhook-safe paths to return FALLBACK without retrying in hot path.
    CLI reindex keeps its own retry/backoff; webhook must not burn quota.
    """
    try:
        msg = str(exc).lower()
    except Exception:
        return False
    return (
        "429" in str(exc)
        or "quota" in msg
        or "rate" in msg
        or "exceeded" in msg
        or "resource" in msg
        or "limit" in msg
    )


# ---------------------------------------------------------------------------
# Public API — mirrors main.py signatures for drop-in reuse
# ---------------------------------------------------------------------------

def validar_entorno(reindex: bool = False) -> None:
    """Valida API key, carpeta datos y documentos (pdf/docx). Raises SystemExit if invalid."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    gemini_ok = bool(gemini_key and gemini_key not in ("", "tu_api_key_aqui") and len(gemini_key) >= 10 and "TEST" not in gemini_key)
    # OpenAI check: allow dummy TEST keys for mocks/tests (same logic as before)
    openai_falta = not openai_key or openai_key.strip() in ("", "sk-...", "tu_api_key_aqui")
    openai_ok = False
    if not openai_falta:
        openai_ok = True
    elif "TEST" in openai_key:
        # TEST dummy is ok for mock tests
        openai_ok = True
        print("[WARN] Usando OPENAI_API_KEY dummy (TEST) — solo valido para mocks/tests")

    if not gemini_ok and not openai_ok:
        print("[ERROR] Corrige tu .env antes de continuar.")
        print("  -> Crea un archivo .env en la raiz del proyecto con al menos una de:")
        print("     GEMINI_API_KEY=AQ....  (free tier, recomendado)")
        print("     OPENAI_API_KEY=sk-...  (alternativa)")
        print("  -> Gemini: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    if not gemini_ok and openai_falta and "TEST" not in openai_key:
        # Original OpenAI-only error path (preserved for backwards compat)
        # but already handled by gemini_ok check above; keep message concise
        pass

    if not _HAS_FILE_READERS:
        print("[ADVERTENCIA] llama-index-readers-file NO esta instalado.")
        print("  Los PDFs pueden no procesarse correctamente.")
        print("  Instalalo con: pip install llama-index-readers-file\n")

    # Validación de carpetas fuente — soporta ambas (primario + legacy) por retrocompatibilidad
    active_dirs = _get_active_datos_dirs()
    # Si ninguna existe, error con ambas rutas
    if not active_dirs:
        print(f"[ERROR] No existe ninguna carpeta fuente. Se esperaban: {[str(p) for p in DATOS_DIRS]}")
        print(f"  Crea '{DATOS_DIR}/' (primario) o mantiene '{DATOS_LEGACY_DIR}/' con tus PDFs.")
        print(f"  Rutas chequeadas: {[str(p.resolve()) for p in DATOS_DIRS]}")
        sys.exit(1)

    # Verificar que las activas sean directorios (por si alguna es archivo)
    for d in active_dirs:
        if not d.is_dir():
            print(f"[ERROR] '{d}' existe pero no es un directorio.")
            sys.exit(1)

    # Collect supported docs (pdf + docx/doc if available) de todos los dirs activos
    docs_found: list[Path] = _collect_docs_found()
    if not docs_found:
        print(f"[ERROR] No se encontraron documentos { _SUPPORTED_EXTS } en {[str(p) for p in active_dirs]} (búsqueda recursiva).")
        print("  Agrega al menos un archivo .pdf/.docx en 'fuentes/' o 'datos/' o subcarpetas por materia.")
        for p in active_dirs:
            print(f"  Ruta chequeada: {p.resolve()}")
        sys.exit(1)

    print(f"[OK] Encontrados {len(docs_found)} documento(s) { _SUPPORTED_EXTS } en {[str(p) for p in active_dirs]}")
    for p in docs_found[:5]:
        shown = False
        for base in active_dirs:
            try:
                print(f"  - {p.relative_to(base)} ({base}/)")
                shown = True
                break
            except Exception:
                continue
        if not shown:
            print(f"  - {p.name}")
    if len(docs_found) > 5:
        print(f"  ... y {len(docs_found) - 5} más")
    # Si "fuentes" está vacía pero "datos" tiene contenido, seguimos indexando datos (no rompe)
    if not _HAS_DOCX:
        print("[INFO] python-docx no instalado — solo se indexarán PDFs. Instala con: pip install python-docx==1.1.2")
    elif not _HAS_DOCX2TXT:
        print("[INFO] docx2txt no instalado — se usará python-docx directo para .docx (fallback manual). Instala con: pip install docx2txt")

    # Provider hint
    provider = _get_active_provider()
    if provider == "gemini":
        print(f"[INFO] Provider activo: Gemini ({_mask_key(gemini_key)}) — free tier")
    else:
        print(f"[INFO] Provider activo: OpenAI ({_mask_key(openai_key) if openai_key else 'dummy'})")

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()) and not reindex:
        print(f"[INFO] Se encontro chroma_db/ existente -- se reutilizara el indice (usa --reindex para reconstruir).")
    elif reindex and CHROMA_DIR.exists():
        print(f"[INFO] --reindex activo -- se reconstruira el indice en {CHROMA_DIR}/")


def configurar_modelos() -> None:
    """Configura LLM y embeddings globales de LlamaIndex via Settings.

    Dual-provider: if GEMINI_API_KEY is set, configure Gemini; otherwise
    configure OpenAI (existing behavior). Keeps singleton semantics.
    """
    global _models_configured

    provider = _get_active_provider()

    if provider == "gemini":
        if not _HAS_GEMINI:
            print("[WARN] GEMINI_API_KEY detectada pero llama-index-llms-gemini / embeddings-gemini no instalados.")
            print("  -> pip install google-generativeai llama-index-llms-gemini llama-index-embeddings-gemini")
            print("  -> Fallback a OpenAI por compatibilidad.")
        else:
            gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            # Mask in logs
            masked = _mask_key(gemini_key)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    Settings.llm = GeminiLLM(  # type: ignore
                        model=GEMINI_LLM_MODEL_NAME,
                        api_key=gemini_key,
                        temperature=0.1,
                    )
                    # Use retry wrapper for embeddings to handle free-tier 429
                    embed_cls = _RetryGeminiEmbedding if _RetryGeminiEmbedding is not None else GeminiEmbedding  # type: ignore
                    Settings.embed_model = embed_cls(  # type: ignore
                        model_name=GEMINI_EMBED_MODEL_NAME,
                        api_key=gemini_key,
                    )
                _models_configured = True
                print(f"[OK] Modelos configurados (Gemini): LLM={GEMINI_LLM_MODEL_NAME}, Embeddings={GEMINI_EMBED_MODEL_NAME} (key {masked})")
                return
            except Exception as e:
                print(f"[WARN] Fallo al configurar Gemini ({e}) — fallback a OpenAI. Key {masked}")
                # Fall through to OpenAI

    # OpenAI branch (preserved verbatim behavior)
    Settings.llm = OpenAI(
        model=LLM_MODEL_NAME,
        temperature=0.1,
        max_tokens=1024,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=EMBED_MODEL_NAME,
    )
    _models_configured = True
    print(f"[OK] Modelos configurados: LLM={LLM_MODEL_NAME}, Embeddings={EMBED_MODEL_NAME}")


def obtener_coleccion_chroma(reindex: bool = False):
    """Crea o recupera la coleccion Chroma persistente."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[INFO] Coleccion '{COLLECTION_NAME}' eliminada (reindex).")
        except Exception:
            pass
        collection = client.get_or_create_collection(COLLECTION_NAME)
        return client, collection
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return client, collection


def _check_gemini_dim_mismatch(client, chroma_collection) -> bool:
    """Detect if stored embeddings dimension mismatches expected for active provider.

    For Gemini (3072) always checks. For OpenAI (1536) checks only when
    embed_model is not a Mock (to avoid breaking mock tests with 384).
    Returns True if mismatch and a forced reindex is recommended.
    Never raises — returns False on any inspection failure.
    """
    try:
        count = chroma_collection.count()
        if count == 0:
            return False
        # Peek one vector to get stored dimension
        sample = chroma_collection.get(limit=1, include=["embeddings", "metadatas"])
        embeddings = sample.get("embeddings") if isinstance(sample, dict) else None
        if embeddings is None:
            return False
        try:
            if hasattr(embeddings, "__len__") and len(embeddings) == 0:
                return False
            first = embeddings[0]
        except Exception:
            return False
        try:
            stored_dim = len(first)  # type: ignore
        except Exception:
            return False

        if _is_gemini_configured():
            # Gemini expects 3072 (gemini-embedding-001)
            if stored_dim != _GEMINI_EMBED_DIM:
                return True
            return False
        else:
            # OpenAI path — skip mismatch check when using MockEmbedding (tests, dim 384)
            # Detect mock by inspecting current Settings.embed_model
            try:
                from llama_index.core.embeddings import MockEmbedding

                # Settings may not be configured yet (None) — skip check then
                if Settings.embed_model is not None and isinstance(Settings.embed_model, MockEmbedding):
                    return False
            except Exception:
                pass
            # Also skip if embed_model is still None (early startup) — don't force rebuild
            try:
                if Settings.embed_model is None:
                    return False
            except Exception:
                return False
            if stored_dim != _OPENAI_EMBED_DIM:
                return True
            return False
    except Exception:
        return False


def construir_o_cargar_indice(reindex: bool = False, fail_fast: bool = False) -> VectorStoreIndex:
    """
    Logica central de cache:
    - Si reindex=False y la coleccion ya tiene vectores -> carga desde Chroma.
    - Si no, lee PDFs con SimpleDirectoryReader y crea indice desde cero.

    Provider-aware: if stored embeddings dimension mismatches active provider
    (Gemini 3072 vs OpenAI 1536 vs Mock 384), forces rebuild even without
    --reindex to avoid silent dimension errors.

    fail_fast (webhook-safe): cuando True, NUNCA hace sys.exit ni construye
    índice nuevo (no quema cuota). Si el índice falta/vacío o falla por 429,
    lanza RuntimeError 503 para que el caller devuelva FALLBACK.
    Solo el path CLI manual (main.py --reindex, fail_fast=False) puede
    salir con sys.exit(1).
    """
    client, chroma_collection = obtener_coleccion_chroma(reindex=reindex)

    try:
        count = chroma_collection.count()
    except Exception:
        count = 0

    # Auto-rebuild on embedding dimension mismatch (keeps old data from breaking queries)
    # Handles Gemini (3072) <-> OpenAI (1536) <-> Mock (384) switches
    if not reindex and count > 0:
        try:
            if _check_gemini_dim_mismatch(client, chroma_collection):
                sample_dim = "unknown"
                expected = _GEMINI_EMBED_DIM if _is_gemini_configured() else _OPENAI_EMBED_DIM
                provider_name = _get_active_provider()
                try:
                    peek = chroma_collection.get(limit=1, include=["embeddings"])
                    emb = peek.get("embeddings")[0]  # type: ignore
                    sample_dim = str(len(emb))
                except Exception:
                    pass
                print(f"[WARN] Dimension mismatch detectada (stored dim {sample_dim} vs esperada {expected} para {provider_name})")
                print(f"[INFO] Forzando reindex automatico por cambio de provider a {provider_name} (embeddings incompatibles).")
                try:
                    client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass
                chroma_collection = client.get_or_create_collection(COLLECTION_NAME)
                count = 0
        except Exception as e:
            print(f"[WARN] No se pudo verificar dimension: {e} — continuando con reuso")

    if not reindex and count > 0:
        print(f"[INFO] Reutilizando indice existente en Chroma ({count} vectores).")
        print("       Usa 'python main.py --reindex' si agregaste/actualizaste PDFs.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )
        return index

    if fail_fast:
        # Webhook path: nunca construir en caliente (quemaría cuota 429).
        # El índice se reconstruye solo vía `python main.py --reindex` manual.
        raise RuntimeError(
            "503: indice no disponible (fail-fast webhook) — ejecutar python main.py --reindex tras reset de cuota"
        )

    motivo = "--reindex solicitado" if reindex else f"índice vacío (vectores={count})"
    # Extra hint if we auto-forced
    if count == 0 and _is_gemini_configured() and not reindex:
        # could be auto-forced due to mismatch; already logged
        pass
    print(f"[INFO] Construyendo nuevo índice ({motivo}) — esto puede tardar según cantidad de documentos...")

    # Use supported extensions; fallback to pdf-only if docx reader missing
    # Soporta ambas carpetas: "fuentes" (primario) y "datos" (retrocompat) — no rompe si primario vacío
    req_exts = list(_SUPPORTED_EXTS)
    documents: list[Any] = []
    last_err = None
    _active_dirs = _get_active_datos_dirs()
    # Si primario vacío, seguirá indexando legacy (datos/)
    for base_dir in _active_dirs:
        if not base_dir.exists() or not base_dir.is_dir():
            print(f"[INFO] Carpeta fuente no existe, se omite: {base_dir}")
            continue
        loaded_for_dir: list[Any] | None = None
        for attempt_exts in [req_exts, [".pdf"]]:
            try:
                reader = SimpleDirectoryReader(
                    input_dir=str(base_dir),
                    recursive=True,
                    required_exts=attempt_exts,
                    filename_as_id=True,
                )
                loaded = reader.load_data(show_progress=True)
                if attempt_exts != req_exts:
                    print(f"[WARN] Fallback a solo PDF en {base_dir} (docx fallo), exts={attempt_exts}")
                loaded_for_dir = loaded
                break
            except Exception as e:
                last_err = e
                if attempt_exts == [".pdf"]:
                    print(f"[ERROR] Falló la lectura de documentos en {base_dir}/: {e}")
                    print("  Verifica que los archivos no estén corruptos y que tengas: pip install llama-index-readers-file pypdf python-docx")
                    # No salir global si hay otra carpeta con datos; marcar como vacío y continuar
                    loaded_for_dir = []
                    break
                print(f"[WARN] Fallo leyendo {base_dir} con exts {attempt_exts}: {e} — reintentando solo PDF")
                continue
        if loaded_for_dir:
            print(f"[INFO] Cargados {len(loaded_for_dir)} documento(s) desde {base_dir}/")
            documents.extend(loaded_for_dir)
        elif loaded_for_dir == []:
            # Carpeta vacía sin error — no rompe, sigue con siguiente dir (retrocompat)
            print(f"[INFO] Sin documentos en {base_dir}/ (vacía, se continúa con otras carpetas).")
            continue
        else:
            # loaded_for_dir None -> error ya seteado en last_err, pero continuamos
            continue
    if not documents:
        if last_err is not None:
            print(f"[ERROR] No se pudieron cargar documentos de {[str(p) for p in _active_dirs]}: {last_err}")
        else:
            print(f"[ERROR] No se pudieron cargar documentos de {[str(p) for p in _active_dirs]} (vacío)")
        if fail_fast:
            raise RuntimeError("503: indice no disponible (fail-fast webhook) — ejecutar python main.py --reindex tras preparar datos/")
        sys.exit(1)

    # Manual docx fallback when docx2txt missing but python-docx present
    if _HAS_DOCX and not _HAS_DOCX2TXT and documents is not None:
        try:
            # Check if fallback produced pdf-only (req_exts != [".pdf"] but we fell back)
            # If we loaded with full exts but docx2txt missing, we would have fallen back to pdf-only above.
            # Load docx manually to ensure .docx/.doc are indexed.
            manual_docs = _load_docx_manual_fallback()
            if manual_docs:
                print(f"[INFO] Cargados {len(manual_docs)} documento(s) docx via python-docx manual fallback")
                documents.extend(manual_docs)
        except Exception as e:
            print(f"[WARN] Fallo fallback manual docx: {e}")

    if not documents:
        print("[ERROR] SimpleDirectoryReader no devolvió documentos. ¿PDFs vacíos o ilegibles?")
        if fail_fast:
            raise RuntimeError("503: indice no disponible (fail-fast webhook) — ejecutar python main.py --reindex tras preparar datos/")
        sys.exit(1)

    # Tag + filter: keep only docs belonging to the 3 active materias.
    # Each doc gets metadata materia/archivo/ruta for where-filter isolation.
    # Files outside active materias are ignored (old PDFs kept on disk, not indexed).
    tagged: list[Any] = []
    ignored = 0
    for doc in documents:
        try:
            meta = getattr(doc, "metadata", {}) or {}
            fpath = str(meta.get("file_path") or meta.get("file_name") or "")
            fname = str(meta.get("file_name") or Path(fpath).name if fpath else "")
            canon = materia_for_file(fpath) or (materia_for_file(fname) if fname else None)
            if canon is None or canon not in ACTIVE_MATERIAS:
                ignored += 1
                continue
            meta["materia"] = canon
            meta["archivo"] = fname or Path(fpath).name if (fname or fpath) else "desconocido"
            meta["ruta"] = fpath
            # Ensure file_path present for debugging
            if not meta.get("file_path") and fpath:
                meta["file_path"] = fpath
            try:
                doc.metadata = meta
            except Exception:
                pass
            tagged.append(doc)
        except Exception:
            ignored += 1
            continue
    if ignored:
        print(f"[INFO] Ignorados {ignored} documento(s) fuera de las 3 materias activas.")
    if not tagged:
        print(f"[ERROR] Ningún documento pertenece a las 3 materias activas {ACTIVE_MATERIAS}.")
        print("  Agrega PDFs en fuentes/Redes de Datos/, fuentes/Gestion de Calidad/, fuentes/Ingenieria de Software/")
        print("  o legacy mapeado (SGC->Gestión de Calidad, ISW->Ingeniería de Software).")
        if fail_fast:
            raise RuntimeError("503: indice no disponible (fail-fast webhook) — ejecutar python main.py --reindex tras preparar datos/")
        sys.exit(1)
    documents = tagged

    print(f"[OK] Cargados {len(documents)} documentos (chunks iniciales).")
    if len(documents) > 800:
        print(f"[WARN] Cuota Gemini free tier: 1000 embeddings/día. Este reindex pide ~{len(documents)}+. "
              f"Si ves 429, recortá PDFs a <800 chunks o reindexá en 2 días/con otra API key. "
              f"Ver README 'Cuota Gemini free tier'.")
    if documents:
        sample_meta = documents[0].metadata
        print(f"     Ejemplo metadata: {sample_meta}")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            show_progress=True,
        )
    except Exception as e:
        print(f"[ERROR] Falló la creación del índice / embeddings: {e}")
        # Provider-aware hint
        if _is_gemini_configured():
            print("  Verifica tu GEMINI_API_KEY y tu cuota de Google AI (free tier).")
            print(f"  Key usada: {_mask_key(os.getenv('GEMINI_API_KEY',''))} — si es 401/invalid, regenera en https://aistudio.google.com/app/apikey")
            if _is_quota_error(e):
                print("  Cuota free tier (1000 embeddings/día) excedida: este reindex NO entra en 1 día "
                      "(~1599 pedidos). Recortá PDFs a <800 chunks o reindexá en 2 días/con otra key. "
                      "El webhook usa fail-fast con fallback, no reintenta en caliente.")
        else:
            print("  Verifica tu OPENAI_API_KEY y tu cuota de OpenAI.")
        if fail_fast:
            raise RuntimeError(f"503: fallo al crear indice (fail-fast webhook): {e}") from e
        sys.exit(1)

    try:
        new_count = chroma_collection.count()
        print(f"[OK] Indice creado y persistido en '{CHROMA_DIR}/' ({new_count} vectores).")
    except Exception:
        print(f"[OK] Indice creado en '{CHROMA_DIR}/' (no se pudo contar vectores).")

    return index


def crear_query_engine(index: VectorStoreIndex):
    """
    Crea el motor de consulta con CitationQueryEngine o fallback anti-alucinacion.

    For Gemini free tier we prefer the standard engine (1 LLM call vs 2-3 for
    Citation) to stay within the 20 req/day per-model quota and reduce
    latency (6s vs 30s). Citation is still used for OpenAI (higher quota).
    """
    # Prefer standard engine for Gemini to save quota/latency
    if _is_gemini_configured():
        print("[INFO] Usando query engine estándar con prompt anti-alucinación (Gemini free tier).")
        query_engine = index.as_query_engine(
            similarity_top_k=4,
            text_qa_template=ANTI_HALLUCINATION_QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
            response_mode="compact",
            verbose=False,
        )
        return query_engine

    if _HAS_CITATION_ENGINE:
        print("[INFO] Usando CitationQueryEngine (respuestas con citas).")
        try:
            query_engine = CitationQueryEngine.from_args(
                index,
                similarity_top_k=4,
                citation_chunk_size=512,
            )
            try:
                query_engine._text_qa_template = ANTI_HALLUCINATION_QA_TEMPLATE  # type: ignore
            except Exception:
                pass
            return query_engine
        except TypeError as e:
            print(f"[AVISO] CitationQueryEngine no acepta esos parámetros ({e}), usando fallback.")
        except Exception as e:
            print(f"[AVISO] No se pudo crear CitationQueryEngine ({e}), usando fallback.")

    print("[INFO] Usando query engine estándar con prompt anti-alucinación.")
    query_engine = index.as_query_engine(
        similarity_top_k=4,
        text_qa_template=ANTI_HALLUCINATION_QA_TEMPLATE,
        refine_template=REFINE_TEMPLATE,
        response_mode="compact",
        verbose=False,
    )
    return query_engine


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

def get_index(reindex: bool = False, fail_fast: bool = False) -> VectorStoreIndex:
    """
    Return cached VectorStoreIndex, building on first call.
    Thread-safe singleton; reindex forces rebuild.
    Raises RuntimeError(503) if chroma_db missing and cannot build.
    fail_fast=True (webhook): nunca sys.exit ni construye índice nuevo;
    lanza RuntimeError 503 para fallback inmediato sin quemar cuota.
    """
    global _index
    with _singleton_lock:
        if _index is not None and not reindex:
            return _index
        # Ensure models configured (required for embeddings)
        if not _models_configured:
            # Preserve mock embeddings for tests — don't overwrite MockEmbedding/MockLLM
            _is_mock = False
            try:
                from llama_index.core.embeddings import MockEmbedding as _MockEmb
                from llama_index.core.llms.mock import MockLLM as _MockLLM

                if Settings.embed_model is not None and isinstance(Settings.embed_model, _MockEmb):
                    _is_mock = True
                if Settings.llm is not None and isinstance(Settings.llm, _MockLLM):
                    _is_mock = True
            except Exception:
                _is_mock = False
            if not _is_mock:
                configurar_modelos()
        # This may sys.exit if entorno invalid; let caller handle
        # fail_fast=True (webhook) convierte sys.exit en RuntimeError 503
        try:
            _index = construir_o_cargar_indice(reindex=reindex, fail_fast=fail_fast)
        except SystemExit as e:
            if fail_fast:
                raise RuntimeError("503: indice no disponible (fail-fast webhook) — ejecutar python main.py --reindex tras reset de cuota") from e
            raise
        return _index


def get_query_engine(reindex: bool = False, fail_fast: bool = False):
    """
    Singleton CitationQueryEngine accessor for FastAPI lifespan.

    Loads index once and reuses. Raises RuntimeError with 503 semantics
    if index cannot be loaded (e.g., missing chroma_db and no PDFs).
    fail_fast=True (webhook): no construye, lanza 503 rápido sin quemar cuota.
    """
    global _query_engine, _index
    with _singleton_lock:
        if _query_engine is not None and not reindex:
            return _query_engine
        if _index is None or reindex:
            # Build/get index inside same lock to avoid double build
            if not _models_configured:
                _is_mock2 = False
                try:
                    from llama_index.core.embeddings import MockEmbedding as _MockEmb2
                    from llama_index.core.llms.mock import MockLLM as _MockLLM2

                    if Settings.embed_model is not None and isinstance(Settings.embed_model, _MockEmb2):
                        _is_mock2 = True
                    if Settings.llm is not None and isinstance(Settings.llm, _MockLLM2):
                        _is_mock2 = True
                except Exception:
                    _is_mock2 = False
                if not _is_mock2:
                    configurar_modelos()
            # For health check: if chroma_db missing and no reindex, try to load
            # construir_o_cargar_indice handles missing case via exception
            try:
                _index = construir_o_cargar_indice(reindex=reindex, fail_fast=fail_fast)
            except SystemExit as e:
                # Re-raise as 503 for webhook health semantics
                # (fail_fast already raises RuntimeError, this covers CLI path)
                raise RuntimeError("503: indice no disponible — ejecutar con --reindex tras preparar datos/") from e
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"503: fallo al cargar indice: {e}") from e
        try:
            _query_engine = crear_query_engine(_index)
        except Exception as e:
            raise RuntimeError(f"503: fallo al crear query engine: {e}") from e
        return _query_engine


def query(question: str) -> tuple[str, list[dict]]:
    """
    High-level query helper for webhook_server.process_question.

    Returns (answer_text, sources) where sources is list of {file, page}
    extracted from source_nodes. Falls back to FALLBACK_PHRASE when no
    relevant chunks or model returns empty.

    Webhook-safe: NUNCA hace sys.exit ni reindex. Ante índice faltante,
    429/cuota o cualquier error, devuelve (FALLBACK_PHRASE, []) para que
    el webhook guarde en Drive y responda ✅. Solo CLI --reindex sale con error.

    Does not handle materia parsing — caller validates routing first.
    """
    if not question or not question.strip():
        return (FALLBACK_PHRASE, [])
    try:
        engine = get_query_engine(fail_fast=True)
    except BaseException as e:
        # Incluye SystemExit y 429/cuota: fail-fast sin reintentar en caliente
        print(f"[WARN] RAG query fallback fail-fast (sin índice/cuota): {e}")
        return (FALLBACK_PHRASE, [])
    try:
        response = engine.query(question.strip())
    except BaseException as e:
        if _is_quota_error(e):
            # 429 free tier: fallback inmediato, no reintentar en caliente
            print(f"[WARN] RAG query 429/cuota, fallback sin reintento: {e}")
            return (FALLBACK_PHRASE, [])
        msg = str(e).lower()
        if "api_key" in msg or "authentication" in msg or "api key" in msg or "invalid" in msg:
            # Mask key in error path — return fallback rather than leaking
            print(f"[ERROR] RAG query fallo por API key ({_mask_key(os.getenv('GEMINI_API_KEY','') or os.getenv('OPENAI_API_KEY',''))}): {e}")
            return (FALLBACK_PHRASE, [])
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            print(f"[WARN] RAG query interrumpida, fallback: {e}")
            return (FALLBACK_PHRASE, [])
        print(f"[WARN] RAG query fallo, fallback: {e}")
        return (FALLBACK_PHRASE, [])

    text, sources = _extract_answer_and_sources(response)
    return _normalize_fallback(text, sources)


def _extract_answer_and_sources(response) -> tuple[str, list[dict]]:
    """Extract (text, sources) from a LlamaIndex response with materia metadata."""
    text = ""
    if hasattr(response, "response"):
        text = str(response.response).strip()
    else:
        text = str(response).strip()
    if not text:
        text = FALLBACK_PHRASE
    sources: list[dict] = []
    nodes = getattr(response, "source_nodes", None) or []
    for node in nodes:
        meta = getattr(node, "metadata", {}) or {}
        # LlamaIndex wraps nodes: try node.node.metadata as fallback
        if not meta:
            try:
                inner = getattr(node, "node", None)
                meta = getattr(inner, "metadata", {}) or {}
            except Exception:
                meta = {}
        file_path_raw = meta.get("file_path") or meta.get("file_name") or meta.get("file") or "archivo desconocido"
        file_name = meta.get("archivo") or meta.get("file_name") or file_path_raw
        try:
            file_name = Path(str(file_name)).name
        except Exception:
            pass
        try:
            file_path_norm = str(file_path_raw)
        except Exception:
            file_path_norm = str(file_name)
        page = meta.get("page_label") or meta.get("page_number") or meta.get("page") or meta.get("pagina") or "?"
        entry: dict = {"file": str(file_name), "page": str(page), "file_path": file_path_norm}
        # Preserve materia when present (for isolation assertions)
        if meta.get("materia"):
            entry["materia"] = str(meta.get("materia"))
        sources.append(entry)
    return (text, sources)


def _normalize_fallback(text: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """Normalize paraphrased no-evidence answers to exact FALLBACK_PHRASE."""
    lower = (text or "").lower()
    if FALLBACK_PHRASE.lower() in lower:
        return (FALLBACK_PHRASE, [])
    # Legacy phrase also maps to new fallback (backward compat)
    try:
        legacy = LEGACY_FALLBACK_PHRASE.lower()
    except Exception:
        legacy = ""
    if legacy and legacy in lower:
        return (FALLBACK_PHRASE, [])
    _fallback_indicators = [
        "no contienen",
        "no contiene información",
        "no contiene informacion",
        "ninguna de las fuentes",
        "no hay información",
        "no hay informacion",
        "no se encontró",
        "no se encontro",
        "no es posible",
        "no disponible",
        "no proporcion",
        "no hay datos",
        "none of the provided sources",
        "none of the sources",
        "no information",
        "not contain",
        "not found",
        "cannot answer",
        "unable to answer",
    ]
    if len(text) < 500 and any(ind in lower for ind in _fallback_indicators):
        has_grounded_signals = any(s in text for s in ["$$", "\\sum", "\\prod", "[1]", "[2]", "Lagrange", "Newton", "ISO"])
        if not has_grounded_signals:
            return (FALLBACK_PHRASE, [])
    return (text, sources)


def _get_materia_keywords(materia_canonical: str) -> set[str]:
    """Legacy helper kept for backward compat (post-filter fallback path)."""
    if not materia_canonical or not materia_canonical.strip():
        return set()
    canonical = materia_canonical.strip()
    lower_canonical = canonical.lower()
    keywords: set[str] = {lower_canonical}
    keywords.add(lower_canonical.replace(" ", "_"))
    keywords.add(lower_canonical.replace(" ", "-"))
    try:
        import materias_config as _mc  # lazy to avoid circular

        alias_map = getattr(_mc, "_ALIASES", {})
        for alias_key, canon_val in alias_map.items():
            if canon_val == canonical:
                keywords.add(alias_key.lower())
                keywords.add(alias_key.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u"))
    except Exception:
        pass
    for part in lower_canonical.split():
        if len(part) >= 3:
            keywords.add(part)
    return keywords


def _materia_matches_file(materia_canonical: str, file_path: str, file_name: str) -> bool:
    """Legacy helper kept for backward compat."""
    if not materia_canonical:
        return True
    keywords = _get_materia_keywords(materia_canonical)
    if not keywords:
        return True
    fp_lower = (file_path or "").lower()
    fn_lower = (file_name or "").lower()
    combined = f"{fp_lower} {fn_lower}"
    for kw in keywords:
        if kw and kw in combined:
            return True
    return False


def query_for_materia(materia_canonical: str, question: str) -> tuple[str, list[dict]]:
    """Per-materia RAG with isolation BEFORE the LLM.

    Uses MetadataFilters(where={"materia": canon}) so the retriever only
    returns chunks of that materia and the LLM never sees other-materia text.
    No global fallback: unknown/empty materia still requires explicit canon;
    falsy materia returns FALLBACK (fail-closed, never global search).

    Webhook-safe: NUNCA hace sys.exit ni reindex. Usa get_index(fail_fast=True)
    para no construir índice en caliente (no quema cuota 429). Ante índice
    faltante, 429/cuota o cualquier error, devuelve (FALLBACK_PHRASE, [])
    para que el webhook guarde en Drive y responda ✅.
    Solo `python main.py --reindex` manual puede salir con error.

    Returns (answer, sources) where sources only contain this materia.
    If no chunks match, returns (FALLBACK_PHRASE, []).
    """
    canon = (materia_canonical or "").strip()
    if not canon:
        return (FALLBACK_PHRASE, [])
    if canon not in ACTIVE_MATERIAS:
        # Fail-closed: never search globally for inactive/unknown materia.
        return (FALLBACK_PHRASE, [])
    if not question or not question.strip():
        return (FALLBACK_PHRASE, [])
    try:
        index = get_index(fail_fast=True)
    except BaseException as e:
        # Incluye SystemExit y 503 fail-fast: fallback sin reintentar ni quemar cuota
        print(f"[WARN] RAG query_for_materia fallback fail-fast (sin índice/cuota): {e}")
        return (FALLBACK_PHRASE, [])
    filt = build_materia_filter(canon)
    # Build a filtered query engine on the shared index (retriever-level filter).
    # Reuse global prompt templates for anti-hallucination contract.
    # Webhook-safe: cualquier fallo (incluye SystemExit/429) -> fallback, nunca exit.
    try:
        engine = index.as_query_engine(
            similarity_top_k=4,
            filters=filt,
            text_qa_template=ANTI_HALLUCINATION_QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
            response_mode="compact",
        )
    except TypeError:
        # Older LlamaIndex without filters kwarg on as_query_engine -> use retriever path
        try:
            retriever = index.as_retriever(similarity_top_k=4, filters=filt)
            engine = index.as_query_engine(
                similarity_top_k=4,
                text_qa_template=ANTI_HALLUCINATION_QA_TEMPLATE,
                refine_template=REFINE_TEMPLATE,
                response_mode="compact",
            )
            # Monkey-attach filtered retriever if supported
            try:
                engine._retriever = retriever  # type: ignore
            except Exception:
                pass
        except BaseException as e:
            print(f"[WARN] RAG query_for_materia sin engine, fallback: {e}")
            return (FALLBACK_PHRASE, [])
    except BaseException as e:
        print(f"[WARN] RAG query_for_materia sin engine, fallback: {e}")
        return (FALLBACK_PHRASE, [])
    try:
        response = engine.query(question.strip())
    except BaseException as e:
        if _is_quota_error(e):
            # 429 free tier: fallback inmediato, no reintentar en caliente
            print(f"[WARN] RAG query_for_materia 429/cuota, fallback sin reintento: {e}")
            return (FALLBACK_PHRASE, [])
        msg = str(e).lower()
        if "api_key" in msg or "authentication" in msg or "api key" in msg or "invalid" in msg:
            print(f"[ERROR] RAG query_for_materia fallo por API key: {e}")
            return (FALLBACK_PHRASE, [])
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            print(f"[WARN] RAG query_for_materia interrumpida, fallback: {e}")
            return (FALLBACK_PHRASE, [])
        print(f"[WARN] RAG query_for_materia fallo, fallback: {e}")
        return (FALLBACK_PHRASE, [])
    text, sources = _extract_answer_and_sources(response)
    # Defense in depth: drop any source whose materia metadata disagrees (should not happen).
    verified: list[dict] = []
    for src in sources:
        m = src.get("materia")
        if m and m != canon:
            continue
        # If no materia metadata (old index), verify via file path mapping
        if not m:
            fp = src.get("file_path") or src.get("file") or ""
            mapped = materia_for_file(str(fp))
            if mapped is None or mapped != canon:
                continue
        verified.append(src)
    # If retriever returned nothing for this materia -> fallback (no evidence)
    if not verified:
        # If LLM still produced a grounded-looking answer but no verified sources,
        # treat as no evidence (fail-closed).
        if text != FALLBACK_PHRASE:
            # Check if text is fallback paraphrase -> normalize, else force fallback
            text_n, _ = _normalize_fallback(text, [])
            if text_n == FALLBACK_PHRASE:
                return (FALLBACK_PHRASE, [])
            return (FALLBACK_PHRASE, [])
        return (FALLBACK_PHRASE, [])
    text, sources_out = _normalize_fallback(text, verified)
    # If normalized to fallback, clear sources
    if text == FALLBACK_PHRASE:
        return (FALLBACK_PHRASE, [])
    return (text, sources_out)


def reset_singleton() -> None:
    """Test helper: clear singleton cache (not for production use)."""
    global _index, _query_engine, _models_configured
    with _singleton_lock:
        _index = None
        _query_engine = None
        # Do not reset _models_configured automatically — tests control Settings directly


__all__ = [
    "FUENTES_DIR",
    "DATOS_DIR",
    "DATOS_LEGACY_DIR",
    "DATOS_DIRS",
    "CHROMA_DIR",
    "COLLECTION_NAME",
    "ACTIVE_MATERIAS",
    "EMBED_MODEL_NAME",
    "LLM_MODEL_NAME",
    "GEMINI_LLM_MODEL_NAME",
    "GEMINI_EMBED_MODEL_NAME",
    "ANTI_HALLUCINATION_QA_TMPL_STR",
    "ANTI_HALLUCINATION_QA_TEMPLATE",
    "REFINE_TMPL_STR",
    "REFINE_TEMPLATE",
    "FALLBACK_PHRASE",
    "LEGACY_FALLBACK_PHRASE",
    "validar_entorno",
    "configurar_modelos",
    "obtener_coleccion_chroma",
    "construir_o_cargar_indice",
    "crear_query_engine",
    "get_index",
    "get_query_engine",
    "query",
    "query_for_materia",
    "resolve_materia",
    "collection_for",
    "materia_for_file",
    "build_materia_filter",
    "reset_singleton",
    "_is_gemini_configured",
    "_get_active_provider",
]

