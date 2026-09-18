#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
local_app.py — Bot Facultad LOCAL (alternativa simple a WhatsApp)

Para uso SOLO tuyo, sin WhatsApp ni Drive ni ngrok.

  python local_app.py
  -> abre http://localhost:8000 en el navegador

Qué hace:
  - Chat web local bonito (selector de materia + caja de pregunta)
  - Usa rag_core directamente (mismo índice que main.py / chroma_db)
  - No necesita WHATSAPP_TOKEN, VERIFY_TOKEN, ni GOOGLE_SERVICE_ACCOUNT_JSON
  - Solo necesita GEMINI_API_KEY o OPENAI_API_KEY (ya lo tenés en .env)
  - Guarda historial local en historial_local.json (opcional)

Endpoints:
  GET  /              -> Chat UI (HTML)
  GET  /api/materias  -> lista de materias disponibles
  POST /api/chat      -> {materia, pregunta} -> {respuesta, fuentes, materia, is_fallback}
  GET  /api/health    -> estado del índice
  GET  /api/historial -> historial local (últimas preguntas)

Run:
  uvicorn local_app:app --host 127.0.0.1 --port 8000 --reload
  o
  python local_app.py
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

try:
    import materias_config
except ImportError as e:
    materias_config = None  # type: ignore
    logger.warning("[local] materias_config no encontrado: %s", e)

# rag_core es PESADO (llama_index + chroma = 8-11s de import). Lo cargamos LAZY
# para que el servidor arranque en <2s y vos veas la página rápido.
# El primer /api/chat pagará ese costo (3-4s), los siguientes son instantáneos.
rag_core = None  # type: ignore
_rag_import_error: str | None = None

def _get_rag_core():
    global rag_core, _rag_import_error
    if rag_core is not None:
        return rag_core
    if _rag_import_error is not None:
        return None
    try:
        import rag_core as _rc  # type: ignore
        rag_core = _rc
        logger.info("[local] rag_core lazy import OK")
        return rag_core
    except Exception as e:
        _rag_import_error = str(e)
        logger.warning("[local] rag_core lazy import fallo: %s", e)
        return None

# ---------------------------------------------------------------------------
# Historial local (archivo JSON simple, sin Drive)
# ---------------------------------------------------------------------------
HISTORIAL_PATH = Path("historial_local.json")
HISTORIAL_MAX = 200

def _append_historial(entry: dict) -> None:
    try:
        data: list[dict] = []
        if HISTORIAL_PATH.exists():
            try:
                raw = HISTORIAL_PATH.read_text(encoding="utf-8").strip()
                if raw:
                    loaded = json.loads(raw)
                    if isinstance(loaded, list):
                        data = loaded
            except Exception:
                data = []
        data.append(entry)
        # keep last N
        if len(data) > HISTORIAL_MAX:
            data = data[-HISTORIAL_MAX:]
        HISTORIAL_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning("[local] no se pudo guardar historial: %s", e)

def _load_historial(limit: int = 50) -> list[dict]:
    try:
        if not HISTORIAL_PATH.exists():
            return []
        raw = HISTORIAL_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        # last N reversed (más reciente primero si se quiere, pero devolvemos cronológico)
        return data[-limit:]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Lifespan — carga perezosa (no bloquea el arranque)
# ---------------------------------------------------------------------------
_query_engine: Any | None = None
_index_error: str | None = None
_engine_lock = __import__("threading").Lock()

def _ensure_engine():
    """Lazy-load del índice en el primer /api/chat. Thread-safe, no bloquea startup."""
    global _query_engine, _index_error
    if _query_engine is not None:
        return _query_engine
    rc = _get_rag_core()
    if rc is None:
        _index_error = _rag_import_error or "rag_core no disponible"
        return None
    with _engine_lock:
        if _query_engine is not None:
            return _query_engine
        try:
            logger.info("[local] lazy-load índice RAG (primer query, puede tardar 5-10s la primera vez)...")
            _query_engine = rc.get_query_engine()
            _index_error = None
            logger.info("[local] índice cargado OK (engine=%s)", type(_query_engine).__name__)
            return _query_engine
        except RuntimeError as e:
            _index_error = str(e)
            _query_engine = None
            logger.warning("[local] índice no disponible: %s", e)
            return None
        except SystemExit as e:
            _index_error = f"SystemExit: {e}"
            _query_engine = None
            logger.warning("[local] SystemExit al cargar índice: %s", e)
            return None
        except Exception as e:
            _index_error = str(e)
            _query_engine = None
            logger.warning("[local] error inesperado cargando índice: %s", e)
            return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # No bloquea: solo verifica que chroma_db existe, el índice se carga en el primer chat
    logger.info("[local] iniciando — listo en http://localhost:8000 (índice lazy)")
    chroma_dir = Path("chroma_db")
    if not chroma_dir.exists():
        logger.warning("[local] chroma_db no existe — primer query intentará cargar y puede tardar; si falla, ejecuta: python main.py --reindex")
    yield
    logger.info("[local] shutdown")

app = FastAPI(title="Bot Facultad — Local", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    materia: str
    pregunta: str

class ChatResponse(BaseModel):
    materia: str
    pregunta: str
    respuesta: str
    fuentes: list[dict]
    is_fallback: bool

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/materias")
async def api_materias():
    """Only the 3 active materias with sources (fuentes/ + legacy mapped, rest ignored)."""
    try:
        if materias_config is not None:
            whitelist = list(getattr(materias_config, "MATERIAS_WHITELIST", []))
        else:
            whitelist = ["Redes de Datos", "Gestión de Calidad", "Ingeniería de Software"]
        # Scan new fuentes/ + legacy dirs, but only keep whitelisted canonicals.
        candidates: list[str] = []
        search_dirs = [Path("fuentes"), Path("fuentes by bayzon"), Path("datos")]
        valid_exts = {".pdf", ".docx", ".doc"}
        for base in search_dirs:
            try:
                if not base.exists() or not base.is_dir():
                    continue
                for sub in base.iterdir():
                    if not sub.is_dir():
                        continue
                    try:
                        has_doc = any(
                            f.suffix.lower() in valid_exts
                            for f in sub.rglob("*") if f.is_file()
                        )
                    except Exception:
                        continue
                    if not has_doc:
                        continue
                    folder_name = sub.name.strip()
                    canonical = None
                    if materias_config is not None:
                        try:
                            # Map legacy folders: SGC->Gestión de Calidad, ISW->Ingeniería de Software
                            canonical = materias_config.validate_materia(folder_name)
                            if canonical is None:
                                try:
                                    import rag_core as _rc

                                    canonical = _rc.materia_for_file(str(sub))
                                except Exception:
                                    canonical = None
                        except Exception:
                            canonical = None
                    else:
                        canonical = folder_name
                    if canonical and canonical in whitelist and canonical not in candidates:
                        candidates.append(canonical)
                # Also check loose PDFs directly in base that map via path
                try:
                    import rag_core as _rc2

                    for f in base.glob("*"):
                        if f.is_file() and f.suffix.lower() in valid_exts:
                            try:
                                canon2 = _rc2.materia_for_file(str(f))
                            except Exception:
                                canon2 = None
                            if canon2 and canon2 in whitelist and canon2 not in candidates:
                                candidates.append(canon2)
                except Exception:
                    pass
            except Exception:
                continue
        if candidates:
            return JSONResponse({"materias": sorted(candidates)})
        # Fallback to whitelist (bot still starts even with zero PDFs; check_setup warns)
        return JSONResponse({"materias": whitelist})
    except Exception as e:
        logger.warning("[local] api_materias scan fallo: %s", e)
        if materias_config is not None:
            return JSONResponse({"materias": getattr(materias_config, "MATERIAS_WHITELIST", [])})
        return JSONResponse({"materias": []})

@app.get("/api/health")
async def api_health():
    # Si ya se cargó, OK inmediato. Si no, chequeo filesystem sin bloquear (no importa si rag_core aún no se importó)
    if _query_engine is not None:
        return JSONResponse({"status": "ok", "index": "loaded"})
    chroma_dir = Path("chroma_db")
    # Check filesystem directly — no necesita rag_core para esto
    if chroma_dir.exists():
        try:
            has_files = any(chroma_dir.iterdir())
            if not has_files:
                return JSONResponse({"status": "unavailable", "reason": "chroma_db vacío", "hint": "python main.py --reindex"}, status_code=503)
        except Exception:
            pass
    if _index_error:
        hint = "ejecutar python main.py --reindex tras preparar datos/" if "reindex" not in _index_error.lower() else _index_error
        return JSONResponse({"status": "unavailable", "reason": _index_error, "hint": hint}, status_code=503)
    if not chroma_dir.exists():
        return JSONResponse({"status": "unavailable", "reason": "chroma_db no encontrado", "hint": "python main.py --reindex tras preparar datos/"}, status_code=503)
    # Si rag_core aún no se importó pero chroma existe, reportamos como present (carga en primer chat)
    return JSONResponse({"status": "ok", "index": "present (lazy — se cargará en primer pregunta)"})

@app.get("/api/historial")
async def api_historial(limit: int = 50):
    limit = max(1, min(limit, 200))
    data = _load_historial(limit)
    return JSONResponse({"historial": data})

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    materia_raw = (req.materia or "").strip()
    pregunta = (req.pregunta or "").strip()

    if not materia_raw:
        raise HTTPException(status_code=400, detail="Materia requerida")
    if not pregunta:
        raise HTTPException(status_code=400, detail="Pregunta requerida")

    if materias_config is None:
        raise HTTPException(status_code=500, detail="materias_config no disponible")

    rc = _get_rag_core()
    if rc is None:
        raise HTTPException(status_code=503, detail=f"RAG no disponible — {_rag_import_error or 'verifica GEMINI_API_KEY/OPENAI_API_KEY y chroma_db'}")

    canonical = materias_config.validate_materia(materia_raw)
    if canonical is None:
        # ayuda sin RAG
        try:
            help_msg = materias_config.help_text()
        except Exception:
            help_msg = "Materia no reconocida. Materias disponibles: " + ", ".join(getattr(materias_config, "MATERIAS_WHITELIST", []))
        raise HTTPException(status_code=400, detail=help_msg)

    # Asegurar índice cargado (lazy, primera vez puede tardar 5-10s)
    _ensure_engine()

    # RAG query (per-materia filtrado)
    try:
        if hasattr(rc, "query_for_materia"):
            answer, fuentes = rc.query_for_materia(canonical, pregunta)  # type: ignore
        else:
            answer, fuentes = rc.query(pregunta)  # type: ignore
        answer = str(answer).strip() if answer else ""
        if not isinstance(fuentes, list):
            fuentes = []
    except RuntimeError as e:
        logger.warning("[local] RAG 503: %s", e)
        fallback = getattr(rc, "FALLBACK_PHRASE", "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad.")
        answer = fallback
        fuentes = []
    except Exception as e:
        logger.error("[local] RAG error: %s", e)
        fallback = getattr(rc, "FALLBACK_PHRASE", "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad.")
        answer = fallback
        fuentes = []

    if not answer:
        answer = getattr(rc, "FALLBACK_PHRASE", "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad.")

    fallback_phrase = getattr(rc, "FALLBACK_PHRASE", "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad.")
    is_fallback = bool(answer == fallback_phrase or fallback_phrase in answer or answer.startswith("No se encontró") or answer.startswith("No encontré"))

    # Guardar historial local (siempre, incluso fallback — útil para repasar)
    try:
        _append_historial({
            "fecha": datetime.now(timezone.utc).isoformat(),
            "materia": canonical,
            "pregunta": pregunta,
            "respuesta": answer,
            "fuentes": fuentes,
            "is_fallback": is_fallback,
        })
    except Exception:
        pass

    return ChatResponse(
        materia=canonical,
        pregunta=pregunta,
        respuesta=answer,
        fuentes=fuentes,
        is_fallback=is_fallback,
    )

# ---------------------------------------------------------------------------
# UI — HTML inline (sin archivos extras, no necesita static/)
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot Facultad — Local</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--card2:#232730;--text:#e8eaf0;--muted:#9aa0b2;--accent:#6ea8fe;--accent2:#8b5cf6;--ok:#22c55e;--warn:#f59e0b;--border:#2a2f3a;--radius:14px}
  *{box-sizing:border-box}
  body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;background:radial-gradient(1200px 600px at 20% -10%, #1e293b 0%, transparent 60%),radial-gradient(900px 500px at 95% 0%, #1e1b4b 0%, transparent 55%), var(--bg);color:var(--text);min-height:100vh}
  header{max-width:900px;margin:28px auto 0;padding:0 18px;display:flex;align-items:center;gap:14px}
  .logo{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;font-weight:800;color:white;box-shadow:0 8px 24px rgba(110,168,254,.35)}
  h1{margin:0;font-size:22px;letter-spacing:.2px}
  .sub{color:var(--muted);font-size:13px;margin-top:2px}
  .badge{margin-left:auto;background:rgba(34,197,94,.14);color:#86efac;border:1px solid rgba(34,197,94,.3);padding:6px 10px;border-radius:999px;font-size:12px}
  .badge.off{background:rgba(245,158,11,.14);color:#fcd34d;border-color:rgba(245,158,11,.3)}
  .wrap{max-width:900px;margin:18px auto;padding:0 18px 40px;display:grid;gap:16px}
  .card{background:rgba(26,29,36,.9);backdrop-filter:blur(10px);border:1px solid var(--border);border-radius:var(--radius);padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
  .row{display:flex;gap:12px;flex-wrap:wrap;align-items:end}
  .field{flex:1;min-width:160px;display:flex;flex-direction:column;gap:6px}
  label{font-size:12px;color:var(--muted);letter-spacing:.3px;text-transform:uppercase}
  select, textarea{width:100%;background:var(--card2);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:11px 12px;font-size:14px;outline:none}
  select:focus, textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(110,168,254,.2)}
  textarea{resize:vertical;min-height:68px}
  .hint{font-size:12px;color:var(--muted)}
  .btn{background:linear-gradient(135deg,var(--accent),var(--accent2));color:white;border:none;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer;box-shadow:0 6px 18px rgba(99,102,241,.35);transition:.15s}
  .btn:hover{transform:translateY(-1px);box-shadow:0 10px 22px rgba(99,102,241,.45)}
  .btn:disabled{opacity:.55;cursor:not-allowed;transform:none}
  .btn-ghost{background:transparent;border:1px solid var(--border);color:var(--text);box-shadow:none}
  .chat{display:flex;flex-direction:column;gap:12px;max-height:62vh;overflow:auto;padding-right:4px}
  .msg{background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:12px 14px;line-height:1.5}
  .msg.user{background:linear-gradient(135deg,rgba(110,168,254,.18),rgba(139,92,246,.14));border-color:rgba(110,168,254,.35)}
  .msg.assistant{border-left:3px solid var(--accent)}
  .msg.fallback{border-left:3px solid var(--warn);background:rgba(245,158,11,.08)}
  .meta{font-size:12px;color:var(--muted);margin-bottom:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .tag{padding:2px 8px;border-radius:999px;background:var(--card);border:1px solid var(--border);font-size:11px}
  .tag.ok{background:rgba(34,197,94,.14);border-color:rgba(34,197,94,.3);color:#86efac}
  .tag.warn{background:rgba(245,158,11,.14);border-color:rgba(245,158,11,.3);color:#fcd34d}
  .fuentes{margin-top:8px;padding-top:8px;border-top:1px dashed var(--border);font-size:12px;color:var(--muted)}
  .fuentes b{color:var(--text)}
  .empty{color:var(--muted);text-align:center;padding:18px;border:1px dashed var(--border);border-radius:12px;background:rgba(255,255,255,.02)}
  .footer{font-size:12px;color:var(--muted);text-align:center;padding:8px}
  a{color:var(--accent)}
  kbd{background:var(--card2);border:1px solid var(--border);border-bottom-width:2px;padding:2px 6px;border-radius:6px;font-size:11px}
  @media(max-width:640px){header{margin-top:16px}.row{flex-direction:column}.field{min-width:100%}}
</style>
</head>
<body>
<header>
  <div class="logo">UTN</div>
  <div>
    <h1>Bot Facultad — Local</h1>
    <div class="sub">Tu asistente académico. Sin WhatsApp, sin Drive, solo vos. Escribí <b>MATERIA + pregunta</b> y responde con fuentes.</div>
  </div>
  <div id="healthBadge" class="badge off">verificando…</div>
</header>

<div class="wrap">
  <div class="card">
    <div class="row">
      <div class="field" style="max-width:260px">
        <label for="materia">Materia</label>
        <select id="materia"></select>
        <div class="hint">Solo materias con PDFs en <code>datos/</code></div>
      </div>
      <div class="field">
        <label for="pregunta">Pregunta</label>
        <textarea id="pregunta" placeholder="Ej: ¿Qué dice el teorema de Rolle sobre continuidad y derivabilidad?"></textarea>
        <div class="hint"><kbd>Enter</kbd> envía · <kbd>Shift</kbd> + <kbd>Enter</kbd> salto de línea</div>
      </div>
    </div>
    <div class="row" style="margin-top:12px;justify-content:space-between">
      <div style="display:flex;gap:8px">
        <button id="sendBtn" class="btn">Preguntar →</button>
        <button id="clearBtn" class="btn btn-ghost">Limpiar chat</button>
      </div>
      <div class="hint" id="hintLine">Listo para preguntar.</div>
    </div>
  </div>

  <div id="chat" class="chat">
    <div class="empty">👋 ¡Hola! Elegí una materia y hacé tu primera pregunta.<br><span style="font-size:12px">Ejemplo: <b>AM2: ¿Qué condiciones pide el teorema de Rolle?</b></span></div>
  </div>

  <div class="card" id="historialCard" style="display:none">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <b style="font-size:13px">Historial local</b>
      <span class="hint" id="histCount"></span>
      <button id="reloadHist" class="btn btn-ghost" style="margin-left:auto;padding:6px 10px;font-size:12px">Recargar</button>
    </div>
    <div id="historial" style="display:flex;flex-direction:column;gap:8px;max-height:260px;overflow:auto"></div>
  </div>

  <div class="footer">Local • RAG <code>chroma_db/</code> + <code>datos/</code> • Historial en <code>historial_local.json</code> • <a href="/api/health" target="_blank">/api/health</a> • <a href="/docs" target="_blank">/docs</a></div>
</div>

<script>
const $materia = document.getElementById('materia');
const $pregunta = document.getElementById('pregunta');
const $sendBtn = document.getElementById('sendBtn');
const $clearBtn = document.getElementById('clearBtn');
const $chat = document.getElementById('chat');
const $hint = document.getElementById('hintLine');
const $badge = document.getElementById('healthBadge');
const $histCard = document.getElementById('historialCard');
const $hist = document.getElementById('historial');
const $histCount = document.getElementById('histCount');

let materias = [];

async function loadMaterias(){
  try{
    const r = await fetch('/api/materias');
    const j = await r.json();
    materias = j.materias || [];
  }catch(e){ materias = ["AM2","AM1","Algebra","Fisica","Analisis Numerico","Investigacion Operativa","Economia","Gestion de Calidad"]; }
  $materia.innerHTML = materias.map(m=>`<option value="${m}">${m}</option>`).join('');
}

async function checkHealth(){
  try{
    const r = await fetch('/api/health');
    const j = await r.json();
    if(r.ok){ $badge.textContent='índice OK ✓'; $badge.className='badge'; }
    else { $badge.textContent='índice no listo (reindex)'; $badge.className='badge off'; $hint.textContent = j.hint || j.reason || 'Índice no disponible'; }
  }catch(e){ $badge.textContent='sin conexión'; $badge.className='badge off'; }
}

function addMessage(role, html, opts={}){
  // remove empty placeholder
  const empty = $chat.querySelector('.empty');
  if(empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + (role==='user' ? 'user' : (opts.fallback ? 'fallback' : 'assistant'));
  let meta = '';
  if(role==='user'){
    meta = `<div class="meta"><span class="tag">${opts.materia||''}</span> <span>${new Date().toLocaleString()}</span></div>`;
  } else {
    const tag = opts.fallback ? '<span class="tag warn">sin evidencia</span>' : '<span class="tag ok">con fuentes</span>';
    meta = `<div class="meta">${tag} <span class="tag">${opts.materia||''}</span> <span>${new Date().toLocaleString()}</span></div>`;
  }
  let fuentesHtml = '';
  if(opts.fuentes && opts.fuentes.length){
    fuentesHtml = `<div class="fuentes"><b>Fuentes:</b> ${opts.fuentes.map(f=>`${esc(f.file)} p. ${esc(f.page)}`).join(', ')}</div>`;
  } else if(role!=='user' && opts.fallback){
    fuentesHtml = `<div class="fuentes">Fuentes: N/A · Pregunta guardada igual en historial.</div>`;
  }
  div.innerHTML = meta + `<div>${html}</div>` + fuentesHtml;
  $chat.appendChild(div);
  $chat.scrollTop = $chat.scrollHeight;
}

function esc(s){ return String(s||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;'); }
function fmtText(t){ return esc(t).replaceAll('\n','<br>'); }

async function send(){
  const materia = $materia.value.trim();
  const pregunta = $pregunta.value.trim();
  if(!materia){ $hint.textContent='Elegí una materia.'; return; }
  if(!pregunta){ $hint.textContent='Escribí una pregunta.'; $pregunta.focus(); return; }
  $sendBtn.disabled=true; $hint.textContent='Consultando RAG…';
  addMessage('user', fmtText(pregunta), {materia});
  $pregunta.value='';
  try{
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({materia, pregunta})});
    const j = await r.json();
    if(!r.ok){
      addMessage('assistant', fmtText(j.detail || 'Error'), {materia, fallback:true});
      $hint.textContent = j.detail || 'Error';
    } else {
      addMessage('assistant', fmtText(j.respuesta), {materia: j.materia, fuentes: j.fuentes, fallback: j.is_fallback});
      $hint.textContent = j.is_fallback ? 'Sin evidencia en bibliografía — revisá tu pregunta o agregá PDFs.' : `Respondido con ${j.fuentes.length} fuente(s).`;
      loadHistorial();
    }
  }catch(e){
    addMessage('assistant', 'Error de red: '+esc(e.message||e), {materia, fallback:true});
    $hint.textContent='Error de red';
  }finally{ $sendBtn.disabled=false; }
}

async function loadHistorial(){
  try{
    const r = await fetch('/api/historial?limit=12');
    const j = await r.json();
    const arr = j.historial || [];
    if(!arr.length){ $histCard.style.display='none'; return; }
    $histCard.style.display='block';
    $histCount.textContent = `(${arr.length} últimas)`;
    $hist.innerHTML = arr.slice().reverse().map(h=>`
      <div style="background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:8px 10px">
        <div style="font-size:12px;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap"><span class="tag">${esc(h.materia)}</span><span>${esc(h.pregunta).slice(0,80)}</span><span style="margin-left:auto">${new Date(h.fecha).toLocaleString()}</span></div>
        <div style="font-size:13px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(h.respuesta).slice(0,120)}</div>
      </div>
    `).join('');
  }catch(e){ /* ignore */ }
}

$sendBtn.addEventListener('click', send);
$clearBtn.addEventListener('click', ()=>{ $chat.innerHTML='<div class="empty">Chat limpio. ¡Seguí preguntando! 👋</div>'; $hint.textContent='Chat limpiado.'; });
$pregunta.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
document.getElementById('reloadHist').addEventListener('click', loadHistorial);

loadMaterias(); checkHealth(); loadHistorial();
setInterval(checkHealth, 15000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def ui_root():
    return HTMLResponse(content=HTML_PAGE)

@app.get("/health", response_class=JSONResponse)
async def health_alias():
    # alias for webhook compatibility
    return await api_health()

if __name__ == "__main__":
    import uvicorn, socket
    # Detectar IP local para mostrar en consola
    try:
        s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        local_ip=s.getsockname()[0]
        s.close()
    except: local_ip="192.168.1.7"
    print("\n" + "="*72)
    print(" Bot Facultad — LOCAL")
    print(f" PC:    http://localhost:8000")
    print(f" Celu (mismo WiFi): http://{local_ip}:8000")
    print(f" Docs:  http://localhost:8000/docs")
    print(" Para salir: Ctrl+C")
    print("="*72 + "\n")
    uvicorn.run("local_app:app", host="0.0.0.0", port=8000, reload=False)
