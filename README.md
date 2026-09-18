# Bot Facultad — WhatsApp + RAG + Drive

Asistente académico UTN vía WhatsApp (solo interfaz: nunca envía contenido académico por WhatsApp) que consulta exclusivamente la materia pedida y acumula cada Q&A en Google Docs por materia.

Activas (solo 3): `Redes de Datos`, `Gestión de Calidad`, `Ingeniería de Software`.
Fuentes: `fuentes/Redes de Datos/`, `fuentes/Gestion de Calidad/` (sin tilde en carpeta, canon con tilde), `fuentes/Ingenieria de Software/` + legacy `fuentes by bayzon/` y `datos/` solo para las 3 (SGC→Gestión de Calidad, ISW→Ingeniería de Software, resto ignorado).

`MATERIA: pregunta` → FastAPI `/webhook` (ACK <200 ms, `BackgroundTasks`) → `materias_config` (alias + normalización) → `rag_core` exclusivo por materia (`where={"materia":canon}` antes del LLM) → `drive_docs_service` (`{MATERIA} - Preguntas`, privado) → `whatsapp_client` (solo `✅ Pregunta guardada y respondida`).

> `wa_qr_bot/` queda fuera del flujo principal (QR personal, responde RAG completo). No se usa en producción Cloud API. Se conserva solo como alternativa local, no se modifica.

## Instalación

```bash
python -m venv env
.\env\Scripts\activate        # Windows
# source env/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

Requisitos: Python 3.12+, `llama-index==0.14.24` + `chromadb==1.5.9` + `openai==2.54.0` + `fastapi`/`uvicorn`/`httpx`/`google-api-python-client`/`google-auth`/`python-dotenv`.

## Variables de entorno (`.env`)

Crear `.env` en la raíz (ver `.env.example`):

| Variable | Descripción | Ejemplo |
|---|---|---|
| `OPENAI_API_KEY` | Key de OpenAI para embeddings/LLM | `sk-proj-...` |
| `VERIFY_TOKEN` | Token para verificación `GET /webhook` (Meta Dashboard) | `mi_token_secreto` |
| `WHATSAPP_TOKEN` | Token de WhatsApp Cloud API (Bearer) | `EAA...` |
| `WHATSAPP_PHONE_NUMBER_ID` | Phone Number ID del webhook (URL `.../v20.0/{id}/messages`) | `123456789012345` |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Ruta al JSON de Service Account | `./bot-facultad-sa.json` |
| `GOOGLE_DRIVE_FOLDER_ID` | ID de carpeta Drive `Bot_Facultad_Materias` | `1AbC...` |

`.env` está git-ignorado. Nunca commitear el JSON de Service Account (`*-sa.json`, `bot-facultad-sa.json`).

## Google Drive — carpeta y permisos

1. Crear carpeta Drive `Bot_Facultad_Materias`.
2. Crear Service Account en Google Cloud → generar JSON → guardar como `bot-facultad-sa.json` (ruta en `GOOGLE_SERVICE_ACCOUNT_JSON`).
3. Compartir la carpeta con el email de la Service Account (`...@...iam.gserviceaccount.com`) con rol **Editor**.
4. Copiar el ID de la carpeta (URL `.../folders/{ID}`) a `GOOGLE_DRIVE_FOLDER_ID`.
5. El sistema crea automáticamente por materia el Doc `"{MATERIA} - Bot Facultad"` y cachea el `docId` en `materias_docs.json` (git-ignorado, seed `{}`). Si el cache se pierde, busca y recrea.

## Índice RAG

```bash
# Primera vez o tras cambiar PDFs en datos/
python main.py --reindex
# Uso normal (reutiliza chroma_db/)
python main.py
# Loop interactivo: preguntar en terminal con citas
python main.py --help   # muestra --reindex
```

`main.py` es un wrapper delgado (~70 líneas) que delega todo el ciclo RAG a `rag_core` singleton (`validar_entorno`, `configurar_modelos`, `get_query_engine`, `query`, `ANTI_HALLUCINATION_QA_TMPL_STR`). No hace `sys.exit` al importar. `rag_core` expone `FALLBACK_PHRASE = "No encontré información suficiente en los documentos para responder esa pregunta."` verbatim (contrato anti-alucinación).

PDFs recursivos en `datos/` (solo `.pdf`). Índice persistente en `chroma_db/` (prod) y `chroma_db_test/` (tests, 80 vectores con `MockEmbedding`).

### Cuota Gemini free tier (1000 embeddings/día)

Un reindex grande NO entra en 1 día free tier (ej. 1331 docs → ~1599 pedidos de embedding → `429 quota exceeded`, 5 reintentos, `sys.exit(1)` en CLI).

- Antes de reindexar, mirá el log `[OK] Cargados N documentos`. Si `N > 800`, el reindex probablemente falle por cuota.
- Opciones: recortar PDFs a <800 chunks (sacar libros completos, dejar apuntes), o reindexar en 2 días / con otra API key. No hay batch multi-día automático (a propósito, para no quemar cuota).
- El webhook NUNCA reindexa ni hace `sys.exit`: si el índice falta o hay `429`, devuelve fail-fast el fallback exacto `No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad.` con `fuentes=[]`, igual guarda en Drive y responde `✅`. Solo `python main.py --reindex` manual puede salir con error.

## Webhook (solo interfaz)

```bash
start_bot.bat
# o: uvicorn webhook_server:app --host 0.0.0.0 --port 8000
# salud
curl http://localhost:8000/health        # {"status":"ok"} si índice listo
# túnel para Meta:
start_tunnel.bat
# diagnóstico / prueba:
python check_setup.py
python test_question.py --dry-run "Redes de Datos: ¿Qué es una VLAN?"
```

WhatsApp nunca recibe contenido académico. Mensajes byte-exactos:
- Éxito: `✅ Pregunta guardada y respondida`
- Sin `:`: `❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?`
- Desconocida: `❌ No encontré la materia "{tal cual}".`
- Fallo Docs: `❌ No pude guardar la pregunta. Intentá nuevamente.`

Endpoints:

- `GET /webhook?hub.mode=subscribe&hub.verify_token=&hub.challenge=` → `200` challenge | `403` si token inválido.
- `POST /webhook` → `200 {status:"ok", enqueued:n}` en <200 ms (`BackgroundTasks.add_task(process_question)`), LRU(1000) dedup por `message_id`, skip silencioso si payload sin `entry.changes.value.messages`.
- `GET /health` → `200` si índice cargado | `503` con hint `python main.py --reindex`.

`process_question` flow: `parse_message` → `validate_materia` (unknown → help text with whitelist, no RAG/Drive) → `rag_core.query_for_materia` (fail-closed per materia; fallback phrase preserved) → `drive_docs_service.append_entry` (template `Fecha/Pregunta/Respuesta/Fuentes/---`, `- Sin fuentes registradas` on fallback, per-materia `Lock`, `message_id` dedup) → `whatsapp_client.send_text` (interface-only `✅ Pregunta guardada y respondida` always; never academic content, never a `Fuentes:` footer; sources live in Drive only).

## Deploy

### Local con ngrok (manual E2E)

```bash
uvicorn webhook_server:app --port 8000
ngrok http 8000
# Copiar URL https://xxx.ngrok-free.app → Meta Dashboard > WhatsApp > Configuration > Webhook
# URL: https://xxx.ngrok-free.app/webhook, Verify Token: $VERIFY_TOKEN → Verify and Save
# Suscribir a campo `messages`
# Enviar WhatsApp al número de prueba:
#   AM2: ¿Qué dice el teorema de Rolle?  → debe append Doc AM2 con "---" y responder con fuentes
#   Fisica2: x                           → debe responder "Materias disponibles: ..."
#   Reenviar mismo wamid                 → debe escribir solo una vez (LRU dedup)
# Borrar/renombrar chroma_db → /health debe dar 503
```

### Render (producción)

- Build: `pip install -r requirements.txt`
- Start: `uvicorn webhook_server:app --host 0.0.0.0 --port $PORT`
- Env vars: setear todas las de la tabla arriba en Dashboard → Environment.
- Disk: `materias_docs.json` es efímero; el cache se reconstruye vía Drive search si se pierde (próxima escritura crea si falta). Para persistencia extra, usar volume o buscar doc por título en Drive antes de crear.
- Health check: `/health` para uptime.

## Rollback (<1 min)

Deshabilitar webhook en Meta Dashboard → Configuration → Webhook → `Edit` → quitar URL o `Unsubscribe`. El tráfico deja de llegar instantáneamente. Revert de código:

```bash
git checkout -- main.py
rm -rf tests/
git checkout -- README.md
# o revert del PR completo
```

No hay migración de datos; los Docs en Drive y `chroma_db/` permanecen. Rollback no borra Docs.

## Tests

```bash
pip install pytest
pytest tests/ -q                # suite formal (70 tests) — debe quedar en verde
pytest tests/test_materias_config.py tests/test_rag_core.py -v
pytest tests/test_drive_service.py tests/test_whatsapp_client.py -v
pytest tests/test_webhook.py -v # incluye harness E2E automatizado vía TestClient
```

Harness automatizado (sin ngrok) en `tests/test_webhook.py::test_e2e_harness_simulates_full_flow` simula vía `TestClient`:

- `AM2: Rolle?` → `drive.append_entry` con `---` y `whatsapp.send_text` con `Fuentes:`
- `Fisica2: x` → `help` sin RAG/Drive
- `dup` message_id → `LRU` escribe una sola vez
- `chroma_db` faltante → `/health` 503

Coverage por capa:

| Capa | Tests |
|---|---|
| Unit `materias_config` | regex, trim, unknown, alias, help_text whitelist |
| Unit `rag_core` | singleton reuse, missing chroma 503, query `MockEmbedding` 384 + `MockLLM` vs `chroma_db_test` 80 vectores, fallback phrase |
| Unit `drive_docs_service` | template `Fecha/Pregunta/Respuesta/Fuentes/---`, `N/A` fallback, cache hit/miss `googleapiclient` mock, per-materia `Lock` distinct, `message_id` dedup `OrderedDict`, LRU eviction 2000 |
| Unit `whatsapp_client` | `httpx.Client` mock success/failure/timeout/400, `Authorization: Bearer`, URL `.../{phone_id}/messages`, limpieza `to` |
| Integration `webhook_server` | `GET` valid/invalid `200/403`, `POST` ACK <200 ms + `BackgroundTasks`, payload inválido `200-skip`, dedup `LRU` 1000→1001, unknown `no RAG/Drive`, fallback `N/A` con Drive, `/health` `200/503` |

Verificar antes de PR:

```bash
python -m py_compile main.py rag_core.py materias_config.py drive_docs_service.py whatsapp_client.py webhook_server.py
python main.py --help
python -c "from webhook_server import app; print(app.title, app.routes)"
pytest tests/ -q
```

## Estructura

```
Bot_Facultad/
├── main.py                 # wrapper CLI (~70 líneas) → rag_core
├── rag_core.py             # singleton CitationQueryEngine + fallback
├── materias_config.py      # MATERIA_REGEX, whitelist 7 + aliases, help_text
├── whatsapp_client.py      # send_text via httpx + Graph API
├── drive_docs_service.py   # SA auth, cache materias_docs.json, append batchUpdate, Lock, dedup
├── webhook_server.py       # FastAPI lifespan, GET/POST /webhook, LRU, /health
├── datos/                  # PDFs recursivos
├── chroma_db/              # índice prod (persistido)
├── chroma_db_test/         # índice test (80 vectores, MockEmbedding)
├── materias_docs.json      # cache {materia: docId} (git-ignorado, seed {})
├── requirements.txt        # pinned llama-index/chroma/openai + fastapi/uvicorn/httpx/google
├── .env / .env.example
├── tests/
│   ├── test_materias_config.py
│   ├── test_rag_core.py
│   ├── test_drive_service.py
│   ├── test_whatsapp_client.py
│   └── test_webhook.py     # + harness E2E
└── test_rag_mock.py        # legacy harness mock (no pytest, pre-PR1)
```

## Decisiones de arquitectura

- `BackgroundTasks` vs `Celery/Redis`: ACK <200 ms sin infra extra; dedup LRU cubre reintentos Meta (ver `design`).
- Extracción `rag_core.py`: single source para `main.py` y `webhook_server` lifespan; preserva `ANTI_HALLUCINATION_QA_TMPL_STR` verbatim.
- Service Account `drive+documents` + `materias_docs.json` cache: lookup O(1), auto-create `"MATERIA - Bot Facultad"` + share Editor best-effort.
- Per-materia `Lock` + LRU 1000 (webhook) / 2000 (drive): appends ordenados sin serializar global, idempotencia en reintentos.

## Licencia

MIT — ver repo. No commitear secretos; `materias_docs.json` y `*-sa.json` están en `.gitignore`.
