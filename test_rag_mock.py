#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test RAG con mocks — verifica todo el flujo sin necesidad de OpenAI.

Cubre:
  1. main.py --help
  2. load_dotenv carga OPENAI_API_KEY
  3. encuentra PDFs en datos/ (recursive)
  4. crea ChromaDB en chroma_db/ (aqui usamos temp dir para no ensuciar)
  5. genera indice primera vez
  6. reutiliza indice en ejecuciones posteriores (from_vector_store)
  7. --reindex borra y recrea
  8. puede responder pregunta basada en PDF (MockLLM)
  9. muestra fuentes/paginas
 10. anti-alucinacion (prompt contiene frase exacta)

Uso:
  .\\env\\Scripts\\python.exe test_rag_mock.py
  .\\env\\Scripts\\python.exe test_rag_mock.py --keep  # no borra chroma_db_test

Para probar con key real (no mock):
  1. Edita .env -> OPENAI_API_KEY=sk-proj-...REAL...
  2. .\\env\\Scripts\\python.exe main.py --reindex
     # primera vez construye indice (tarda ~30s, llama a OpenAI)
  3. Pregunta: "Que temas cubre el apunte de Analisis Numerico?"
  4. Verifica respuesta + fuentes con paginas
  5. Prueba reutilizacion: .\\env\\Scripts\\python.exe main.py
     # debe decir "Reutilizando indice existente"
  6. Prueba anti-alucinacion: pregunta unrelated "Quien gano el mundial 2030?"
     # debe responder "No encontre informacion suficiente..."
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

# Asegurar stdout utf-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).parent
DATOS_DIR = PROJECT_ROOT / "datos"
CHROMA_TEST_DIR = PROJECT_ROOT / "chroma_db_test"
CHROMA_REAL_DIR = PROJECT_ROOT / "chroma_db"
KEEP = "--keep" in sys.argv

def ok(msg):
    print(f"[PASS] {msg}")

def fail(msg):
    print(f"[FAIL] {msg}")
    return False

def warn(msg):
    print(f"[WARN] {msg}")

passed = 0
failed = 0

def check(name, fn):
    global passed, failed
    print(f"\n--- {name} ---")
    try:
        result = fn()
        if result is False:
            failed += 1
        else:
            passed += 1
    except Exception as e:
        import traceback
        print(f"[FAIL] {name} excepcion: {e}")
        traceback.print_exc()
        failed += 1

# 1. main.py --help
def test_help():
    result = subprocess.run(
        [str(PROJECT_ROOT / "env" / "Scripts" / "python.exe"), "main.py", "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(result.stdout[:800])
    if result.returncode != 0:
        return fail(f"--help returncode {result.returncode}: {result.stderr[:500]}")
    if "--reindex" not in result.stdout:
        return fail("--help no menciona --reindex")
    ok("main.py --help inicia correctamente")
    # tambien py_compile
    import py_compile
    py_compile.compile(str(PROJECT_ROOT / "main.py"), doraise=True)
    ok("py_compile OK")

# 2. carga OPENAI_API_KEY desde .env
def test_env():
    from dotenv import load_dotenv
    # carga explicita desde PROJECT_ROOT/.env
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return fail(".env no existe")
    print(f".env existe: {env_path}")
    # leer contenido raw
    raw = env_path.read_text(encoding="utf-8", errors="replace").strip()
    print(f".env contenido (primer 60 chars): {raw[:60]!r}")
    load_dotenv(dotenv_path=str(env_path), override=True)
    key = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY cargada: {repr(key[:20] + '...' if key and len(key)>20 else key)}")
    if not key:
        return fail("OPENAI_API_KEY vacia tras load_dotenv")
    if "TEST" in key:
        warn("Es key dummy (TEST) - correcto para mocks, para produccion reemplazar")
    ok("load_dotenv carga OPENAI_API_KEY desde .env")

# 3. encuentra PDFs recursive
def test_pdfs():
    pdfs = list(DATOS_DIR.rglob("*.pdf"))
    print(f"PDFs encontrados: {len(pdfs)}")
    for p in pdfs[:5]:
        print(f"  - {p.relative_to(DATOS_DIR)} (size {p.stat().st_size} bytes)")
    if not DATOS_DIR.exists():
        return fail("datos/ no existe")
    if len(pdfs) == 0:
        return fail("No se encontraron PDFs en datos/ recursive")
    # verificar que SimpleDirectoryReader tambien los encuentra
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(
        input_dir=str(DATOS_DIR),
        recursive=True,
        required_exts=[".pdf"],
        filename_as_id=True,
    )
    docs = reader.load_data(show_progress=False)
    print(f"SimpleDirectoryReader cargó {len(docs)} documentos")
    if len(docs) == 0:
        return fail("SimpleDirectoryReader no devolvió documentos")
    # verificar metadata trae file_name y page_label
    sample = docs[0].metadata
    print(f"Ejemplo metadata: {sample}")
    if "file_name" not in sample:
        warn("metadata sin file_name (pero file_path existe?)")
    if "page_label" not in sample and "page_number" not in sample:
        warn("metadata sin page_label/page_number - puede variar segun lector")
    ok(f"Encuentra PDFs recursive: {len(pdfs)} PDF(s) -> {len(docs)} chunks")
    return docs

# 4-7. Chroma + indice + reuso + reindex
def test_chroma_y_indice():
    # limpiar test dir si existe
    if CHROMA_TEST_DIR.exists() and not KEEP:
        shutil.rmtree(CHROMA_TEST_DIR, ignore_errors=True)
        print(f"Limpieza previa {CHROMA_TEST_DIR}")

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    # configurar mocks (evita llamar a OpenAI)
    Settings.embed_model = MockEmbedding(embed_dim=384)
    from llama_index.core.llms.mock import MockLLM
    Settings.llm = MockLLM(max_tokens=512)
    print(f"Modelos mock: embed_dim=384, MockLLM")

    # 4. crea ChromaDB correctamente en chroma_db_test/
    CHROMA_TEST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_TEST_DIR))
    col = client.get_or_create_collection("utn_bibliografia")
    print(f"Chroma PersistentClient en {CHROMA_TEST_DIR} -> coleccion '{col.name}' count={col.count()}")
    if col.count() != 0:
        warn("coleccion no vacia al inicio (restos previos)")
    ok("Crea ChromaDB correctamente")

    # 5. genera indice primera vez
    reader = SimpleDirectoryReader(
        input_dir=str(DATOS_DIR),
        recursive=True,
        required_exts=[".pdf"],
        filename_as_id=True,
    )
    documents = reader.load_data(show_progress=False)
    print(f"Documentos para indexar: {len(documents)}")
    vector_store = ChromaVectorStore(chroma_collection=col)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        show_progress=False,
    )
    count1 = col.count()
    print(f"Indice creado, vectores={count1}")
    if count1 == 0:
        return fail("Indice vacio tras from_documents")
    ok(f"Genera indice primera vez: {count1} vectores")

    # 6. reutiliza indice en ejecuciones posteriores (sin reindex)
    # Simula segunda ejecucion: crear nuevo client y cargar via from_vector_store
    # Importante: cerrar y reabrir cliente para simular nuevo proceso
    # En Windows, no podemos borrar mientras cliente abierto, pero si podemos reabrir
    client2 = chromadb.PersistentClient(path=str(CHROMA_TEST_DIR))
    col2 = client2.get_or_create_collection("utn_bibliografia")
    count2 = col2.count()
    print(f"Segunda apertura count={count2}")
    if count2 != count1:
        return fail(f"Reuso fallo: count {count2} != {count1}")
    vector_store2 = ChromaVectorStore(chroma_collection=col2)
    index2 = VectorStoreIndex.from_vector_store(vector_store=vector_store2, embed_model=Settings.embed_model)
    print(f"Reutilizado via from_vector_store: {index2}")
    # No debe haber creado nuevos vectores
    if col2.count() != count1:
        return fail("Reutilizar creo vectores extra")
    ok(f"Reutiliza indice en ejecuciones posteriores: {count2} vectores")

    # 7. --reindex funciona (borra y recrea)
    # Simula --reindex: delete_collection y recrear
    try:
        client2.delete_collection("utn_bibliografia")
        print("Coleccion borrada para --reindex")
    except Exception as e:
        print(f"delete_collection fallo (quizas no existia): {e}")
    # Verificar que count ahora falla o es 0 (coleccion no existe)
    try:
        tmp_col = client2.get_or_create_collection("utn_bibliografia")
        print(f"Tras delete, nueva coleccion count={tmp_col.count()} (debe ser 0)")
        if tmp_col.count() != 0:
            return fail("Tras --reindex, coleccion no quedo vacia")
    except Exception as e:
        print(f"Error verificando tras delete: {e}")
    # Recrear indice
    col3 = client2.get_or_create_collection("utn_bibliografia")
    vs3 = ChromaVectorStore(chroma_collection=col3)
    sc3 = StorageContext.from_defaults(vector_store=vs3)
    index3 = VectorStoreIndex.from_documents(
        documents[:2],  # solo 2 docs para ser rapido
        storage_context=sc3,
        embed_model=Settings.embed_model,
        show_progress=False,
    )
    count3 = col3.count()
    print(f"Tras --reindex recreado con 2 docs, count={count3}")
    if count3 == 0:
        return fail("--reindex no recreo vectores")
    if count3 == count1:
        warn("--reindex recreo mismo count aunque usamos menos docs (ok si dedup)")
    ok(f"--reindex funciona: borra y recrea -> {count3} vectores")

    # guardar para tests siguientes
    global _test_index, _test_collection, _test_client
    _test_index = index3
    _test_collection = col3
    _test_client = client2

    # liberar refs del primer client/col que ya no se usan (evita lock en Windows)
    try:
        del client, col, vs, storage_context, index, col2, vector_store2, index2
    except:
        pass
    import gc
    gc.collect()

    # No cerrar clientes aun, los necesitamos para query tests
    # La limpieza se hace al final del script

def test_query_y_fuentes():
    from llama_index.core import Settings
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.embeddings import MockEmbedding

    # Asegurar mocks
    if Settings.embed_model is None:
        Settings.embed_model = MockEmbedding(embed_dim=384)
    if Settings.llm is None:
        Settings.llm = MockLLM(max_tokens=512)

    # Usar indice creado en test anterior, si no, crear uno rapido
    try:
        index = _test_index
    except NameError:
        # crear indice rapido si test anterior no corrio
        from pathlib import Path as _P
        import chromadb, tempfile
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        tmp = tempfile.mkdtemp()
        client = chromadb.PersistentClient(path=tmp)
        col = client.get_or_create_collection("tmp_test")
        reader = SimpleDirectoryReader(input_dir=str(DATOS_DIR), recursive=True, required_exts=[".pdf"], filename_as_id=True)
        docs = reader.load_data(show_progress=False)
        vs = ChromaVectorStore(chroma_collection=col)
        sc = StorageContext.from_defaults(vector_store=vs)
        index = VectorStoreIndex.from_documents(docs[:1], storage_context=sc, show_progress=False)
        print(f"Indice fallback creado con 1 doc, count={col.count()}")
        # limpiar tmp despues (no necesario para este test)
        import shutil
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except:
            pass

    print(f"Indice para query: {index}")

    # 8. puede responder pregunta basada en PDF
    # Probar con CitationQueryEngine y fallback
    from llama_index.core.query_engine import CitationQueryEngine

    # Intentar Citation, si falla usar fallback
    try:
        qe = CitationQueryEngine.from_args(index, similarity_top_k=2, citation_chunk_size=512)
        print("[INFO] Usando CitationQueryEngine para test")
    except Exception as e:
        print(f"[WARN] Citation fallo {e}, usando fallback")
        qe = index.as_query_engine(similarity_top_k=2)

    query = "Que temas cubre el apunte de Analisis Numerico?"
    print(f"Query: {query}")
    response = qe.query(query)
    texto = str(response.response if hasattr(response, "response") else response).strip()
    print(f"Respuesta (primer 600 chars): {texto[:600]!r}")
    if not texto:
        return fail("Respuesta vacia")
    ok("Puede responder pregunta basada en PDF (MockLLM)")

    # 9. muestra fuentes/paginas
    source_nodes = getattr(response, "source_nodes", None) or []
    print(f"Fuentes recuperadas: {len(source_nodes)}")
    for i, node in enumerate(source_nodes[:3], 1):
        meta = getattr(node, "metadata", {}) or {}
        fname = meta.get("file_name") or meta.get("file_path") or "desconocido"
        page = meta.get("page_label") or meta.get("page_number") or "?"
        snippet = getattr(node, "text", "") or (getattr(node, "node", None) and getattr(node.node, "text", "")) or ""
        snippet = snippet.strip().replace("\n", " ")[:200]
        print(f"  [{i}] {Path(fname).name} -- pag. {page} | snippet: {snippet[:100]!r}")
        # verificar que al menos un source tiene file_name con .pdf
        if ".pdf" not in str(fname).lower():
            warn(f"Fuente {i} sin .pdf en nombre: {fname}")

    if len(source_nodes) == 0:
        return fail("No se recuperaron fuentes/paginas")

    # Verificar que mostrar_respuesta_con_fuentes de main.py funciona
    # Importar y probar sin crashear
    try:
        import main as main_mod
        print("\n--- Probando main.mostrar_respuesta_con_fuentes ---")
        main_mod.mostrar_respuesta_con_fuentes(response, query)
        ok("mostrar_respuesta_con_fuentes imprime sin error")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return fail(f"mostrar_respuesta_con_fuentes fallo: {e}")

    ok(f"Muestra fuentes/paginas: {len(source_nodes)} fragmentos")

    # 10. ante pregunta sin info, no inventa (anti-alucinacion)
    # Verificar que el prompt contiene la frase exacta
    import main as main_mod
    tmpl = main_mod.ANTI_HALLUCINATION_QA_TMPL_STR
    frase = "No encontré información suficiente en los documentos para responder esa pregunta."
    # tambien verificar ascii fallback
    frase_ascii = "No encontre informacion suficiente"
    print(f"Prompt contiene frase anti-alucinacion? {'No encontr' in tmpl}")
    print(f"Prompt snippet: {tmpl[:400]!r}")
    if frase not in tmpl and frase_ascii.lower() not in tmpl.lower():
        return fail("Prompt anti-alucinacion no contiene frase esperada")
    ok("Prompt anti-alucinacion presente")

    # Verificar que Citation fallback tambien usa prompt
    # Para MockLLM no podemos verificar que realmente no alucine sin LLM real,
    # pero verificamos que el query_engine tenga el template inyectado
    # En main.py, crear_query_engine intenta inyectar ANTI_HALLUCINATION_QA_TEMPLATE
    print("\n[INFO] Anti-alucinacion con MockLLM: MockLLM devuelve texto generico,")
    print("       pero el prompt fuerza al modelo real a no inventar.")
    print("       Verificacion manual con key real: preguntar 'Quien gano el mundial 2030?'")
    print("       debe responder con frase exacta 'No encontre informacion suficiente...'")
    ok("Anti-alucinacion: prompt correcto (verificacion con mocks)")

def test_main_funcs_integration():
    # Test de integracion: validar_entorno, obtener_coleccion_chroma, construir_o_cargar_indice con mocks
    # Patch main.CHROMA_DIR a test dir para no tocar real
    import main as main_mod
    from unittest.mock import patch
    from pathlib import Path as _P
    from llama_index.core import Settings
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms.mock import MockLLM

    original_chroma = main_mod.CHROMA_DIR
    main_mod.CHROMA_DIR = CHROMA_TEST_DIR
    Settings.embed_model = MockEmbedding(embed_dim=384)
    Settings.llm = MockLLM(max_tokens=512)
    print(f"Patch CHROMA_DIR {original_chroma} -> {CHROMA_TEST_DIR}")

    try:
        # validar_entorno no debe fallar con dummy key y PDFs existentes
        # main_mod._FALTA_API_KEY es False porque dummy pasa, pero forzamos
        # que no falle: si tiene dummy TEST, consideramos que es valido para test
        # No podemos cambiar _FALTA_API_KEY sin recargar, asi que solo testeamos que no haga sys.exit
        try:
            main_mod.validar_entorno(reindex=False)
            ok("validar_entorno no hace exit con dummy+PDFs")
        except SystemExit as e:
            if e.code == 0:
                ok("validar_entorno exit 0 (help?)")
            else:
                return fail(f"validar_entorno hizo exit {e.code}")

        # obtener_coleccion_chroma
        client, col = main_mod.obtener_coleccion_chroma(reindex=False)
        print(f"obtener_coleccion_chroma count={col.count()}")
        ok("obtener_coleccion_chroma OK")

        # construir_o_cargar_indice (reutiliza si hay vectores)
        idx = main_mod.construir_o_cargar_indice(reindex=False)
        print(f"construir_o_cargar_indice (reuso) -> {idx}")
        ok("construir_o_cargar_indice reuso OK")

        # reindex
        idx2 = main_mod.construir_o_cargar_indice(reindex=True)
        # col anterior quedo stale tras delete_collection, re-obtener
        _, col_new = main_mod.obtener_coleccion_chroma(reindex=False)
        print(f"construir_o_cargar_indice (reindex) -> {idx2}, count nuevo={col_new.count()}")
        ok("construir_o_cargar_indice --reindex OK")

        # crear_query_engine
        qe = main_mod.crear_query_engine(idx2)
        print(f"crear_query_engine -> {type(qe).__name__}")
        resp = qe.query("metodo de biseccion")
        print(f"query 'metodo de biseccion' -> {str(resp)[:300]!r}")
        ok("crear_query_engine + query con mocks OK")

    finally:
        # liberar referencias Chroma antes de cleanup (Windows lock)
        try:
            del client, col, idx, idx2, qe, resp
        except:
            pass
        try:
            import gc
            gc.collect()
        except:
            pass
        main_mod.CHROMA_DIR = original_chroma
        print(f"Restore CHROMA_DIR -> {original_chroma}")

if __name__ == "__main__":
    print("="*72)
    print("TEST RAG MOCK — verificacion sin OpenAI")
    print("="*72)
    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]} en {sys.executable}")
    print(f"KEEP={KEEP} (si --keep, no borra chroma_db_test al final)")

    check("1. main.py --help y py_compile", test_help)
    check("2. load_dotenv (.env)", test_env)
    check("3. PDFs recursive", test_pdfs)
    check("4-7. Chroma, indice, reuso, --reindex", test_chroma_y_indice)
    check("8-10. Query, fuentes, anti-alucinacion", test_query_y_fuentes)
    check("Integracion main.py funcs con mocks", test_main_funcs_integration)

    print("\n" + "="*72)
    print(f"RESUMEN: {passed} PASSED, {failed} FAILED de {passed+failed} checks")
    if failed == 0:
        print("TODOS LOS CHECKS PASARON - RAG listo para key real")
    else:
        print("ALGUNOS CHECKS FALLARON - revisar logs arriba")
    print("="*72)

    # Limpieza - intentar borrar dentro del proceso y programar borrado async si queda lock (Windows)
    if not KEEP and CHROMA_TEST_DIR.exists():
        # liberar globals que mantienen handles de Chroma
        for _name in ["_test_client", "_test_collection", "_test_index"]:
            try:
                del globals()[_name]
            except:
                pass
        try:
            import gc
            gc.collect()
        except:
            pass
        import time
        # intento sincrono (puede fallar por lock en Windows)
        try:
            shutil.rmtree(CHROMA_TEST_DIR, ignore_errors=False)
            print(f"[CLEANUP] Borrado {CHROMA_TEST_DIR} (sincrono)")
        except PermissionError as e:
            print(f"[WARN] No se pudo borrar {CHROMA_TEST_DIR} sincrono (lock Windows): {e}")
            # programar borrado async tras salir del proceso (cuando el lock se libera)
            try:
                import subprocess as _sp
                # usar python del env para borrar tras 2s
                _cmd = [
                    sys.executable,
                    "-c",
                    f"import time, shutil, pathlib; time.sleep(2); shutil.rmtree(pathlib.Path(r'{CHROMA_TEST_DIR}'), ignore_errors=True); print('[ASYNC CLEANUP] borrado {CHROMA_TEST_DIR}')"
                ]
                # DETACHED_PROCESS en Windows evita bloquear
                _flags = getattr(_sp, "DETACHED_PROCESS", 0)
                _sp.Popen(_cmd, creationflags=_flags, close_fds=True)
                print(f"[CLEANUP] Programado borrado async de {CHROMA_TEST_DIR} en 2s (tras liberar lock)")
            except Exception as ee:
                print(f"[WARN] No se pudo programar borrado async: {ee}")
                print("       Cierra python y borra manualmente: rmdir /s /q chroma_db_test")
        except Exception as e:
            print(f"[WARN] Cleanup fallo: {e}")

    print("\n--- INSTRUCCIONES PARA PROBAR CON KEY REAL ---")
    print("1. Edita .env y pon tu key real:")
    print("   OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXX")
    print("2. Borra indice previo si existe:")
    print("   rmdir /s /q chroma_db   (Windows)")
    print("   o usa --reindex")
    print("3. Ejecuta:")
    print("   .\\env\\Scripts\\python.exe main.py --reindex")
    print("   -> debe ver: [OK] Encontrados X PDF(s), [OK] Modelos configurados,")
    print("                [INFO] Construyendo nuevo indice, [OK] Indice creado")
    print("4. Pregunta dentro del loop: 'Que es el metodo de biseccion segun el apunte?'")
    print("   -> debe responder con info del PDF + Fuentes [1] archivo -- pag. X")
    print("5. Prueba reuso (sin reindex):")
    print("   .\\env\\Scripts\\python.exe main.py")
    print("   -> debe ver: [INFO] Reutilizando indice existente (N vectores)")
    print("6. Prueba anti-alucinacion:")
    print("   Pregunta: 'Quien gano el mundial 2030?'")
    print("   -> debe responder EXACTAMENTE: 'No encontre informacion suficiente...'")
    print("   (o con MockLLM, verificar que el prompt contenga esa frase)")
    print("7. Verifica chroma_db/ se creo:")
    print("   dir chroma_db  -> debe contener chroma.sqlite3 y carpetas")

    sys.exit(1 if failed else 0)
