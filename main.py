#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bot Facultad — wrapper CLI delegating RAG to rag_core singleton."""
import argparse, sys
from pathlib import Path
try:
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    if hasattr(sys.stderr,"reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8",errors="replace")
except Exception:
    pass
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
import rag_core as _rc
from rag_core import ANTI_HALLUCINATION_QA_TMPL_STR,FALLBACK_PHRASE,configurar_modelos,crear_query_engine,construir_o_cargar_indice,get_query_engine,obtener_coleccion_chroma,query,validar_entorno
CHROMA_DIR=_rc.CHROMA_DIR; DATOS_DIR=_rc.DATOS_DIR
def _sync(): _rc.CHROMA_DIR=CHROMA_DIR; _rc.DATOS_DIR=DATOS_DIR
# wrappers for legacy test_rag_mock patching main.CHROMA_DIR
def validar_entorno(reindex=False):
    _sync(); return _rc.validar_entorno(reindex)
def configurar_modelos():
    _sync(); return _rc.configurar_modelos()
def obtener_coleccion_chroma(reindex=False):
    _sync(); return _rc.obtener_coleccion_chroma(reindex)
def construir_o_cargar_indice(reindex=False, fail_fast=False):
    _sync(); return _rc.construir_o_cargar_indice(reindex, fail_fast=fail_fast)
def crear_query_engine(index):
    _sync(); return _rc.crear_query_engine(index)
def get_query_engine(reindex=False, fail_fast=False):
    _sync(); return _rc.get_query_engine(reindex, fail_fast=fail_fast)
def query(q):
    _sync(); return _rc.query(q)
def mostrar_respuesta_con_fuentes(response,query:str):
    print("\n"+"="*72);print("RESPUESTA:");print("-"*72)
    texto=str(getattr(response,"response",response)).strip()
    print(texto or "[Sin texto]");print("="*72)
    nodes=getattr(response,"source_nodes",None) or []
    if not nodes: print("(Sin fuentes)");return
    print(f"\nFuentes ({len(nodes)}):")
    for i,n in enumerate(nodes,1):
        m=getattr(n,"metadata",{}) or {}
        f=m.get("file_name") or m.get("file_path") or "archivo desconocido"
        try: f=Path(f).name
        except Exception: pass
        p=m.get("page_label") or m.get("page_number") or "?"
        s=getattr(n,"score",None); sc=f" | {s:.3f}" if isinstance(s,(int,float)) else ""
        txt=getattr(n,"text","") or (getattr(getattr(n,"node",None),"text","") or "")
        txt=txt.strip().replace("\n"," ")[:400]
        print(f"\n  [{i}] {f} — pág. {p}{sc}")
        print(f'      "{txt}"' if txt else "      (sin texto)")
    print("-"*72+"\n")
def bucle_interactivo(qe):
    print("\n"+"#"*72);print("#  Sistema RAG UTN listo. Escribí tu pregunta.");print("#  Salir: salir/exit/Ctrl+C");print("#"*72+"\n")
    while True:
        try: q=input("Pregunta > ").strip()
        except (KeyboardInterrupt,EOFError): print("\nHasta luego!");break
        if not q: continue
        if q.lower() in ("salir","exit","quit","q"): print("Hasta luego!");break
        try:
            print(f"\n[Buscando en {DATOS_DIR}/ ...]")
            r=qe.query(q); mostrar_respuesta_con_fuentes(r,q)
        except KeyboardInterrupt: print("\n[Interrumpido]");continue
        except Exception as e:
            m=str(e);print(f"\n[ERROR] {m}")
            if "api_key" in m.lower(): print("  -> Verifica OPENAI_API_KEY")
            elif "rate" in m.lower() or "quota" in m.lower(): print("  -> Cuota/rate alcanzado")
            elif "timeout" in m.lower(): print("  -> Timeout")
            print("  Podés seguir preguntando.\n")
def main():
    p=argparse.ArgumentParser(description="RAG local UTN — consulta PDFs con Chroma + OpenAI",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reindex",action="store_true",help="Fuerza reconstrucción del índice Chroma")
    p.add_argument("--no-interactive",action="store_true",help="Build the index then exit without the interactive loop (for CI/build).")
    a=p.parse_args(); validar_entorno(reindex=a.reindex); configurar_modelos(); qe=get_query_engine(reindex=a.reindex)
    if a.no_interactive:
        print("[INFO] Index ready (non-interactive, exiting).")
        sys.exit(0)
    bucle_interactivo(qe)
if __name__=="__main__": main()
