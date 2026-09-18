#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_question.py — CLI: detection -> exclusive retrieval -> LLM -> Docs.

Usage:
  python test_question.py "Redes de Datos: ¿Qué es una VLAN?"
  python test_question.py --dry-run "Redes de Datos: ¿Qué es una VLAN?"

--dry-run: no writes to Google Docs (skips Docs append).
Exit codes: 0 ok, 2 usage/format error, 3 docs failure.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Test a MATERIA: pregunta end-to-end.")
    parser.add_argument("message", nargs="?", help='e.g. "Redes de Datos: ¿Qué es una VLAN?"')
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Google Docs.")
    args = parser.parse_args()

    if not args.message or not args.message.strip():
        print('Usage: python test_question.py [--dry-run] "Materia: pregunta"')
        print('Example: python test_question.py "Redes de Datos: ¿Qué es una VLAN?"')
        return 2

    raw = args.message.strip()

    import materias_config

    if ":" not in raw:
        print("❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?")
        return 2

    parsed = materias_config.parse_message(raw)
    if parsed is None:
        print("❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?")
        return 2

    raw_materia, question = parsed
    canon = materias_config.validate_materia(raw_materia)
    if canon is None:
        print(f'❌ No encontré la materia "{raw_materia.strip()}".')
        return 2

    print(f"[Materia] {canon}")
    print(f"[Pregunta] {question}")

    import rag_core

    print(f"[RAG] Consultando exclusivamente {canon}")
    try:
        answer, fuentes = rag_core.query_for_materia(canon, question)
    except RuntimeError as e:
        print(f"[RAG] Engine unavailable (503): {e}")
        print("Hint: python main.py --reindex tras preparar fuentes/")
        return 3
    except Exception as e:
        print(f"[RAG] Query failed: {e}")
        return 3

    print(f"[RAG] {len(fuentes)} fragmentos encontrados")
    print("[LLM] Respuesta generada")
    print("-" * 48)
    print(answer)
    if fuentes:
        print("-" * 48)
        for s in fuentes:
            f = s.get("file", "?")
            p = s.get("page", "?")
            if p and p != "?":
                print(f"- {f} — página {p}")
            else:
                print(f"- {f}")
    print("-" * 48)

    if args.dry_run:
        print("[Google Docs] dry-run: no se escribió (usa sin --dry-run para guardar).")
        return 0

    import drive_docs_service

    fecha_iso = datetime.now(timezone.utc).isoformat()
    ok = drive_docs_service.append_entry(
        materia=canon,
        fecha_iso=fecha_iso,
        pregunta=question,
        respuesta=answer,
        fuentes=fuentes,
        message_id=f"cli-{datetime.now(timezone.utc).timestamp()}",
    )
    if not ok:
        print("❌ No pude guardar la pregunta. Intentá nuevamente.")
        return 3
    print("[Google Docs] Pregunta guardada")
    print("✅ Pregunta guardada y respondida")
    return 0


if __name__ == "__main__":
    sys.exit(main())
