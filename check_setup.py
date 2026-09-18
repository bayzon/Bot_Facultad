#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_setup.py — Health/diagnostic for Bot Facultad. Prints [OK]/[ERROR], never secrets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _present(name: str) -> bool:
    v = os.getenv(name, "").strip()
    if not v:
        return False
    if v in ("REEMPLAZAR", "tu_api_key_aqui", "sk-...", "tu_verify_token_seguro"):
        return False
    if "REEMPLAZAR" in v or "xxxx" in v.lower():
        # Placeholder like EAAxxxx — treat as missing for setup purposes
        if len(v) < 12:
            return False
        # EAA placeholder with xxxx is still placeholder
        if "xxxx" in v.lower():
            return False
    if "TEST" in v:
        return False
    return len(v) >= 4


def main() -> int:
    print("Bot Facultad — check_setup")
    print("=" * 48)
    errors = 0

    def report(ok: bool, label: str, hint: str = ""):
        nonlocal errors
        tag = "[OK]" if ok else "[ERROR]"
        if not ok:
            errors += 1
        print(f"{tag} {label}" + (f" — {hint}" if hint and not ok else ""))

    # Meta
    report(_present("VERIFY_TOKEN"), "Meta VERIFY_TOKEN", "set VERIFY_TOKEN in .env")
    report(_present("WHATSAPP_TOKEN"), "Meta WHATSAPP_TOKEN", "set WHATSAPP_TOKEN in .env")
    report(_present("WHATSAPP_PHONE_NUMBER_ID"), "Meta WHATSAPP_PHONE_NUMBER_ID", "set WHATSAPP_PHONE_NUMBER_ID in .env")
    # APP_SECRET optional in dev but required in prod
    if _present("WHATSAPP_APP_SECRET"):
        report(True, "Meta WHATSAPP_APP_SECRET")
    else:
        print("[OK] Meta WHATSAPP_APP_SECRET (dev mode: HMAC skipped — set in prod)")

    # Google
    report(_present("GOOGLE_SERVICE_ACCOUNT_JSON") or Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip() or "./bot-facultad-sa.json").exists(), "Google SERVICE_ACCOUNT_JSON", "set GOOGLE_SERVICE_ACCOUNT_JSON path in .env")
    report(_present("GOOGLE_DRIVE_FOLDER_ID"), "Google DRIVE_FOLDER_ID", "set GOOGLE_DRIVE_FOLDER_ID in .env")

    # LLM (at least one)
    has_gemini = _present("GEMINI_API_KEY")
    has_openai = _present("OPENAI_API_KEY")
    has_groq = _present("GROQ_API_KEY")
    report(has_gemini or has_openai or has_groq, "LLM provider key (GEMINI/OPENAI/GROQ)", "set GEMINI_API_KEY or OPENAI_API_KEY in .env")
    if has_gemini:
        print("[OK] LLM provider: Gemini")
    elif has_openai:
        print("[OK] LLM provider: OpenAI")
    elif has_groq:
        print("[OK] LLM provider: Groq")

    # 3 materias
    try:
        import materias_config as mc

        wl = list(getattr(mc, "MATERIAS_WHITELIST", []))
        expected = ["Redes de Datos", "Gestión de Calidad", "Ingeniería de Software"]
        ok = wl == expected
        report(ok, f"3 materias activas ({', '.join(wl) if wl else 'none'})", f"expected {expected}")
    except Exception as e:
        report(False, "3 materias activas", f"import failed: {e}")

    # Fuentes per materia
    try:
        import rag_core as rc

        for canon, folder in [
            ("Redes de Datos", Path("fuentes/Redes de Datos")),
            ("Gestión de Calidad", Path("fuentes/Gestion de Calidad")),
            ("Ingeniería de Software", Path("fuentes/Ingenieria de Software")),
        ]:
            files = []
            if folder.exists():
                files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".doc") and p.name != ".gitkeep" and not p.name.startswith("README")]
            # Legacy fallback
            legacy_count = 0
            try:
                for base in [Path("datos")]:
                    if base.exists():
                        for p in base.rglob("*"):
                            if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".doc"):
                                try:
                                    if rc.materia_for_file(str(p)) == canon:
                                        legacy_count += 1
                                except Exception:
                                    pass
            except Exception:
                pass
            total = len(files) + legacy_count
            if total > 0:
                report(True, f"Fuentes {canon} ({total} archivo(s))")
            else:
                report(False, f"Fuentes {canon} (0 archivos)", f"add PDFs to {folder}/ then python main.py --reindex (bot still starts)")
    except Exception as e:
        report(False, "Fuentes por materia", f"scan failed: {e}")

    # Docs cache
    try:
        cache = Path("materias_docs.json")
        if cache.exists():
            report(True, "Docs cache materias_docs.json")
        else:
            print("[OK] Docs cache materias_docs.json (missing = will be created on first save)")
    except Exception:
        pass

    print("=" * 48)
    if errors:
        print(f"Result: {errors} error(s) — bot can still start, fix above for full flow.")
    else:
        print("Result: all checks passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
