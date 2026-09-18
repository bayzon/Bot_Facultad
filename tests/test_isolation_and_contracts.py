#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""18 mandatory tests: isolation x6, WhatsApp exact, Docs, dedup, aliases, rejections."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import materias_config
import rag_core
from rag_core import FALLBACK_PHRASE


# --- Isolation x6 ------------------------------------------------------------

def test_iso_1_redes_filter_before_llm():
    """query_for_materia must pass MetadataFilters(materia=Redes) to the engine."""
    mock_index = MagicMock()
    mock_engine = MagicMock()
    mock_engine.query.return_value = MagicMock(response="R", source_nodes=[])
    mock_index.as_query_engine.return_value = mock_engine
    with patch.object(rag_core, "get_index", return_value=mock_index):
        # Force verified sources via materia metadata
        node = MagicMock()
        node.metadata = {"file_name": "vlan.pdf", "page_label": "3", "materia": "Redes de Datos", "file_path": "fuentes/Redes de Datos/vlan.pdf"}
        mock_engine.query.return_value = MagicMock(response="Una VLAN es...", source_nodes=[node])
        answer, fuentes = rag_core.query_for_materia("Redes de Datos", "¿Qué es una VLAN?")
        assert mock_index.as_query_engine.called
        kwargs = mock_index.as_query_engine.call_args[1]
        filt = kwargs.get("filters")
        assert filt is not None
        vals = [f.value for f in filt.filters]
        assert "Redes de Datos" in vals
        assert answer == "Una VLAN es..."
        assert fuentes and fuentes[0]["file"] == "vlan.pdf"


def test_iso_2_calidad_filter_before_llm():
    mock_index = MagicMock()
    mock_engine = MagicMock()
    node = MagicMock()
    node.metadata = {"file_name": "iso.pdf", "page_label": "5", "materia": "Gestión de Calidad", "file_path": "fuentes/Gestion de Calidad/iso.pdf"}
    mock_engine.query.return_value = MagicMock(response="ISO dice...", source_nodes=[node])
    mock_index.as_query_engine.return_value = mock_engine
    with patch.object(rag_core, "get_index", return_value=mock_index):
        answer, fuentes = rag_core.query_for_materia("Gestión de Calidad", "¿Qué dice ISO?")
        kwargs = mock_index.as_query_engine.call_args[1]
        assert "Gestión de Calidad" in [f.value for f in kwargs["filters"].filters]
        assert answer == "ISO dice..."


def test_iso_3_software_filter_before_llm():
    mock_index = MagicMock()
    mock_engine = MagicMock()
    node = MagicMock()
    node.metadata = {"file_name": "scrum.pdf", "page_label": "7", "materia": "Ingeniería de Software", "file_path": "fuentes/Ingenieria de Software/scrum.pdf"}
    mock_engine.query.return_value = MagicMock(response="Scrum es...", source_nodes=[node])
    mock_index.as_query_engine.return_value = mock_engine
    with patch.object(rag_core, "get_index", return_value=mock_index):
        answer, fuentes = rag_core.query_for_materia("Ingeniería de Software", "¿Qué es Scrum?")
        kwargs = mock_index.as_query_engine.call_args[1]
        assert "Ingeniería de Software" in [f.value for f in kwargs["filters"].filters]
        assert answer == "Scrum es..."


def test_iso_4_no_global_fallback_unknown():
    """Unknown/inactive materia never touches the index (fail-closed, no global search)."""
    mock_index = MagicMock()
    with patch.object(rag_core, "get_index", return_value=mock_index):
        answer, fuentes = rag_core.query_for_materia("Matemática Superior", "¿Qué es?")
        assert answer == FALLBACK_PHRASE
        assert fuentes == []
        mock_index.as_query_engine.assert_not_called()


def test_iso_5_materia_for_file_mapping():
    assert rag_core.materia_for_file("fuentes/Redes de Datos/vlan.pdf") == "Redes de Datos"
    assert rag_core.materia_for_file("fuentes/Gestion de Calidad/iso.pdf") == "Gestión de Calidad"
    assert rag_core.materia_for_file("fuentes/Ingenieria de Software/scrum.pdf") == "Ingeniería de Software"
    assert rag_core.materia_for_file("fuentes by bayzon/Sistema y Gestion de calidad (Electiva)/NOM.pdf") == "Gestión de Calidad"
    assert rag_core.materia_for_file("fuentes by bayzon/Ingenieria de Software/scrum.pdf") == "Ingeniería de Software"
    assert rag_core.materia_for_file("datos/Gestion de calidad/NOM.pdf") == "Gestión de Calidad"
    # Inactive ignored
    assert rag_core.materia_for_file("datos/Analisis Numerico/apunte.pdf") is None
    assert rag_core.materia_for_file("datos/Fisica/f.pdf") is None


def test_iso_6_collection_for_single_collection():
    assert rag_core.collection_for("Redes de Datos") == rag_core.COLLECTION_NAME
    assert rag_core.collection_for("Gestión de Calidad") == rag_core.COLLECTION_NAME
    assert rag_core.collection_for("Ingeniería de Software") == rag_core.COLLECTION_NAME


# --- WhatsApp exact + never academic -----------------------------------------

def test_wa_success_exact_and_never_academic():
    import webhook_server

    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = ("Secreto académico con http://x y página 5", [{"file": "a.pdf", "page": "5"}])
    mock_rag.FALLBACK_PHRASE = FALLBACK_PHRASE
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = True
    mock_drive.is_already_processed.return_value = False
    mock_wa = MagicMock()
    mock_wa.send_text.return_value = True
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_wa):
        webhook_server.clear_lru_for_tests()
        webhook_server.process_question("wamid.iso7", "54911", "Redes de Datos: ¿Qué es una VLAN?")
        body = mock_wa.send_text.call_args[0][1]
        assert body == "✅ Pregunta guardada y respondida"
        assert "Secreto" not in body
        assert "a.pdf" not in body
        assert "página" not in body
        assert "http" not in body.lower()


# --- Docs saves Q+A+fuentes+fallback ------------------------------------------

def test_docs_saves_qa_fuentes_and_fallback(monkeypatch):
    import drive_docs_service

    monkeypatch.setattr(drive_docs_service, "get_or_create_doc", lambda m: "doc1")
    mock_docs = MagicMock()
    mock_get = MagicMock()
    mock_get.execute.return_value = {"body": {"content": [{"endIndex": 10}]}}
    mock_docs.documents.return_value.get.return_value = mock_get
    mock_batch = MagicMock()
    mock_batch.execute.return_value = {}
    mock_docs.documents.return_value.batchUpdate.return_value = mock_batch
    monkeypatch.setattr(drive_docs_service, "_get_docs_service", lambda: mock_docs)

    # Grounded
    assert drive_docs_service.append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q?", "A con evidencia", [{"file": "r.pdf", "page": "2"}], "wamid.iso9a")
    text = mock_docs.documents.return_value.batchUpdate.call_args[1]["body"]["requests"][0]["insertText"]["text"]
    assert "Q?" in text and "A con evidencia" in text and "- r.pdf — página 2" in text
    # Fallback
    assert drive_docs_service.append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q2?", FALLBACK_PHRASE, [], "wamid.iso9b")
    text2 = mock_docs.documents.return_value.batchUpdate.call_args[1]["body"]["requests"][0]["insertText"]["text"]
    assert FALLBACK_PHRASE in text2


def test_fallback_not_in_whatsapp_still_success():
    import webhook_server

    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = (FALLBACK_PHRASE, [])
    mock_rag.FALLBACK_PHRASE = FALLBACK_PHRASE
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = True
    mock_drive.is_already_processed.return_value = False
    mock_wa = MagicMock()
    mock_wa.send_text.return_value = True
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_wa):
        webhook_server.clear_lru_for_tests()
        webhook_server.process_question("wamid.iso10", "54911", "Redes de Datos: xyz inexistente")
        # Docs got fallback
        assert mock_drive.append_entry.call_args[1]["respuesta"] == FALLBACK_PHRASE
        # WhatsApp got success, not fallback
        body = mock_wa.send_text.call_args[0][1]
        assert body == "✅ Pregunta guardada y respondida"
        assert FALLBACK_PHRASE not in body


# --- wamid dedup + no mark on fail --------------------------------------------

def test_wamid_no_duplica(monkeypatch):
    import drive_docs_service

    monkeypatch.setattr(drive_docs_service, "get_or_create_doc", lambda m: "d")
    mock_docs = MagicMock()
    mock_get = MagicMock()
    mock_get.execute.return_value = {"body": {"content": [{"endIndex": 10}]}}
    mock_docs.documents.return_value.get.return_value = mock_get
    mock_batch = MagicMock()
    mock_batch.execute.return_value = {}
    mock_docs.documents.return_value.batchUpdate.return_value = mock_batch
    monkeypatch.setattr(drive_docs_service, "_get_docs_service", lambda: mock_docs)
    drive_docs_service.clear_dedup_for_tests()
    assert drive_docs_service.append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q", "A", [], "wamid.iso11")
    assert drive_docs_service.append_entry("Redes de Datos", "2026-09-10T12:00:00+00:00", "Q", "A", [], "wamid.iso11")
    assert mock_batch.execute.call_count == 1


def test_no_marca_si_docs_falla():
    import webhook_server

    # LRU unmarked + error message, never success
    mock_rag = MagicMock()
    mock_rag.query_for_materia.return_value = ("A", [])
    mock_rag.FALLBACK_PHRASE = FALLBACK_PHRASE
    mock_drive = MagicMock()
    mock_drive.append_entry.return_value = False
    mock_drive.is_already_processed.return_value = False
    mock_wa = MagicMock()
    mock_wa.send_text.return_value = True
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_wa):
        webhook_server.clear_lru_for_tests()
        mid = "wamid.iso12"
        webhook_server.process_question(mid, "54911", "Redes de Datos: ¿Qué es una VLAN?")
        body = mock_wa.send_text.call_args[0][1]
        assert body == "❌ No pude guardar la pregunta. Intentá nuevamente."
        # LRU must be unmarked so retry is allowed
        from webhook_server import _seen_ids

        assert mid not in _seen_ids


# --- aliases / rejections / help / private / format ----------------------------

def test_aliases_tres_materias_y_parse_vlan():
    assert materias_config.parse_message("redes: ¿Qué es una VLAN?")[0] == "redes"
    assert materias_config.validate_materia("redes") == "Redes de Datos"
    assert materias_config.validate_materia("gc") == "Gestión de Calidad"
    assert materias_config.validate_materia("is") == "Ingeniería de Software"
    assert materias_config.resolve_materia("calidad") == "Gestión de Calidad"


def test_materia_extra_rechazada():
    import webhook_server
    from unittest.mock import MagicMock, patch

    mock_rag = MagicMock()
    mock_drive = MagicMock()
    mock_wa = MagicMock()
    mock_wa.send_text.return_value = True
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_wa):
        webhook_server.clear_lru_for_tests()
        webhook_server.process_question("wamid.iso14", "54911", "Matemática Superior: ¿Qué es?")
        assert mock_wa.send_text.call_args[0][1] == '❌ No encontré la materia "Matemática Superior".'
        assert not mock_drive.append_entry.called


def test_sin_dos_puntos_ayuda_exacta():
    import webhook_server

    mock_rag = MagicMock()
    mock_drive = MagicMock()
    mock_wa = MagicMock()
    mock_wa.send_text.return_value = True
    with patch("webhook_server.rag_core", mock_rag), \
         patch("webhook_server.drive_docs_service", mock_drive), \
         patch("webhook_server.whatsapp_client", mock_wa):
        webhook_server.clear_lru_for_tests()
        webhook_server.process_question("wamid.iso15", "54911", "hola sin dos puntos")
        assert mock_wa.send_text.call_args[0][1] == "❌ Indicá la materia antes de la pregunta.\n\nEjemplo:\nRedes de Datos: ¿Qué es una VLAN?"
        assert not mock_drive.append_entry.called


def test_docs_nunca_publicos(monkeypatch):
    import drive_docs_service
    from unittest.mock import MagicMock, patch

    monkeypatch.setattr(drive_docs_service, "_get_credentials", lambda: MagicMock())
    monkeypatch.setattr(drive_docs_service, "_get_folder_id", lambda: "F")
    mock_drive = MagicMock()
    mock_files = MagicMock()
    mock_create = MagicMock()
    mock_create.execute.return_value = {"id": "x"}
    mock_files.create.return_value = mock_create
    mock_drive.files.return_value = mock_files
    mock_perms = MagicMock()
    mock_drive.permissions.return_value = mock_perms
    with patch("googleapiclient.discovery.build", return_value=mock_drive):
        drive_docs_service.reset_cache_for_tests()
        try:
            drive_docs_service.CACHE_PATH.unlink()
        except Exception:
            pass
        # Use tmp cache to avoid touching real file
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "m.json"
        monkeypatch.setattr(drive_docs_service, "CACHE_PATH", tmp)
        drive_docs_service.reset_cache_for_tests()
        drive_docs_service.get_or_create_doc("Redes de Datos")
        mock_perms.create.assert_not_called()
        assert not mock_drive.permissions.called


def test_docs_formato_exacto_y_numeracion():
    import drive_docs_service
    import re

    block = drive_docs_service._format_block(
        "2026-09-10T12:00:00+00:00", "¿Qué es una VLAN?", "Una red virtual.",
        [{"file": "r.pdf", "page": "4"}], numero=5,
    )
    assert block.startswith("--------------------------------------------------\n\nPregunta 5\n")
    assert "\nRespuesta:\n\nUna red virtual.\n" in block
    assert "\nFuentes utilizadas:\n- r.pdf — página 4\n" in block
    assert re.search(r"\nFecha:\n\d{2}/\d{2}/\d{4} \d{2}:\d{2}\n", block)
    assert block.rstrip().endswith("--------------------------------------------------")
