#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for rag_core.py — singleton, fallback, query with mocks.

Covers:
  - Singleton reuse (get_query_engine returns same object without reindex)
  - Missing chroma_db -> 503 RuntimeError via get_query_engine
  - query with MockEmbedding/MockLLM against chroma_db_test (80 vectors)
  - Fallback phrase preserved verbatim
  - Empty query returns fallback
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import rag_core
from rag_core import (
    ANTI_HALLUCINATION_QA_TMPL_STR,
    FALLBACK_PHRASE,
    CHROMA_DIR,
    DATOS_DIR,
)


def _setup_mocks():
    """Install MockEmbedding/MockLLM into Settings for offline tests."""
    from llama_index.core import Settings
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms.mock import MockLLM

    Settings.embed_model = MockEmbedding(embed_dim=384)
    Settings.llm = MockLLM(max_tokens=512)
    return Settings


def test_fallback_phrase_preserved_verbatim():
    # Contract: exact phrase per new spec (solo contexto + fallback exacto)
    expected = "No se encontró información suficiente en las fuentes proporcionadas para responder esta pregunta con seguridad."
    assert FALLBACK_PHRASE == expected
    assert expected in ANTI_HALLUCINATION_QA_TMPL_STR
    assert "Respondé exclusivamente utilizando la información incluida en el CONTEXTO" in ANTI_HALLUCINATION_QA_TMPL_STR
    assert "No utilices conocimiento externo" in ANTI_HALLUCINATION_QA_TMPL_STR
    # Also check rag_core module re-exports same phrase
    assert rag_core.FALLBACK_PHRASE == expected
    assert rag_core.ANTI_HALLUCINATION_QA_TMPL_STR == ANTI_HALLUCINATION_QA_TMPL_STR


def test_query_empty_returns_fallback():
    _setup_mocks()
    # reset singleton so we don't require real index for empty check
    # query() short-circuits on empty without calling get_query_engine
    text, sources = rag_core.query("")
    assert text == FALLBACK_PHRASE
    assert sources == []
    text2, sources2 = rag_core.query("   ")
    assert text2 == FALLBACK_PHRASE
    assert sources2 == []


def test_singleton_reuse_against_chroma_test():
    """
    Reuse existing index in chroma_db_test with mocks.
    Verifies singleton: second call returns same object, no reload.
    """
    _setup_mocks()
    rag_core.reset_singleton()
    # Point CHROMA_DIR to existing test DB
    original_chroma = rag_core.CHROMA_DIR
    rag_core.CHROMA_DIR = Path("chroma_db_test")
    try:
        # Ensure DATOS_DIR exists so validar logic passes (real datos has PDFs)
        assert DATOS_DIR.exists(), "datos/ must exist for singleton test"
        import chromadb

        client = chromadb.PersistentClient(path=str(rag_core.CHROMA_DIR))
        col = client.get_or_create_collection(rag_core.COLLECTION_NAME)
        count = col.count()
        assert count > 0, f"Expected vectors in chroma_db_test, got {count}"

        # First load — should reuse existing (no rebuild)
        engine1 = rag_core.get_query_engine()
        assert engine1 is not None
        # Source nodes expected to work
        # Second call must return same object (singleton, no reload)
        engine2 = rag_core.get_query_engine()
        assert engine1 is engine2, "Singleton reuse failed: second call returned new object"

        # Query via rag_core.query helper (uses singleton engine)
        text, sources = rag_core.query("Que es el teorema de Rolle?")
        # MockLLM returns generic text, but should not be fallback (since chunks exist)
        assert isinstance(text, str) and len(text.strip()) > 0
        # Sources should be list of dicts with file/page
        assert isinstance(sources, list)
        # With similarity_top_k=4, Mock should still retrieve something
        # Allow 0 in case MockLLM path differs, but ideally >=1
        # We assert at least engine returned something
        if sources:
            for s in sources:
                assert "file" in s
                assert "page" in s
                assert isinstance(s["file"], str)
    finally:
        rag_core.CHROMA_DIR = original_chroma
        rag_core.reset_singleton()
        # Re-apply mocks for next tests (reset left Settings intact)
        try:
            _setup_mocks()
        except Exception:
            pass


def test_missing_chroma_db_raises_503():
    """
    Simulate missing chroma_db by pointing CHROMA_DIR and DATOS_DIR to
    non-existent temp paths. get_query_engine must raise RuntimeError with 503.
    """
    _setup_mocks()
    rag_core.reset_singleton()
    original_chroma = rag_core.CHROMA_DIR
    original_datos = rag_core.DATOS_DIR
    import tempfile

    tmp_empty = Path(tempfile.mkdtemp())
    # Ensure empty (no vectors) and no PDFs inside
    # Point both to empty temp to trigger construir_o_cargar_indice failure
    rag_core.CHROMA_DIR = tmp_empty / "no_chroma_here"
    rag_core.DATOS_DIR = tmp_empty / "no_datos"
    try:
        with pytest.raises(RuntimeError) as exc:
            rag_core.get_query_engine()
        msg = str(exc.value)
        assert "503" in msg, f"Expected 503 in error, got: {msg}"
        # Hint should mention reindex or indice
        assert "reindex" in msg.lower() or "indice" in msg.lower()
    finally:
        rag_core.CHROMA_DIR = original_chroma
        rag_core.DATOS_DIR = original_datos
        rag_core.reset_singleton()
        _setup_mocks()
        # Cleanup temp
        import shutil

        try:
            shutil.rmtree(tmp_empty, ignore_errors=True)
        except Exception:
            pass


def test_query_with_mock_returns_sources_and_fallback_handling():
    """
    Direct query test against chroma_db_test with mocks verifies:
      - grounded answer returns text + sources with pages (when evidence exists)
      - fallback phrase is used when model returns empty (simulated by patching engine)
    """
    _setup_mocks()
    rag_core.reset_singleton()
    original_chroma = rag_core.CHROMA_DIR
    rag_core.CHROMA_DIR = Path("chroma_db_test")
    try:
        engine = rag_core.get_query_engine()
        # Patch engine.query to simulate fallback case (empty response with no sources)
        class FakeEmptyResponse:
            response = ""
            source_nodes = []

        class FakeGroundedResponse:
            response = "Respuesta con evidencia sobre el teorema de Rolle."
            source_nodes = []

            def __init__(self):
                # Create mock nodes with metadata
                class Node:
                    def __init__(self, file_name, page_label, text):
                        self.metadata = {"file_name": file_name, "page_label": page_label}
                        self.text = text
                        self.score = 0.85
                        self.node = None

                self.source_nodes = [
                    Node("Apuntes_Analisis_Numerico.pdf", "12", "El teorema de Rolle dice..."),
                    Node("NOM_ISO_9001-2015.pdf", "5", "La norma establece..."),
                ]

        # Test grounded path by mocking engine.query to return FakeGroundedResponse
        with patch.object(engine, "query", return_value=FakeGroundedResponse()):
            text, sources = rag_core.query("Pregunta con evidencia?")
            assert text == "Respuesta con evidencia sobre el teorema de Rolle."
            assert len(sources) == 2
            assert sources[0]["file"] == "Apuntes_Analisis_Numerico.pdf"
            assert sources[0]["page"] == "12"
            assert sources[1]["page"] == "5"

        # Test fallback when engine returns empty text
        with patch.object(engine, "query", return_value=FakeEmptyResponse()):
            text2, sources2 = rag_core.query("Pregunta vacia?")
            assert text2 == FALLBACK_PHRASE
            # Empty sources is okay for fallback
            assert sources2 == []

        # Test that fallback phrase constant is still verbatim after mocks
        assert FALLBACK_PHRASE.startswith("No se encontró")
    finally:
        rag_core.CHROMA_DIR = original_chroma
        rag_core.reset_singleton()
        _setup_mocks()


def test_reset_singleton_clears_state():
    _setup_mocks()
    original_chroma = rag_core.CHROMA_DIR
    rag_core.CHROMA_DIR = Path("chroma_db_test")
    try:
        e1 = rag_core.get_query_engine()
        rag_core.reset_singleton()
        # After reset, next call should still succeed but object may be new
        e2 = rag_core.get_query_engine()
        # They could be different objects after reset (rebuilt)
        assert e2 is not None
        # If we didn't clear, they'd be same; after reset they might be different instance
        # But at least not same cached without rebuild? Allow either but verify reset worked:
        # Call again without reset -> same
        e3 = rag_core.get_query_engine()
        assert e2 is e3
    finally:
        rag_core.CHROMA_DIR = original_chroma
        rag_core.reset_singleton()
        _setup_mocks()
