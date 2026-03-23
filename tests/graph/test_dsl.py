"""Tests for the GDSP DSL parser (tokenizer, parser, compiler)."""

from __future__ import annotations

from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph.dsl import (
    GDSPSyntaxError,
    Parser,
    parse,
    parse_file,
    parse_multi,
    tokenize,
)


class TestTokenizer:
    """Tokenizer smoke tests."""

    def test_basic_tokens(self) -> None:
        """Test tokenizing a graph header."""
        tokens = tokenize("graph foo { }")
        values = [t.value for t in tokens if t.value != ""]
        assert "graph" in values
        assert "foo" in values

    def test_unterminated_string(self) -> None:
        """Test unterminated strings raise syntax errors."""
        with pytest.raises(GDSPSyntaxError, match="unterminated string"):
            tokenize('"unterminated')


class TestParser:
    """Parser smoke tests."""

    def test_parse_graph(self) -> None:
        """Test parse returns a graph list."""
        graphs = Parser(tokenize("graph empty { }")).parse_file()
        assert len(graphs) == 1
        assert graphs[0].name == "empty"

    def test_parse_helper(self) -> None:
        """Test parse helper returns a graph model."""
        graph = parse("graph test { }")
        assert graph.name == "test"

    def test_parse_multi(self) -> None:
        """Test multi-graph parsing."""
        graphs = parse_multi("graph a { }\ngraph b { }")
        assert [g.name for g in graphs] == ["a", "b"]

    def test_parse_file(self, tmp_path: Path) -> None:
        """Test parsing a graph file."""
        path = tmp_path / "graph.gdsp"
        path.write_text("graph file_graph { }")
        graph = parse_file(path)
        assert graph.name == "file_graph"
