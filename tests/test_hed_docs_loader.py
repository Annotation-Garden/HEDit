"""Tests for HED documentation loader."""

from unittest.mock import patch

from src.utils.hed_docs_loader import MAX_DOC_CHARS, clear_cache, load_hed_docs


class TestLoadHedDocs:
    """Tests for load_hed_docs function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_load_returns_both_docs(self):
        """Test that both official docs are loaded."""
        docs = load_hed_docs()

        assert "annotation_semantics" in docs
        assert "terminology" in docs
        assert len(docs["annotation_semantics"]) > 0
        assert len(docs["terminology"]) > 0

    def test_annotation_semantics_has_key_content(self):
        """Test annotation semantics doc contains expected content."""
        docs = load_hed_docs()
        content = docs["annotation_semantics"]

        # Key concepts from HedAnnotationSemantics.md
        assert "reversibility" in content.lower()
        assert "grouping" in content.lower()
        assert "Sensory-event" in content

    def test_terminology_has_key_content(self):
        """Test terminology doc contains expected content."""
        docs = load_hed_docs()
        content = docs["terminology"]

        # Key terms from 02_Terminology.md
        assert "HED tag" in content
        assert "Tag-group" in content

    def test_caching(self):
        """Test that second call returns cached result."""
        docs1 = load_hed_docs()
        docs2 = load_hed_docs()

        assert docs1 is docs2

    def test_clear_cache(self):
        """Test that clear_cache forces reload."""
        docs1 = load_hed_docs()
        clear_cache()
        docs2 = load_hed_docs()

        # Same content but different object
        assert docs1 is not docs2
        assert docs1 == docs2

    def test_missing_files_returns_empty(self):
        """Test graceful fallback when files are missing."""
        with patch(
            "src.utils.hed_docs_loader._get_docs_dir",
            return_value=__import__("pathlib").Path("/nonexistent/path"),
        ):
            clear_cache()
            docs = load_hed_docs()

        assert docs == {}

    def test_truncation_not_needed(self):
        """Test that bundled docs are under the truncation limit."""
        docs = load_hed_docs()

        for doc_id, content in docs.items():
            assert len(content) <= MAX_DOC_CHARS, (
                f"{doc_id} exceeds {MAX_DOC_CHARS} chars: {len(content)}"
            )


class TestCleanMystMarkdown:
    """Tests for MyST directive processing in fetched docs."""

    def test_no_raw_myst_directives_in_bundled_docs(self):
        """Test that bundled docs have no raw MyST directives."""
        docs = load_hed_docs()

        for doc_id, content in docs.items():
            assert "```{admonition}" not in content, f"{doc_id} contains raw admonition directive"
            assert "```{list-table}" not in content, f"{doc_id} contains raw list-table directive"
            assert "```{index}" not in content, f"{doc_id} contains raw index directive"
            assert "```{code-block}" not in content, f"{doc_id} contains raw code-block directive"

    def test_no_quadruple_bold_markers(self):
        """Test that admonition titles don't produce ****."""
        docs = load_hed_docs()

        for doc_id, content in docs.items():
            assert "****" not in content, f"{doc_id} contains quadruple bold markers (****)"

    def test_headings_preserved(self):
        """Test that markdown headings are preserved."""
        docs = load_hed_docs()
        content = docs.get("annotation_semantics", "")

        # Should have markdown headings
        assert "## " in content or "# " in content

    def test_code_blocks_preserved(self):
        """Test that code examples are preserved."""
        docs = load_hed_docs()
        content = docs.get("annotation_semantics", "")

        # Should have code blocks with HED examples
        assert "```" in content
