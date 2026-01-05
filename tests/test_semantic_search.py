"""Tests for semantic search module."""

import pytest

from src.utils.semantic_search import (
    KEYWORD_INDEX,
    SemanticSearchManager,
    TagMatch,
    get_semantic_search_manager,
)


class TestKeywordIndex:
    """Tests for the KEYWORD_INDEX deterministic lookup."""

    def test_keyword_index_not_empty(self):
        """Keyword index should have entries."""
        assert len(KEYWORD_INDEX) > 0

    def test_keyword_index_has_common_terms(self):
        """Keyword index should include common neuroscience terms."""
        assert "mouse" in KEYWORD_INDEX
        assert "reward" in KEYWORD_INDEX
        assert "seizure" in KEYWORD_INDEX
        assert "stimulus" in KEYWORD_INDEX

    def test_keyword_index_values_are_lists(self):
        """Each keyword should map to a list of HED tags."""
        for keyword, tags in KEYWORD_INDEX.items():
            assert isinstance(tags, list), f"'{keyword}' should map to a list"
            assert len(tags) > 0, f"'{keyword}' should have at least one tag"

    def test_keyword_index_library_prefixes(self):
        """Library schemas should have proper prefixes."""
        # SCORE library tags
        assert "seizure" in KEYWORD_INDEX
        seizure_tags = KEYWORD_INDEX["seizure"]
        assert any(tag.startswith("sc:") for tag in seizure_tags)


class TestSemanticSearchManager:
    """Tests for SemanticSearchManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        return SemanticSearchManager()

    def test_initialization(self, manager):
        """Manager should initialize without errors."""
        assert manager is not None
        assert manager.model_id == "Qwen/Qwen3-Embedding-0.6B"

    def test_find_by_keyword_exact_match(self, manager):
        """Should find tags for exact keyword match."""
        results = manager.find_by_keyword("mouse")
        assert len(results) > 0
        assert all(isinstance(r, TagMatch) for r in results)

        # Should include Animal tag
        tags = [r.tag for r in results]
        assert "Animal" in tags or "Animal-agent" in tags

    def test_find_by_keyword_case_insensitive(self, manager):
        """Keyword lookup should be case insensitive."""
        results_lower = manager.find_by_keyword("mouse")
        results_upper = manager.find_by_keyword("MOUSE")
        results_mixed = manager.find_by_keyword("Mouse")

        assert len(results_lower) == len(results_upper) == len(results_mixed)

    def test_find_by_keyword_unknown(self, manager):
        """Unknown keyword should return empty list."""
        results = manager.find_by_keyword("xyznonexistent123")
        assert results == []

    def test_find_by_keyword_score(self, manager):
        """Keyword matches should have high confidence score."""
        results = manager.find_by_keyword("reward")
        assert len(results) > 0
        for result in results:
            assert result.score == 0.95  # Exact keyword match score
            assert result.source == "keyword"

    def test_find_by_keyword_library_prefix(self, manager):
        """Library tags should have correct prefix."""
        results = manager.find_by_keyword("seizure")
        assert len(results) > 0

        # Should include SCORE library tag with prefix
        prefixed_tags = [(r.prefix, r.tag) for r in results]
        assert any(prefix == "sc:" for prefix, _tag in prefixed_tags)

    def test_find_similar_without_embeddings(self, manager):
        """find_similar should work with keyword index only when no embeddings."""
        # Without embeddings loaded, should still find keyword matches
        results = manager.find_similar(["mouse", "reward"], use_embeddings=False)
        assert len(results) > 0

        # Should combine results from both keywords
        tags = [r.tag for r in results]
        assert "Animal" in tags or "Reward" in tags or "Animal-agent" in tags

    def test_find_similar_multiple_keywords(self, manager):
        """Should combine results from multiple keywords."""
        results = manager.find_similar(
            ["mouse", "seizure", "button"],
            top_k=20,
            use_embeddings=False,
        )
        assert len(results) > 0

        # Check variety of tags
        tags = {r.tag for r in results}
        # Should have tags from different categories
        has_animal = any(t in tags for t in ["Animal", "Animal-agent"])
        has_device = any(t in tags for t in ["Push-button", "Press"])
        has_clinical = any(r.prefix == "sc:" for r in results)

        # At least some should match
        assert has_animal or has_device or has_clinical

    def test_is_available_without_embeddings(self, manager):
        """is_available should be False when embeddings not loaded."""
        assert manager.is_available() is False

    def test_get_stats(self, manager):
        """get_stats should return stats dict."""
        stats = manager.get_stats()
        assert "tag_embeddings" in stats
        assert "keyword_index_size" in stats
        assert "loaded_files" in stats
        assert stats["keyword_index_size"] == len(KEYWORD_INDEX)


class TestGlobalManager:
    """Tests for global manager instance."""

    def test_get_semantic_search_manager(self):
        """Should return singleton instance."""
        manager1 = get_semantic_search_manager()
        manager2 = get_semantic_search_manager()
        assert manager1 is manager2


class TestModularLoading:
    """Tests for modular embeddings loading."""

    def test_load_from_directory(self, tmp_path):
        """Should load all embeddings-*.json files from directory."""
        import json

        # Create test embeddings files
        base_data = {
            "version": "1.0.0",
            "model_id": "test-model",
            "schema": "8.4.0",
            "type": "tags",
            "dimensions": 4,
            "count": 2,
            "embeddings": [
                {
                    "tag": "Animal",
                    "long_form": "Animal",
                    "prefix": "",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                },
                {
                    "tag": "Event",
                    "long_form": "Event",
                    "prefix": "",
                    "vector": [0.5, 0.6, 0.7, 0.8],
                },
            ],
        }

        lib_data = {
            "version": "1.0.0",
            "model_id": "test-model",
            "schema": "sc:score_2.1.0",
            "type": "tags",
            "dimensions": 4,
            "count": 1,
            "embeddings": [
                {
                    "tag": "Seizure",
                    "long_form": "sc:Seizure",
                    "prefix": "sc:",
                    "vector": [0.9, 0.8, 0.7, 0.6],
                },
            ],
        }

        kw_data = {
            "version": "1.0.0",
            "model_id": "test-model",
            "schema": "keywords",
            "type": "keywords",
            "dimensions": 4,
            "count": 1,
            "embeddings": [
                {"keyword": "mouse", "targets": ["Animal"], "vector": [0.2, 0.3, 0.4, 0.5]},
            ],
        }

        # Write files
        (tmp_path / "embeddings-base-8.4.0.json").write_text(json.dumps(base_data))
        (tmp_path / "embeddings-sc-score_2.1.0.json").write_text(json.dumps(lib_data))
        (tmp_path / "embeddings-keywords.json").write_text(json.dumps(kw_data))

        # Load from directory
        manager = SemanticSearchManager()
        result = manager.load_embeddings(tmp_path)

        assert result is True
        assert manager.is_available() is True
        stats = manager.get_stats()
        assert stats["tag_embeddings"] == 3  # 2 base + 1 library
        assert stats["keyword_embeddings"] == 1
        assert len(stats["loaded_files"]) == 3


class TestTagMatch:
    """Tests for TagMatch dataclass."""

    def test_tag_match_creation(self):
        """Should create TagMatch with all fields."""
        match = TagMatch(
            tag="Animal",
            long_form="Animal",
            prefix="",
            score=0.95,
            source="keyword",
        )
        assert match.tag == "Animal"
        assert match.score == 0.95
        assert match.source == "keyword"

    def test_tag_match_repr(self):
        """TagMatch repr should include tag and score."""
        match = TagMatch(
            tag="Seizure",
            long_form="Seizure",
            prefix="sc:",
            score=0.85,
            source="embedding",
        )
        repr_str = repr(match)
        assert "sc:Seizure" in repr_str
        assert "0.85" in repr_str
