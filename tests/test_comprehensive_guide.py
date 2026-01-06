"""Tests for HED comprehensive guide generation."""

from src.utils.hed_comprehensive_guide import get_comprehensive_hed_guide


class TestComprehensiveGuide:
    """Tests for comprehensive HED guide generation."""

    def test_guide_basic_generation(self):
        """Test basic guide generation."""
        vocabulary = ["Event", "Sensory-event", "Visual-presentation"]
        extendable_tags = ["Label", "Description"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        assert "## CRITICAL RULE: CHECK VOCABULARY FIRST" in guide
        assert "Event" in guide
        assert "Sensory-event" in guide

    def test_guide_with_no_extend_false(self):
        """Test guide generation with no_extend=False (default)."""
        vocabulary = ["Event", "Agent-action", "Animal-agent"]
        extendable_tags = ["Label", "Description"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags, no_extend=False)

        # Should NOT contain the no-extend warning
        assert "EXTENSIONS STRICTLY PROHIBITED" not in guide
        assert "(Extensions disabled)" not in guide
        # Extendable tags should be shown normally
        assert "Label" in guide
        assert "Description" in guide

    def test_guide_with_no_extend_true(self):
        """Test guide generation with no_extend=True."""
        vocabulary = ["Event", "Agent-action", "Animal-agent"]
        extendable_tags = ["Label", "Description"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags, no_extend=True)

        # Should contain the no-extend warning section
        assert "EXTENSIONS STRICTLY PROHIBITED" in guide
        assert "MUST NOT create any new tags" in guide
        assert "What is FORBIDDEN" in guide
        # Should show extensions as disabled
        assert "(Extensions disabled)" in guide

    def test_guide_with_semantic_hints(self):
        """Test guide generation with semantic hints."""
        vocabulary = ["Event", "Reward", "Animal-agent"]
        extendable_tags = ["Label"]
        semantic_hints = [
            {"tag": "Reward", "prefix": "", "score": 0.95, "source": "keyword"},
            {"tag": "Animal-agent", "prefix": "", "score": 0.85, "source": "embedding"},
        ]

        guide = get_comprehensive_hed_guide(
            vocabulary, extendable_tags, semantic_hints=semantic_hints
        )

        assert "POTENTIALLY RELEVANT TAGS" in guide
        assert "Reward" in guide
        assert "Animal-agent" in guide
        # Check confidence indicators
        assert "high" in guide.lower() or "0.95" in guide

    def test_guide_with_semantic_hints_and_no_extend(self):
        """Test guide with both semantic hints and no_extend."""
        vocabulary = ["Event", "Visual-presentation"]
        extendable_tags = ["Label"]
        semantic_hints = [
            {"tag": "Visual-presentation", "prefix": "", "score": 0.9, "source": "keyword"},
        ]

        guide = get_comprehensive_hed_guide(
            vocabulary, extendable_tags, semantic_hints=semantic_hints, no_extend=True
        )

        # Should have both features
        assert "POTENTIALLY RELEVANT TAGS" in guide
        assert "EXTENSIONS STRICTLY PROHIBITED" in guide
        assert "(Extensions disabled)" in guide
