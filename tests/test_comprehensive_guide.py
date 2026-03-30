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

    def test_guide_has_semantic_hints_pointer(self):
        """Test guide includes a pointer to check user message for semantic hints."""
        vocabulary = ["Event", "Reward", "Animal-agent"]
        extendable_tags = ["Label"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        # System prompt should point to user message for hints (not contain them)
        assert "SEMANTIC HINTS" in guide
        assert "user message" in guide.lower()

    def test_guide_no_extend_with_hints_pointer(self):
        """Test guide with no_extend has both hints pointer and extension warning."""
        vocabulary = ["Event", "Visual-presentation"]
        extendable_tags = ["Label"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags, no_extend=True)

        assert "SEMANTIC HINTS" in guide
        assert "EXTENSIONS STRICTLY PROHIBITED" in guide
        assert "(Extensions disabled)" in guide
