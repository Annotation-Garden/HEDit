"""Tests for HED comprehensive guide generation."""

from unittest.mock import patch

from src.utils.hed_comprehensive_guide import get_comprehensive_hed_guide


class TestComprehensiveGuide:
    """Tests for comprehensive HED guide generation."""

    def test_guide_basic_generation(self):
        """Test basic guide generation includes key sections."""
        vocabulary = ["Event", "Sensory-event", "Visual-presentation"]
        extendable_tags = ["Label", "Description"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        assert "## CRITICAL RULE: CHECK VOCABULARY FIRST" in guide
        assert "Event" in guide
        assert "Sensory-event" in guide

    def test_guide_includes_official_docs(self):
        """Test guide includes official HED documentation."""
        vocabulary = ["Event", "Sensory-event"]
        extendable_tags = ["Label"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        assert "## OFFICIAL HED ANNOTATION SEMANTICS" in guide
        assert "## HED TERMINOLOGY" in guide
        # Key content from official docs
        assert "reversibility" in guide.lower()

    def test_guide_includes_hedit_specific_sections(self):
        """Test guide preserves all HEDit-specific sections."""
        vocabulary = ["Event", "Red", "Circle"]
        extendable_tags = ["Label"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        assert "## CRITICAL RULE: CHECK VOCABULARY FIRST" in guide
        assert "## CORRECTION WORKFLOW" in guide
        assert "## SEMANTIC HINTS" in guide
        assert "## VOCABULARY LOOKUP" in guide
        assert "## EXTENDABLE TAGS" in guide
        assert "## COMMON ERRORS AND TROUBLESHOOTING" in guide
        assert "## OUTPUT FORMAT" in guide

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

    def test_guide_vocabulary_injected(self):
        """Test vocabulary and extendable tags are injected into the guide."""
        vocabulary = ["MyCustomTag1", "MyCustomTag2", "MyCustomTag3"]
        extendable_tags = ["ExtendableParent"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        assert "MyCustomTag1" in guide
        assert "MyCustomTag2" in guide
        assert "MyCustomTag3" in guide
        assert "ExtendableParent" in guide

    def test_guide_old_sections_removed(self):
        """Test that hand-written sections replaced by official docs are gone."""
        vocabulary = ["Event"]
        extendable_tags = ["Label"]

        guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        # These old section headers should no longer appear
        assert "## SEMANTIC GROUPING RULES" not in guide
        assert "## RELATION TAGS" not in guide
        assert "## CRITICAL: EVENT AND AGENT SUBTREES" not in guide
        assert "## EXTENSION RULES (for extendable tags)" not in guide
        assert "## DEFINITION SYSTEM" not in guide
        assert "## TEMPORAL SCOPING" not in guide
        assert "## SIDECAR SYNTAX" not in guide
        assert "## EVENT AND TASK-EVENT-ROLE CLASSIFICATION" not in guide
        assert "## TAG USAGE BY CATEGORY" not in guide
        assert "## COMMON PATTERNS" not in guide

    def test_guide_fallback_when_docs_missing(self):
        """Test guide still works when official docs are unavailable."""
        vocabulary = ["Event", "Sensory-event"]
        extendable_tags = ["Label"]

        with patch("src.utils.hed_comprehensive_guide.load_hed_docs", return_value={}):
            guide = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        # HEDit-specific sections should still be present
        assert "## CRITICAL RULE: CHECK VOCABULARY FIRST" in guide
        assert "## CORRECTION WORKFLOW" in guide
        assert "## COMMON ERRORS AND TROUBLESHOOTING" in guide
        assert "## OUTPUT FORMAT" in guide
        # Official docs sections should be absent
        assert "## OFFICIAL HED ANNOTATION SEMANTICS" not in guide

    def test_guide_is_deterministic(self):
        """Test that same inputs produce same output (for prompt caching)."""
        vocabulary = ["Event", "Sensory-event", "Red"]
        extendable_tags = ["Label"]

        guide1 = get_comprehensive_hed_guide(vocabulary, extendable_tags)
        guide2 = get_comprehensive_hed_guide(vocabulary, extendable_tags)

        assert guide1 == guide2
