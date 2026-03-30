"""Tests for keyword extraction and semantic preprocessing in the workflow.

These tests verify:
- LLM-based keyword extraction from descriptions
- Integration of keyword extraction with hed-lsp tag suggestions
- Graceful degradation when hed-lsp is not available
- Deduplication and scoring of semantic hints
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.workflow import HedAnnotationWorkflow
from src.validation.hed_lsp import HedSuggestion, HedSuggestResult


def _make_workflow(enable_semantic_search: bool = True) -> HedAnnotationWorkflow:
    """Create a workflow instance with mocked LLMs for testing.

    Args:
        enable_semantic_search: Whether to enable the semantic preprocessing node

    Returns:
        HedAnnotationWorkflow with mocked dependencies
    """
    mock_llm = MagicMock()
    with patch("src.validation.hed_lsp.is_hed_lsp_available", return_value=False):
        wf = HedAnnotationWorkflow(
            llm=mock_llm,
            feedback_llm=mock_llm,
            enable_semantic_search=enable_semantic_search,
            use_js_validator=False,
        )
    return wf


class TestExtractKeywords:
    """Tests for the _extract_keywords method."""

    @pytest.mark.asyncio
    async def test_extracts_keywords_from_short_description(self):
        """Should extract keywords from a simple event description."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = "button, press, left hand, response"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        keywords = await wf._extract_keywords("A participant presses a button with their left hand")

        assert len(keywords) == 4
        assert "button" in keywords
        assert "press" in keywords
        assert "left hand" in keywords
        assert "response" in keywords

    @pytest.mark.asyncio
    async def test_extracts_keywords_from_long_description(self):
        """Should extract keywords from a complex, multi-sentence description."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = (
            "person, car, road, red light, stopping, urban, traffic signal, visual, daytime"
        )
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        description = (
            "The image shows a person driving a car on a busy urban road. "
            "A red traffic light is visible ahead and the car is stopping. "
            "The scene takes place during daytime with clear visibility."
        )
        keywords = await wf._extract_keywords(description)

        assert len(keywords) == 9
        assert "person" in keywords
        assert "red light" in keywords

    @pytest.mark.asyncio
    async def test_limits_to_20_keywords(self):
        """Should limit output to 20 keywords even if LLM returns more."""
        wf = _make_workflow()

        # Return 25 keywords
        many_keywords = ", ".join([f"keyword{i}" for i in range(25)])
        mock_response = MagicMock()
        mock_response.content = many_keywords
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        keywords = await wf._extract_keywords("A very complex scene with many elements")
        assert len(keywords) == 20

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        """Should return empty list when LLM invocation fails."""
        wf = _make_workflow()
        wf.feedback_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))

        keywords = await wf._extract_keywords("Some description")
        assert keywords == []

    @pytest.mark.asyncio
    async def test_handles_structured_response_content(self):
        """Should handle structured (list of dicts) response content from LLMs."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = [
            {"type": "thinking", "thinking": "Let me analyze..."},
            {"type": "text", "text": "screen, flash, visual, onset"},
        ]
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        keywords = await wf._extract_keywords("A flash appears on screen")
        assert len(keywords) == 4
        assert "screen" in keywords
        assert "flash" in keywords

    @pytest.mark.asyncio
    async def test_filters_empty_keywords(self):
        """Should filter out empty strings from parsed keywords."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = "button, , press, , response"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        keywords = await wf._extract_keywords("Button press response")
        assert "" not in keywords
        assert len(keywords) == 3


class TestSemanticPreprocessNode:
    """Tests for the _semantic_preprocess_node method."""

    @pytest.mark.asyncio
    async def test_skips_on_subsequent_iterations(self):
        """Should skip preprocessing after the first iteration."""
        wf = _make_workflow()

        state = {"total_iterations": 1, "input_description": "test"}
        result = await wf._semantic_preprocess_node(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_extracts_keywords_without_lsp(self):
        """Should extract keywords even when hed-lsp is not available."""
        wf = _make_workflow()
        assert wf.hed_lsp_client is None  # LSP not available

        mock_response = MagicMock()
        mock_response.content = "circle, red, screen, appearing, visual"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        state = {"total_iterations": 0, "input_description": "A red circle appears on screen"}
        result = await wf._semantic_preprocess_node(state)

        assert "extracted_keywords" in result
        assert len(result["extracted_keywords"]) == 5
        assert "circle" in result["extracted_keywords"]
        assert "red" in result["extracted_keywords"]
        # Without LSP, semantic_hints should be empty
        assert result["semantic_hints"] == []

    @pytest.mark.asyncio
    async def test_enriches_keywords_with_lsp(self):
        """Should use hed-lsp to get tag suggestions for extracted keywords."""
        wf = _make_workflow()

        # Mock keyword extraction
        mock_response = MagicMock()
        mock_response.content = "button, press"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Mock LSP client
        mock_lsp = MagicMock()

        def suggest_side_effect(keyword):
            if keyword == "button":
                return HedSuggestResult(
                    success=True,
                    suggestions=[
                        HedSuggestion(
                            tag="Item/Object/Man-made-object/Device/IO-device/Input-device/Response-button",
                            score=0.9,
                        ),
                        HedSuggestion(tag="Action/Move/Press", score=0.3),
                    ],
                )
            elif keyword == "press":
                return HedSuggestResult(
                    success=True,
                    suggestions=[
                        HedSuggestion(tag="Action/Move/Press", score=0.8),
                    ],
                )
            return HedSuggestResult(success=False, suggestions=[], error="Unknown")

        mock_lsp.suggest = suggest_side_effect
        wf.hed_lsp_client = mock_lsp

        state = {"total_iterations": 0, "input_description": "A button press event"}
        result = await wf._semantic_preprocess_node(state)

        assert len(result["extracted_keywords"]) == 2
        # Should have deduplicated tags; "Action/Move/Press" appeared for both keywords
        tags = [h["tag"] for h in result["semantic_hints"]]
        assert "Action/Move/Press" in tags
        # The deduplicated Press tag should have the highest score (0.8 from "press")
        press_hint = next(h for h in result["semantic_hints"] if h["tag"] == "Action/Move/Press")
        assert press_hint["score"] == 0.8

    @pytest.mark.asyncio
    async def test_deduplicates_by_highest_score(self):
        """Should keep the highest-scoring entry when deduplicating tags."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = "word1, word2"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_lsp = MagicMock()

        def suggest_side_effect(keyword):
            if keyword == "word1":
                return HedSuggestResult(
                    success=True,
                    suggestions=[HedSuggestion(tag="Event/Sensory-event", score=0.5)],
                )
            elif keyword == "word2":
                return HedSuggestResult(
                    success=True,
                    suggestions=[HedSuggestion(tag="Event/Sensory-event", score=0.9)],
                )
            return HedSuggestResult(success=False, suggestions=[], error="Unknown")

        mock_lsp.suggest = suggest_side_effect
        wf.hed_lsp_client = mock_lsp

        state = {"total_iterations": 0, "input_description": "test"}
        result = await wf._semantic_preprocess_node(state)

        # Should have only one entry for "Event/Sensory-event" with score 0.9
        assert len(result["semantic_hints"]) == 1
        assert result["semantic_hints"][0]["tag"] == "Event/Sensory-event"
        assert result["semantic_hints"][0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_handles_lsp_failure_for_some_keywords(self):
        """Should handle LSP failures for individual keywords gracefully."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = "valid_word, invalid_word"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_lsp = MagicMock()

        def suggest_side_effect(keyword):
            if keyword == "valid_word":
                return HedSuggestResult(
                    success=True,
                    suggestions=[HedSuggestion(tag="Event", score=0.7)],
                )
            return HedSuggestResult(success=False, suggestions=[], error="Not found")

        mock_lsp.suggest = suggest_side_effect
        wf.hed_lsp_client = mock_lsp

        state = {"total_iterations": 0, "input_description": "test description"}
        result = await wf._semantic_preprocess_node(state)

        # Should still return hints from the successful keyword
        assert len(result["semantic_hints"]) == 1
        assert result["semantic_hints"][0]["tag"] == "Event"

    @pytest.mark.asyncio
    async def test_handles_lsp_exception_gracefully(self):
        """Should handle unexpected LSP exceptions without crashing."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = "button"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_lsp = MagicMock()
        mock_lsp.suggest = MagicMock(side_effect=RuntimeError("Unexpected error"))
        wf.hed_lsp_client = mock_lsp

        state = {"total_iterations": 0, "input_description": "A button press"}
        result = await wf._semantic_preprocess_node(state)

        # Should return keywords but empty hints
        assert result["extracted_keywords"] == ["button"]
        assert result["semantic_hints"] == []

    @pytest.mark.asyncio
    async def test_sorts_hints_by_score_descending(self):
        """Should sort semantic hints by score in descending order."""
        wf = _make_workflow()

        mock_response = MagicMock()
        mock_response.content = "a, b"
        wf.feedback_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_lsp = MagicMock()

        def suggest_side_effect(keyword):
            if keyword == "a":
                return HedSuggestResult(
                    success=True,
                    suggestions=[HedSuggestion(tag="Low-score-tag", score=0.2)],
                )
            elif keyword == "b":
                return HedSuggestResult(
                    success=True,
                    suggestions=[HedSuggestion(tag="High-score-tag", score=0.95)],
                )
            return HedSuggestResult(success=False, suggestions=[], error="Unknown")

        mock_lsp.suggest = suggest_side_effect
        wf.hed_lsp_client = mock_lsp

        state = {"total_iterations": 0, "input_description": "test"}
        result = await wf._semantic_preprocess_node(state)

        assert len(result["semantic_hints"]) == 2
        assert result["semantic_hints"][0]["tag"] == "High-score-tag"
        assert result["semantic_hints"][1]["tag"] == "Low-score-tag"


class TestWorkflowSemanticSearchConfig:
    """Tests for workflow configuration of semantic search."""

    def test_semantic_search_enabled_by_default(self):
        """Should have semantic search enabled by default even without LSP."""
        wf = _make_workflow(enable_semantic_search=True)
        assert wf.enable_semantic_search is True
        # LSP client should be None since we mocked it as unavailable
        assert wf.hed_lsp_client is None

    def test_semantic_search_disabled(self):
        """Should disable semantic preprocessing when explicitly set to False."""
        wf = _make_workflow(enable_semantic_search=False)
        assert wf.enable_semantic_search is False

    def test_feedback_llm_stored(self):
        """Should store the feedback LLM for keyword extraction."""
        mock_llm = MagicMock()
        mock_feedback_llm = MagicMock()
        with patch("src.validation.hed_lsp.is_hed_lsp_available", return_value=False):
            wf = HedAnnotationWorkflow(
                llm=mock_llm,
                feedback_llm=mock_feedback_llm,
                use_js_validator=False,
            )
        assert wf.feedback_llm is mock_feedback_llm

    def test_feedback_llm_defaults_to_main_llm(self):
        """Should default feedback_llm to main llm when not provided."""
        mock_llm = MagicMock()
        with patch("src.validation.hed_lsp.is_hed_lsp_available", return_value=False):
            wf = HedAnnotationWorkflow(
                llm=mock_llm,
                use_js_validator=False,
            )
        assert wf.feedback_llm is mock_llm
