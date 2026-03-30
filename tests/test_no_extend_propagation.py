"""Tests for no_extend flag propagation through API models and state.

These tests verify that the no_extend parameter is properly defined
in both text and image annotation request models, and that it
propagates correctly through create_initial_state.
"""

from src.agents.state import create_initial_state
from src.api.models import AnnotationRequest, ImageAnnotationRequest


class TestAnnotationRequestNoExtend:
    """Tests for no_extend field on AnnotationRequest."""

    def test_defaults_to_false(self):
        """no_extend should default to False."""
        req = AnnotationRequest(description="A red circle appears")
        assert req.no_extend is False

    def test_can_be_set_to_true(self):
        """no_extend should accept True."""
        req = AnnotationRequest(description="A red circle appears", no_extend=True)
        assert req.no_extend is True

    def test_serialization_includes_no_extend(self):
        """Serialized model should include no_extend field."""
        req = AnnotationRequest(description="test", no_extend=True)
        data = req.model_dump()
        assert "no_extend" in data
        assert data["no_extend"] is True


class TestImageAnnotationRequestNoExtend:
    """Tests for no_extend field on ImageAnnotationRequest."""

    def test_defaults_to_false(self):
        """no_extend should default to False."""
        req = ImageAnnotationRequest(image="data:image/png;base64,abc123")
        assert req.no_extend is False

    def test_can_be_set_to_true(self):
        """no_extend should accept True."""
        req = ImageAnnotationRequest(
            image="data:image/png;base64,abc123",
            no_extend=True,
        )
        assert req.no_extend is True

    def test_serialization_includes_no_extend(self):
        """Serialized model should include no_extend field."""
        req = ImageAnnotationRequest(
            image="data:image/png;base64,abc123",
            no_extend=True,
        )
        data = req.model_dump()
        assert "no_extend" in data
        assert data["no_extend"] is True


class TestCreateInitialStateNoExtend:
    """Tests for no_extend propagation through create_initial_state."""

    def test_no_extend_false_by_default(self):
        """State should have no_extend=False by default."""
        state = create_initial_state("test description")
        assert state["no_extend"] is False

    def test_no_extend_true_propagates(self):
        """State should reflect no_extend=True when passed."""
        state = create_initial_state("test description", no_extend=True)
        assert state["no_extend"] is True

    def test_no_extend_with_other_params(self):
        """no_extend should work alongside other parameters."""
        state = create_initial_state(
            "A visual stimulus",
            schema_version="8.3.0",
            max_validation_attempts=5,
            run_assessment=True,
            no_extend=True,
        )
        assert state["no_extend"] is True
        assert state["schema_version"] == "8.3.0"
        assert state["max_validation_attempts"] == 5
        assert state["run_assessment"] is True
