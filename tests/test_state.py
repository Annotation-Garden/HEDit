"""Tests for LangGraph state."""

from src.agents.state import create_initial_state


def test_create_initial_state():
    """Test creating initial annotation state."""
    description = "A red circle appears on screen"
    state = create_initial_state(description)

    assert state["input_description"] == description
    assert state["current_annotation"] == ""
    assert state["validation_status"] == "pending"
    assert state["validation_attempts"] == 0
    assert state["is_valid"] is False
    assert state["is_faithful"] is False
    assert state["is_complete"] is False
    assert state["max_validation_attempts"] == 3
    assert state["schema_version"] == "8.4.0"
    assert state["no_extend"] is False
    assert state["tag_suggestions"] == {}


def test_create_initial_state_custom_params():
    """Test creating initial state with custom parameters."""
    description = "Test event"
    state = create_initial_state(
        description,
        schema_version="8.3.0",
        max_validation_attempts=3,
    )

    assert state["schema_version"] == "8.3.0"
    assert state["max_validation_attempts"] == 3


def test_create_initial_state_no_extend():
    """Test creating initial state with no_extend=True."""
    description = "A reward is delivered to the animal"
    state = create_initial_state(
        description,
        no_extend=True,
    )

    assert state["no_extend"] is True
    assert state["input_description"] == description
    assert state["schema_version"] == "8.4.0"


def test_create_initial_state_semantic_hints():
    """Test creating initial state with semantic hints."""
    description = "A visual stimulus appears"
    semantic_hints = [
        {"tag": "Visual-presentation", "score": 0.9, "source": "keyword"},
    ]
    state = create_initial_state(
        description,
        semantic_hints=semantic_hints,
    )

    assert state["semantic_hints"] == semantic_hints
    assert len(state["semantic_hints"]) == 1
    assert state["semantic_hints"][0]["tag"] == "Visual-presentation"
