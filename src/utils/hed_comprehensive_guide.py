"""Comprehensive HED annotation guide for LLMs.

This module contains a complete guide to HED annotation creation,
consolidatedFrom multiple HED resources and documentation.
"""

def get_comprehensive_hed_guide(vocabulary_sample: list[str], extendable_tags: list[str]) -> str:
    """Generate comprehensive HED annotation guide.

    Args:
        vocabulary_sample: Full list of valid HED tags (complete vocabulary)
        extendable_tags: Tags that allow extension

    Returns:
        Complete HED annotation guide
    """
    # Provide FULL vocabulary (not just first 100)
    vocab_str = ", ".join(vocabulary_sample)
    extend_str = ", ".join(extendable_tags)

    return f"""🚨 CRITICAL: CHECK VOCABULARY FIRST! NEVER EXTEND EXISTING TAGS! 🚨

BEFORE using ANY tag with a slash (/), CHECK if it's in the vocabulary below!

❌ WRONG: Item/Window, Item/Plant, Property/Red, Action/Press
✓ RIGHT: Window, Plant, Red, Press (if these are in vocabulary)

The slash (/) is ONLY for:
1. NEW tags NOT in vocabulary: Item/Spaceship (only if "Spaceship" NOT in vocab list below)
2. Values with units: Duration/2 s, Frequency/440 Hz
3. Definitions: Definition/MyDef

IF YOU SEE TAG_EXTENSION_INVALID ERROR → You extended a tag that exists in vocabulary!

# TAG USAGE RULES BY CATEGORY

## ITEMS (objects, things)
✓ IN VOCABULARY → Use as-is:
  - Window (NOT Item/Window) ← COMMON ERROR!
  - Plant (NOT Item/Plant) ← COMMON ERROR!
  - Circle (NOT Item/Circle)
  - Square (NOT Item/Square)
  - Button (NOT Item/Button)
  - Triangle (NOT Item/Triangle)
  - Screen (NOT Item/Screen)
  - Mouse (NOT Item/Mouse)

✗ NOT IN VOCABULARY → Use Item/NewName:
  - Item/Spaceship (if "Spaceship" not in vocab)
  - Item/Joystick (if "Joystick" not in vocab)
  - Item/Wall (if "Wall" not in vocab)
  - Item/Sofa (if "Sofa" not in vocab)

## PROPERTIES (colors, attributes, states)
✓ IN VOCABULARY → Use as-is:
  - Red (NOT Property/Red)
  - Blue (NOT Property/Blue)
  - Green (NOT Property/Green)
  - Large (NOT Property/Large)

✗ NOT IN VOCABULARY → Use Property/NewName:
  - Property/Turquoise (if "Turquoise" not in vocab)
  - Property/Gigantic (if "Gigantic" not in vocab)

## ACTIONS
✓ IN VOCABULARY → Use as-is:
  - Press (NOT Action/Press)
  - Move (NOT Action/Move)
  - Click (NOT Action/Click)

✗ NOT IN VOCABULARY → Use Action/NewName:
  - Action/Swipe (if "Swipe" not in vocab)
  - Action/Pinch (if "Pinch" not in vocab)

## AGENTS
✓ IN VOCABULARY → Use as-is:
  - Human-agent (NOT Agent/Human-agent)
  - Experiment-participant (NOT Agent/Experiment-participant)

✗ NOT IN VOCABULARY → Use Agent/NewName:
  - Agent/Robot (if "Robot" not in vocab)

IF YOU SEE TAG_EXTENSION_INVALID ERRORS, YOU ADDED A PATH TO AN EXISTING TAG!

# Required Tags

Every event annotation must have:
1. Event type: Sensory-event, Agent-action, Data-feature, etc.
2. Task role: Experimental-stimulus, Participant-response, Cue, etc.
3. If Sensory-event: add modality (Visual-presentation, Auditory-presentation, etc.)

# Grouping Rules

1. Group properties of SAME object: `(Red, Circle)` not `Red, Circle`
2. Agent-action pattern: `Agent-action, ((Agent-tags), (Action-tag, (Object-tags)))`
   Example: `Agent-action, ((Human-agent, Experiment-participant), (Press, (Left, Mouse-button)))`
3. Spatial relationships: `((Red, Circle), (To-left-of, (Green, Square)))`
4. Don't group unrelated things: `(Red, Press)` is WRONG

# Common Patterns (using ONLY vocab tags)

- Visual stimulus: `Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)`
  Tags used: Red, Circle (both IN vocabulary → use as-is)

- Participant response: `Agent-action, Participant-response, ((Human-agent, Experiment-participant), (Press, (Left, Mouse-button)))`
  Tags used: Human-agent, Experiment-participant, Press, Left, Mouse-button (all IN vocabulary → use as-is)

- Spatial relationship: `Sensory-event, Visual-presentation, ((Red, Circle), (To-left-of, (Green, Square)))`
  Tags used: Red, Circle, To-left-of, Green, Square (all IN vocabulary → use as-is)

- Multiple objects: `Sensory-event, Visual-presentation, (Blue, Square), (Yellow, Triangle)`
  Tags used: Blue, Square, Yellow, Triangle (all IN vocabulary → use as-is, NOT Property/Blue or Item/Square)

# STEP-BY-STEP: Before Using ANY Tag with a Slash (/)

1. LOOK UP the tag in the COMPLETE VOCABULARY below
2. IF FOUND → Use it EXACTLY as shown (no slash, no parent path)
3. IF NOT FOUND → Then and only then use extension (e.g., Item/NewTag)

Example Decision Process:
- Need to annotate "window"?
  → Check vocab → Found "Window" → Use "Window" (NOT "Item/Window")
- Need to annotate "plant"?
  → Check vocab → Found "Plant" → Use "Plant" (NOT "Item/Plant")
- Need to annotate "sofa"?
  → Check vocab → NOT found → Use "Item/Sofa" (extension allowed)

# Your COMPLETE Vocabulary

{vocab_str}

CRITICAL: Use these tags EXACTLY as shown - NO parent paths!
- If "Press" is in this list → use "Press" NOT "Action/Press"
- If "Button" is in this list → use "Button" NOT "Item/Button"
- If "Circle" is in this list → use "Circle" NOT "Item/Circle"
- If "Red" is in this list → use "Red" NOT "Property/Red"

# Extendable Tags (Complete List)

{extend_str}

Only extend if the tag is NOT in vocabulary above.
Example: Item/Spaceship (only if "Spaceship" not in vocabulary)

# Output Format

Output ONLY the HED annotation string - NO explanations, NO markdown, NO code blocks.
"""
