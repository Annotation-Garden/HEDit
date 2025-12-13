"""Comprehensive HED annotation guide for LLMs.

This module contains a complete guide to HED annotation creation,
consolidated from multiple HED resources and documentation including
HedAnnotationSemantics.md for proper semantic annotation rules.
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

    return f"""# HED ANNOTATION GUIDE

## CRITICAL RULE: CHECK VOCABULARY FIRST

BEFORE using ANY tag with a slash (/), CHECK if it's in the vocabulary below!

WRONG: Item/Window, Item/Plant, Property/Red, Action/Press
RIGHT: Window, Plant, Red, Press (if these are in vocabulary)

The slash (/) is ONLY for:
1. NEW tags NOT in vocabulary: Building/Cottage (only if "Cottage" NOT in vocab)
2. Values with units: Duration/2 s, Frequency/440 Hz
3. Definitions: Definition/MyDef, Def/MyDef

IF YOU SEE TAG_EXTENSION_INVALID ERROR -> You extended a tag that exists in vocabulary!

---

## SEMANTIC GROUPING RULES

A well-formed HED annotation can be translated back into coherent English.
This reversibility principle is the fundamental validation test for HED semantics.

### Rule 1: Group object properties together
Tags describing properties of the SAME object MUST be grouped.

CORRECT: (Red, Circle) - A single object that is red AND circular
WRONG: Red, Circle - Ambiguous; could be two different things

### Rule 2: Nest agent-action-object relationships
Agent-action, ((Agent-tags), (Action-tag, (Object-tags)))

EXAMPLE: Agent-action, Participant-response, ((Human-agent, Experiment-participant), (Press, (Left, Mouse-button)))
MEANING: "The experiment participant presses the left mouse button"

### Rule 3: Use directional pattern for relationships
Pattern: (A, (Relation-tag, C))
MEANING: "A has the relationship to C"

EXAMPLE: ((Red, Circle), (To-left-of, (Green, Square)))
MEANING: "A red circle is to the left of a green square"

### Rule 4: Group Event and Task-event-role at top level
Event classification tags (Sensory-event, Agent-action) and Task-event-role tags
(Experimental-stimulus, Participant-response) should be at the top level.

EXAMPLE: Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)

### Rule 5: Sensory-event should have Sensory-modality
If the event is a Sensory-event, include Visual-presentation, Auditory-presentation, etc.

EXAMPLE: Sensory-event, Visual-presentation, (Red, Circle)

### Rule 6: Keep independent concepts separate
Do NOT group unrelated things together.

WRONG: (Red, Press) - Color and action are unrelated
WRONG: (Triangle, Mouse-button) - Stimulus shape and response device unrelated

---

## EXTENSION RULES (TAG_EXTENDED Warning)

When you MUST extend (concept not in vocabulary), extend from the MOST SPECIFIC
applicable parent tag while preserving the is-a (taxonomic) relationship.

### WRONG: Extending from overly general parents
- Item/Cottage (too general; Cottage is-a Building, not just Item)
- Action/Cartwheel (too general; Cartwheel is-a Move-body action)
- Object/Rickshaw (too general; Rickshaw is-a Vehicle)
- Agent/Dolphin (too general; Dolphin is-a Mammal)
- Item/Armoire (too general; Armoire is-a Furniture)

### CORRECT: Extending from most specific parents
- Building/Cottage (Cottage is-a Building - correct taxonomy)
- Move-body/Cartwheel (Cartwheel is-a body movement)
- Vehicle/Rickshaw (Rickshaw is-a vehicle)
- Mammal/Dolphin (Dolphin is-a mammal, more specific than Animal)
- Furniture/Armoire (Armoire is-a furniture)

### Extension Decision Process
1. Concept not in vocabulary? Must extend.
2. Find the schema path to similar concepts.
3. Extend from the DEEPEST (most specific) parent that maintains is-a relationship.
4. The extended tag should logically "be a type of" its parent.

### Cannot Extend These
- Event subtree (Sensory-event, Agent-action, etc.) - use existing event types
- Agent subtree - use existing agent types
- Value-taking nodes (tags with # child) - cannot extend after #

---

## DEFINITION SYSTEM

Definitions allow naming reusable annotation patterns.

### Creating Definitions (in sidecars only)
Pattern: (Definition/Name, (tag1, tag2, tag3))
With placeholder: (Definition/Name/#, (Tag1/# units, Tag2))

EXAMPLE: (Definition/RedCircle, (Sensory-event, Visual-presentation, (Red, Circle)))
EXAMPLE: (Definition/Acc/#, (Acceleration/# m-per-s^2, Red))

### Using Definitions with Def
Pattern: Def/Name or Def/Name/value (if definition has placeholder)

EXAMPLE: Def/RedCircle
EXAMPLE: Def/Acc/4.5

### Def-expand (DO NOT USE)
Def-expand is created by tools during processing. Never use it manually.

### Definition Rules
- Definitions can only appear in sidecars or external files
- Cannot contain Def, Def-expand, or nested Definition
- If using #, must have exactly two # characters
- Definition names must be unique

---

## TEMPORAL SCOPING (Onset/Offset/Duration)

### Using Duration (simpler)
Pattern: (Duration/value units, (event-content))

EXAMPLE: (Duration/2 s, (Sensory-event, Visual-presentation, Cue, (Cross)))
MEANING: A cross cue is displayed for 2 seconds

### Using Onset/Offset (for explicit start/end markers)
Requires a Definition anchor.

START: (Def/Event, Onset)
END: (Def/Event, Offset)

EXAMPLE:
  Start: (Def/Fixation-point, Onset)
  End: (Def/Fixation-point, Offset)

---

## SIDECAR SYNTAX (events.json)

### Value Placeholders (#)
For columns with varying values, use # as placeholder.

EXAMPLE: {{"age": {{"HED": "Age/# years"}}}}
For age=25: assembles to "Age/25 years"

### Column References (curly braces)
Reference other columns to assemble grouped annotations.

EXAMPLE:
{{
  "event_type": {{
    "HED": {{
      "visual": "Experimental-stimulus, Sensory-event, Visual-presentation, ({{color}}, {{shape}})"
    }}
  }},
  "color": {{"HED": {{"red": "Red", "blue": "Blue"}}}},
  "shape": {{"HED": {{"circle": "Circle", "square": "Square"}}}}
}}

For event_type=visual, color=red, shape=circle:
ASSEMBLES TO: Experimental-stimulus, Sensory-event, Visual-presentation, (Red, Circle)

### Curly Brace Rules
- Only valid in sidecars (not in event file HED column directly)
- Must reference existing columns with HED annotations
- No circular references (A references B, B references A)
- Use for grouping related properties from different columns

---

## EVENT AND TASK-EVENT-ROLE CLASSIFICATION

### Event Types (from Event subtree)
- Sensory-event: Something presented to senses
- Agent-action: An agent performs an action
- Data-feature: Computed or observed feature
- Experiment-control: Structural/control change
- Experiment-structure: Experiment organization marker
- Measurement-event: Measurement taken

### Task-Event-Role Tags (from Task-event-role subtree)
- Experimental-stimulus: Primary stimulus to respond to
- Cue: Signal about what to expect or do
- Participant-response: Action by participant
- Feedback: Performance information
- Instructional: Task instructions
- Warning: Alert signal
- Incidental: Present but not task-relevant

### When to Use Both
For task-related events, include BOTH Event type AND Task-event-role.

EXAMPLE: Sensory-event, Experimental-stimulus, Auditory-presentation, (Tone, Frequency/440 Hz)
MEANING: An auditory tone that is the experimental stimulus

---

## TAG USAGE BY CATEGORY

### ITEMS (objects, things)
IN VOCABULARY -> Use as-is: Window, Plant, Circle, Square, Button, Triangle

NOT IN VOCABULARY -> Extend from MOST SPECIFIC parent:
- Building/Cottage (not Item/Cottage or Object/Cottage)
- Furniture/Armoire (not Item/Armoire or Furnishing/Armoire)
- Vehicle/Rickshaw (not Item/Rickshaw or Object/Rickshaw)
- Mammal/Dolphin (not Agent/Dolphin or Animal/Dolphin)

### PROPERTIES (colors, attributes)
IN VOCABULARY -> Use as-is: Red, Blue, Green, Large

NOT IN VOCABULARY -> Extend from MOST SPECIFIC parent:
- Blue-green/Turquoise (from specific color category)
- Size/Gigantic

### ACTIONS
IN VOCABULARY -> Use as-is: Press, Move, Click

NOT IN VOCABULARY -> Extend from MOST SPECIFIC parent:
- Move-body/Cartwheel (not Action/Cartwheel)
- Move-fingers/Squeeze (not Action/Squeeze)
- Move-upper-extremity/Swipe (not Action/Swipe)

---

## COMMON PATTERNS

### Visual stimulus
Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)

### Participant response
Agent-action, Participant-response, ((Human-agent, Experiment-participant), (Press, (Left, Mouse-button)))

### Spatial relationship
Sensory-event, Visual-presentation, ((Red, Circle), (To-left-of, (Green, Square)))

### Multiple objects in same event
Sensory-event, Visual-presentation, (Blue, Square), (Yellow, Triangle)

### Feedback event
Sensory-event, Visual-presentation, (Feedback, Positive), (Green, Circle)

### Cue with duration
(Duration/1.5 s, (Sensory-event, Visual-presentation, Cue, (Cross)))

---

## VOCABULARY LOOKUP

ALWAYS check this list before using any tag. Use tags EXACTLY as shown.

{vocab_str}

CRITICAL:
- If "Press" is in this list -> use "Press" NOT "Action/Press"
- If "Button" is in this list -> use "Button" NOT "Item/Button"
- If "Circle" is in this list -> use "Circle" NOT "Item/Circle"
- If "Red" is in this list -> use "Red" NOT "Property/Red"

---

## EXTENDABLE TAGS

Only extend if the concept is NOT in vocabulary above.
When extending, use the MOST SPECIFIC applicable parent.

{extend_str}

---

## OUTPUT FORMAT

Output ONLY the HED annotation string.
NO explanations, NO markdown, NO code blocks, NO commentary.
Just the raw HED annotation.
"""
