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

EXAMPLE: ((Dog), (Chases, (Cat)))
MEANING: "A dog chases a cat"

EXAMPLE: ((Participant), (Focuses-on, (Target)))
MEANING: "The participant focuses on the target"

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

## RELATION TAGS

Relation tags describe how entities relate to each other spatially, temporally, or logically.
Always use the pattern: (Entity-A, (Relation-tag, Entity-B))

### Spatial Relations
Describe where objects are positioned relative to each other.

COMMON SPATIAL RELATIONS:
- To-left-of, To-right-of - horizontal positioning
- Above, Below - vertical positioning
- In-front-of, Behind - depth positioning
- Inside-of, Outside-of - containment
- Near-to, Far-from - distance
- Center-of, Edge-of - position within

EXAMPLES:
((Red, Circle), (To-left-of, (Blue, Square)))
"Red circle is to the left of blue square"

((Face), (Center-of, (Screen)))
"Face is at the center of the screen"

((Target), (Inside-of, (Boundary)))
"Target is inside the boundary"

### Temporal Relations
Describe when events occur relative to each other.

COMMON TEMPORAL RELATIONS:
- Before, After - sequence
- During - simultaneity
- Starts-during, Ends-during - partial overlap

EXAMPLES:
((Cue), (Before, (Target)))
"Cue appears before target"

((Sound), (During, (Visual-presentation)))
"Sound occurs during visual presentation"

### Logical/Functional Relations
Describe functional relationships between entities.

COMMON LOGICAL RELATIONS:
- Associated-with - general association
- Part-of - component relationship
- Related-to - loose connection
- Indicated-by - signaling relationship
- Linked-to - connected entities

EXAMPLES:
((Response), (Associated-with, (Stimulus)))
"Response is associated with stimulus"

((Button), (Part-of, (Response-box)))
"Button is part of response box"

### Relation Pattern Rules
1. The subject comes first: (Subject, (Relation, Object))
2. Relations create directional links
3. Inverse relations may exist (Above/Below, Before/After)
4. Group related entities properly before applying relations
5. Relations can be nested for complex descriptions

COMPLEX EXAMPLE:
((Target, (Red, Circle)), (Above, ((Distractor, (Blue, Square)), (To-left-of, (Green, Triangle)))))
"A red circle target is above a blue square distractor that is to the left of a green triangle"

---

## CRITICAL: EVENT AND AGENT SUBTREES CANNOT BE EXTENDED

The Event subtree (7 tags) and Agent subtree (6 tags) do NOT allow extension.
Instead of extending, you must GROUP these tags with descriptive Items/Properties.

### NON-EXTENDABLE TAGS (memorize these!):
EVENT SUBTREE:
- Event, Sensory-event, Agent-action, Data-feature
- Experiment-control, Experiment-procedure, Experiment-structure, Measurement-event

AGENT SUBTREE:
- Agent, Human-agent, Animal-agent, Avatar-agent
- Controller-agent, Robotic-agent, Software-agent

### PATTERN: Group agents with descriptive Items/Properties, don't extend!

WRONG: Human-agent/Subject (CANNOT extend Human-agent!)
RIGHT: (Human-agent, Experiment-participant)

WRONG: Animal-agent/Marmoset
RIGHT: (Animal-agent, Animal/Marmoset)

WRONG: Robotic-agent/Drone
RIGHT: (Robotic-agent, Robot/Drone)

WRONG: Software-agent/Algorithm
RIGHT: (Software-agent, Label/My-algorithm)

WRONG: Controller-agent/Computer
RIGHT: (Controller-agent, Computer)

### How to describe agents:
1. Pick the agent TYPE from Agent subtree: Human-agent, Animal-agent, etc.
2. GROUP it with descriptive tags from Item or Property subtrees
3. Use Label/X for custom names if no appropriate Item exists

EXAMPLES FOR EACH AGENT TYPE:

Human-agent:
- (Human-agent, Experiment-participant) - subject in experiment
- (Human-agent, Experimenter) - researcher running experiment

Animal-agent:
- (Animal-agent, Animal/Marmoset) - a marmoset (extend from Animal)
- (Animal-agent, Animal/Dolphin) - a dolphin

Robotic-agent:
- (Robotic-agent, Robot/Arm) - a robotic arm
- (Robotic-agent, Robot/Drone) - a drone

Controller-agent:
- (Controller-agent, Computer) - computer controlling experiment
- (Controller-agent, Machine/Stimulator) - a stimulation device

Software-agent:
- (Software-agent, Label/BCI-decoder) - a brain-computer interface algorithm

---

## EXTENSION RULES (for extendable tags)

When you MUST extend (concept not in vocabulary AND parent allows extension),
extend from the MOST SPECIFIC applicable parent tag.

### WRONG: Extending from overly general parents
- Item/Cottage (too general; Cottage is-a Building, not just Item)
- Action/Cartwheel (too general; Cartwheel is-a Move-body action)
- Object/Rickshaw (too general; Rickshaw is-a Vehicle)

### CORRECT: Extending from most specific parents
- Building/Cottage (Cottage is-a Building - correct taxonomy)
- Move-body/Cartwheel (Cartwheel is-a body movement)
- Vehicle/Rickshaw (Rickshaw is-a vehicle)
- Animal/Marmoset (Marmoset is-a animal)
- Furniture/Armoire (Armoire is-a furniture)

### Extension Decision Process
1. Is concept in vocabulary? Use it directly.
2. Is parent in Event or Agent subtree? DO NOT EXTEND - use grouping instead.
3. Find the schema path to similar concepts.
4. Extend from the DEEPEST (most specific) parent that maintains is-a relationship.

### Cannot Extend These (use grouping instead)
- Event subtree - group with modality tags (Visual-presentation, etc.)
- Agent subtree - group with Item tags (Animal/X, Experiment-participant, etc.)
- Value-taking nodes (tags with # child) - cannot extend after #

---

## DEFINITION SYSTEM

Definitions allow naming reusable annotation patterns for consistency and brevity.
They are essential for Onset/Offset temporal scoping and for reducing repetition.

### Why Use Definitions
1. REUSABILITY: Define once, use many times with Def
2. CONSISTENCY: Ensure same annotation structure throughout dataset
3. TEMPORAL SCOPING: Required anchor for Onset/Offset/Inset
4. PARAMETERIZATION: Use # to create templates with variable values

### Creating Definitions (in sidecars only)
Pattern: (Definition/Name, (tag1, tag2, tag3))
With placeholder: (Definition/Name/#, (Tag1/# units, Tag2))

EXAMPLE: (Definition/RedCircle, (Sensory-event, Visual-presentation, (Red, Circle)))
MEANING: Defines "RedCircle" as a sensory event showing a red circle

EXAMPLE: (Definition/Acc/#, (Acceleration/# m-per-s^2))
MEANING: Defines "Acc" as an acceleration value with m-per-s^2 units

EXAMPLE: (Definition/ButtonPress, (Agent-action, Participant-response, (Press, Mouse-button)))
MEANING: Defines "ButtonPress" as a participant pressing a mouse button

### Using Definitions with Def
Pattern: Def/Name or Def/Name/value (if definition has placeholder)

EXAMPLE: Def/RedCircle
EXPANDS TO: (Sensory-event, Visual-presentation, (Red, Circle))

EXAMPLE: Def/Acc/9.8
EXPANDS TO: (Acceleration/9.8 m-per-s^2)

### Definition Naming Conventions
- Use descriptive, meaningful names: RedCircle, TargetAppears, ResponseGiven
- Use CamelCase or hyphenated names: Red-circle, Target-appears
- Avoid generic names: Event1, Def1, Thing
- Keep names concise but clear

### Definitions for Temporal Scoping
When using Onset/Offset, the Definition provides the anchor.

SIDECAR DEFINITION:
(Definition/VideoPlayback, (Sensory-event, Visual-presentation, Movie))

EVENT FILE USAGE:
Row 1: (Def/VideoPlayback, Onset)  # Video starts
Row 2: (Def/VideoPlayback, Offset)  # Video ends

### Parameterized Definitions
Use # as placeholder for values that change.

DEFINITION: (Definition/Tone/#, (Auditory-presentation, Tone, Frequency/# Hz))

USAGE:
Def/Tone/440 -> (Auditory-presentation, Tone, Frequency/440 Hz)
Def/Tone/880 -> (Auditory-presentation, Tone, Frequency/880 Hz)

### Def-expand (DO NOT USE)
Def-expand is created automatically by HED tools during validation/processing.
It shows the expanded content for debugging. Never write Def-expand manually.

### Definition Rules
- Definitions can ONLY appear in sidecars or external definition files
- Definitions CANNOT appear in the HED column of event files directly
- Cannot contain Def, Def-expand, or nested Definition tags
- If using #, must have exactly TWO # characters (one in name, one in content)
- Definition names must be unique across the entire dataset
- Names are case-sensitive: RedCircle and redcircle are different

---

## TEMPORAL SCOPING (Onset/Offset/Duration/Inset)

HED provides several ways to annotate the temporal extent of events.

### Using Duration (simpler approach)
Pattern: (Duration/value units, (event-content))
Use Duration when you know exactly how long something lasts.

EXAMPLE: (Duration/2 s, (Sensory-event, Visual-presentation, Cue, (Cross)))
MEANING: A cross cue is displayed for 2 seconds

EXAMPLE: (Duration/500 ms, (Auditory-presentation, Beep))
MEANING: A beep is presented for 500 milliseconds

### Using Onset/Offset (for explicit start/end markers)
Use Onset/Offset when start and end are recorded as separate events in data.
Requires a Definition anchor to link start and end events.

STEP 1: Define the event type in sidecar
(Definition/Fixation-display, (Sensory-event, Visual-presentation, Fixation-point))

STEP 2: Mark start with Onset
(Def/Fixation-display, Onset)

STEP 3: Mark end with Offset
(Def/Fixation-display, Offset)

### Onset/Offset Rules
- Onset and Offset MUST use the same Definition anchor
- Each Onset must have a matching Offset (eventually)
- Multiple instances can overlap if they use different Definition anchors
- Onset marks when the scoped event BEGINS
- Offset marks when the scoped event ENDS

### Using Inset (for markers during ongoing events)
Use Inset to mark intermediate time points within an Onset/Offset scope.

Pattern: (Def/Event-name, Inset)

EXAMPLE:
  t=0: (Def/Video-playback, Onset)  # Video starts
  t=5: (Def/Video-playback, Inset), (Scene-change)  # Scene change during video
  t=10: (Def/Video-playback, Inset), (Face, Appears)  # Face appears
  t=30: (Def/Video-playback, Offset)  # Video ends

### When to Use Each

Duration:
- Event duration is known and can be specified directly
- Event starts at the annotated time point
- Simpler when you have duration information

Onset/Offset:
- Start and end are recorded as separate event rows
- Duration may vary or be unknown at start
- Need to track overlapping instances

Inset:
- Need to mark points within a longer event
- Annotating sub-events or state changes
- Used between Onset and Offset of the same anchor

### Combining with Delay
Use Delay for events that occur after the trigger point.

EXAMPLE: (Delay/1 s, (Duration/2 s, (Sensory-event, Visual-presentation, Target)))
MEANING: Target appears 1 second after event marker, displays for 2 seconds

---

## SIDECAR SYNTAX (events.json)

### Value Placeholders (#)
For columns with varying values, use # as placeholder.
The # indicates a value that will be substituted from the column.

EXAMPLE: {{"age": {{"HED": "Age/# years"}}}}
For age=25: assembles to "Age/25 years"

EXAMPLE: {{"response_time": {{"HED": "Response-time/# ms"}}}}
For response_time=350: assembles to "Response-time/350 ms"

### Units with # Placeholders
When using # with value-taking tags, always include the unit.
Common unit patterns:
- Time: Duration/# s, Response-time/# ms, Delay/# s
- Frequency: Frequency/# Hz
- Distance: Distance/# m, Height/# cm
- Angle: Angle/# deg, Rotation-angle/# rad
- Acceleration: Acceleration/# m-per-s^2
- Proportion: Probability/# (unitless)

WRONG: Duration/#, Frequency/# (missing units)
RIGHT: Duration/# s, Frequency/# Hz

### Column References (curly braces)
Reference other columns to assemble grouped annotations.
Curly braces control how annotations from multiple columns are assembled together.

BASIC PATTERN: {{column_name}}
This inserts the HED annotation from that column at this position.

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

### Advanced Curly Brace Patterns

PATTERN 1: Grouping properties together
When properties should form a single group, put curly braces inside parentheses:
HED: "({{object_color}}, {{object_shape}}, {{object_size}})"

PATTERN 2: Multiple independent groups
HED: "({{target_color}}, {{target_shape}}), ({{distractor_color}}, {{distractor_shape}})"

PATTERN 3: Nested relationships
HED: "(({{agent_type}}, {{agent_name}}), ({{action}}, ({{object}})))"

### Handling n/a Values in Assembly
When a column value is "n/a" or empty, its annotation is omitted.

EXAMPLE:
{{
  "response": {{
    "HED": {{
      "button_press": "Participant-response, (Press, ({{response_hand}}, Mouse-button))"
    }}
  }},
  "response_hand": {{"HED": {{"left": "Left", "right": "Right"}}}}
}}

For response=button_press, response_hand=left:
ASSEMBLES TO: Participant-response, (Press, (Left, Mouse-button))

For response=button_press, response_hand=n/a:
ASSEMBLES TO: Participant-response, (Press, (Mouse-button))

### Curly Brace Rules
- Only valid in sidecars (not in event file HED column directly)
- Must reference existing columns with HED annotations
- No circular references (A references B, B references A)
- Use for grouping related properties from different columns
- Empty/n/a values in referenced columns are silently omitted
- Curly braces can appear multiple times in the same annotation
- Column references are case-sensitive

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
- Animal/Dolphin (not Agent/Dolphin or Animal/Dolphin)

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

### AGENTS (CANNOT extend - use grouping!)
Agent subtree tags CANNOT be extended. Group with descriptive Items instead.

FOR HUMANS: (Human-agent, Experiment-participant) or (Human-agent, Experimenter)
FOR ANIMALS: (Animal-agent, Animal/Marmoset) - extend from Item/Animal
FOR ROBOTS: (Robotic-agent, Robot/Drone) - extend from Item/Robot
FOR SOFTWARE: (Software-agent, Label/My-algorithm) - use Label for custom names
FOR CONTROLLERS: (Controller-agent, Computer) or (Controller-agent, Machine/Stimulator)

WRONG: Human-agent/Subject, Animal-agent/Marmoset, Robotic-agent/Drone
RIGHT: (Human-agent, Experiment-participant), (Animal-agent, Animal/Marmoset), (Robotic-agent, Robot/Drone)

---

## COMMON PATTERNS

### Visual stimulus
Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)

### Human participant response
Agent-action, Participant-response, ((Human-agent, Experiment-participant), (Press, (Left, Mouse-button)))

### Animal agent action
Agent-action, ((Animal-agent, Animal/Marmoset), (Reach, Target))

### Robot agent action
Agent-action, ((Robotic-agent, Robot/Arm), (Move, Target))

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

## COMMON ERRORS AND TROUBLESHOOTING

### Error: TAG_EXTENSION_INVALID
CAUSE: Extending a tag with a child that already exists in schema vocabulary.

EXAMPLE ERRORS:
- Red-color/Red/DarkRed  (DarkRed may exist in vocab, use it directly)
- Sensory-presentation/Red  (Red exists in vocab, don't re-extend)
- Item/Window  (Window exists in vocab, use it directly)

FIX: Check vocabulary first. If tag exists, use it directly without slash extension.

WRONG: Building/House  (if House is in vocabulary)
RIGHT: House

WRONG: Action/Press  (if Press is in vocabulary)
RIGHT: Press

### Error: TAG_INVALID
CAUSE: Tag or extension is not valid in the schema.

EXAMPLE ERRORS:
- ReallyInvalid/Extension  (base tag doesn't exist)
- ReallyInvalid  (tag not in schema)
- Label #  (# used incorrectly outside sidecar)

FIX: Use only tags from the vocabulary or valid extensions from extendable tags.

WRONG: Stimulus/Visual  (Stimulus not in vocab)
RIGHT: Sensory-event, Visual-presentation

WRONG: Response/Button  (Response not a valid base)
RIGHT: Participant-response, (Press, Button)

### Error: VALUE_INVALID
CAUSE: Value substituted for placeholder (#) is incorrect format.

EXAMPLE ERRORS:
- Def/Acc/MyMy  (text instead of number for acceleration)
- Distance/4mxxx  (malformed unit)
- Duration/fast  (text instead of number)

FIX: Use correct value format with proper units.

WRONG: Duration/fast
RIGHT: Duration/2 s

WRONG: Frequency/high
RIGHT: Frequency/1000 Hz

WRONG: Distance/4mxxx
RIGHT: Distance/4 m

### Error: UNIT_CLASS_INVALID
CAUSE: Wrong unit type for the value.

EXAMPLE ERRORS:
- Duration/5 Hz  (Hz is frequency, not time)
- Frequency/3 s  (s is time, not frequency)

FIX: Match unit to tag's expected unit class.

Time units: s, ms, second, seconds, minute, minutes, hour
Frequency units: Hz, kHz, mHz
Distance units: m, cm, mm, km, ft, mile
Angle units: rad, deg, degree

### Error: CHARACTER_INVALID
CAUSE: Extension name contains invalid characters.

EXAMPLE ERRORS:
- Red/Red$2  ($ not allowed)
- Red/R#d  (# not allowed in extension names)

FIX: Use only letters, numbers, and hyphens in extension names.

WRONG: Animal/Cat$1
RIGHT: Animal/Cat-1 or Animal/Cat1

### Error: PARENTHESES_MISMATCH
CAUSE: Opening and closing parentheses don't match.

EXAMPLE ERRORS:
- ((Red, Circle)  (missing closing paren)
- (Red, Circle))  (extra closing paren)
- ((A, (B, C)))  (correct - properly nested)

FIX: Count parentheses; each ( must have matching ).

### Error: DEFINITION_INVALID
CAUSE: Definition used incorrectly.

EXAMPLE ERRORS:
- Definition/Name in HED column  (definitions only in sidecars)
- (Definition/X, (Def/Y))  (cannot nest Def inside Definition)
- (Definition/A, (Definition/B))  (cannot nest definitions)

FIX: Definitions only in sidecars, cannot contain Def or nested Definition.

### Quick Validation Checklist
Before submitting annotations:
1. Every tag exists in vocabulary OR is valid extension?
2. Extensions use most specific parent?
3. Event/Agent tags are NOT extended (use grouping)?
4. Value tags have proper units?
5. Parentheses are balanced?
6. Definitions only in sidecar, not event file?
7. Properties grouped with their objects?

---

## OUTPUT FORMAT

Output ONLY the HED annotation string.
NO explanations, NO markdown, NO code blocks, NO commentary.
Just the raw HED annotation.
"""
