# HED Annotation Core Rules

## Fundamental Principles

### 1. Reversibility Principle
Well-formed HED annotations can be translated back into coherent English.

### 2. Required Classifications
Every event needs:
- **Event Tag** (what happened): `Sensory-event`, `Agent-action`, etc.
- **Task-event-role** (role in task): `Experimental-stimulus`, `Cue`, `Participant-response`, etc.

## Critical Grouping Rules

### Rule 1: Group Object Properties
- Correct: `(Red, Circle)` - single object
- Wrong: `Red, Circle` - ambiguous

### Rule 2: Agent-Action-Object Nesting
Pattern: `Agent-action, ((Agent), (Action, (Object)))`
Example: `Agent-action, ((Human-agent), (Press, (Mouse-button)))`

### Rule 3: Curly Braces for Assembly
Use `{column_name}` to control multi-column assembly:
```json
"visual": "Sensory-event, Visual-presentation, ({color}, {shape})"
```

### Rule 4: Event Tags at Top Level
Event and Task-event-role should be at top level or grouped together.

### Rule 5: Sensory Events Need Modality
Every `Sensory-event` requires sensory-modality: `Visual-presentation`, `Auditory-presentation`, etc.

### Rule 6: Directional Relationships
Pattern: `(A, (Relation, C))` means "A has relationship to C"
Example: `((Red, Circle), (To-left-of, (Green, Square)))`

### Rule 7: Keep Independent Concepts Separate
Don't group unrelated concepts: `(Red, Press)` is semantically wrong.

### Rule 8: Reserved Tags
- `Definition`: Names reusable annotation strings
- `Def`: References a definition
- `Onset`/`Offset`/`Inset`: Temporal scope (timeline files only)
- `Duration`: Event duration
- `Delay`: Delays event start

## File Type Semantics

### Timeline Files (events.tsv)
- MUST include Event-type tag
- MUST include Task-event-role when applicable
- CAN use temporal scope tags (Onset/Offset)

### Descriptor Files (participants.tsv)
- MUST NOT include Event-type tags
- MUST NOT include temporal scope tags
- Describe properties/characteristics, not events

## Documentation Source
- `/Users/yahya/Documents/git/HED/hed-resources/docs/source/HedAnnotationSemantics.md`
- `/Users/yahya/Documents/git/HED/hed-resources/docs/source/HedAnnotationQuickstart.md`
