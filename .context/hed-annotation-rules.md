# HED Annotation Rules

## Source
The annotation agent system prompt now uses official HED documentation
pulled from the hed-standard GitHub repos (HedAnnotationSemantics.md
and 02_Terminology.md), bundled in `src/data/hed-docs/` and loaded by
`src/utils/hed_docs_loader.py`. HEDit-specific sections (vocabulary
check, correction workflow, error troubleshooting, output format)
are in `src/utils/hed_comprehensive_guide.py`.

## Fundamental Principles

1. **Reversibility**: Every HED annotation must be translatable back to coherent natural language
2. **Short-form tags**: Always use the tag name only (e.g., `Red`), never parent paths (e.g., NOT `Property/Red`)
3. **Vocabulary-constrained**: Tags must come from the HED schema vocabulary
4. **Extension-conservative**: Only extend tags when no schema tag fits AND the base tag has `extensionAllowed`

## Syntax Rules (from `HED_SYNTAX_RULES`)

### Tag Form (CRITICAL)
- Use ONLY the tag name from the vocabulary, NO parent paths
- Tags in the schema are already in short-form
- Correct: `Red`, `Circle`, `Press`, `To-left-of`
- Wrong: `Property/Red`, `Item/Circle`, `Action/Press`

### When to Use Slash (/)
Only three valid uses:
1. **Extending**: Creating a NEW tag not in schema (e.g., `Action/Grasp` if "Grasp" not in vocab)
2. **Values with units**: `Duration/2 s`, `Frequency/440 Hz`
3. **Definitions**: `Definition/Red-circle`, `Def/Red-circle`

### Grouping
- Group properties of same object: `(Red, Circle)`
- Nest agent-action-object: `Agent-action, ((Agent), (Action, (Object)))`
- Ungrouped tags are separate entities

### Curly Braces
- Column references in JSON sidecars: `{column_name}`
- Example: `Visual-presentation, ({color}, {shape})`

### Value Tags
- `#` placeholder for values: `Age/# years`, `Duration/# ms`

## Semantic Rules (from `HED_SEMANTIC_RULES`)

### Required Classifications
Every event MUST include:
- **Event tag**: Sensory-event, Agent-action, etc.
- **Task-event-role**: Experimental-stimulus, Participant-response, etc.

### Sensory-Event Requirements
Must include sensory modality:
- `Visual-presentation`, `Auditory-presentation`, `Somatosensory-presentation`, etc.

### Grouping Rules
- Group object properties: `(Red, Circle, Large)`
- Nest agent-action-object: `Agent-action, ((Human-agent, Experiment-participant), (Press, (Left, Mouse-button)))`
- Spatial relationships: `((Red, Circle), (To-left-of, (Green, Square)))`
- Independent concepts stay separate (don't group unrelated tags)

### Reserved Tags
- `Definition/DefName`: Names reusable definitions
- `Def/DefName`: References a definition
- `Onset`, `Offset`, `Inset`: Temporal markers (timeline files only)

### File Type Semantics
- **Timeline files** (events.tsv): MUST have Event tags, CAN have temporal scope
- **Descriptor files** (participants.tsv): MUST NOT have Event or temporal scope tags

## Common Annotation Patterns (from `HED_ANNOTATION_PATTERNS`)

See `src/utils/hed_rules.py` for 8 documented patterns:
1. Simple visual stimulus
2. Visual stimulus with location
3. Auditory stimulus
4. Participant response
5. Multiple stimuli (same event)
6. With duration
7. Spatial relationship
8. Agent action with target

## Validation Error Guidance (from `HED_VALIDATION_GUIDANCE`)

Most common mistake: Adding parent paths to existing tags
- `TAG_EXTENSION_INVALID`: Using `Property/Red` instead of just `Red`
- `TAG_INVALID`: Tag not in vocabulary
- `PARENTHESES_MISMATCH`: Unbalanced parentheses
- `COMMA_MISSING`: Missing comma between tags
- `REQUIRE_CHILD`: Tag requires a child
- `VALUE_INVALID`: Value format mismatch
