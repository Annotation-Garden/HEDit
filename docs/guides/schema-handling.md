# HED Schema Handling in HED-BOT

## Overview

HED-BOT now uses JSON schemas from the official HED repository with proper support for:
- Short-form (leaf) tags
- Tag extension rules
- Vocabulary validation
- Closest match suggestions

## JSON Schema Location

**Default path**: `/Users/yahya/Documents/git/HED/hed-schemas/schemas_latest_json/`

Available schemas:
- `HEDLatest.json` - Standard HED schema (latest version)
- `HED_score_Latest.json` - SCORE library schema
- `HED_lang_Latest.json` - Language library schema

## Configuration

Set the schema directory in `.env`:
```bash
HED_SCHEMA_DIR=/Users/yahya/Documents/git/HED/hed-schemas/schemas_latest_json
```

## Short-Form Tags

**Critical**: HED-BOT uses SHORT-FORM tags (leaf nodes only).

**Correct**:
- `Square` (not `Item/Object/Geometric-object/2D-shape/Rectangle/Square`)
- `Red` (not `Property/Sensory-property/Visual-property/Color/Red`)
- `Visual-presentation` (not `Property/Sensory-property/Sensory-presentation/Visual-presentation`)

The annotation agent is instructed to:
1. Use only short-form tags from vocabulary
2. Prefer the most specific (deepest leaf) tag
3. Never use full path notation

## Tag Extension

Some tags allow extension with `/` notation:

**Format**: `BaseTag/ExtensionValue`

**Examples**:
- `Action/Reach` (if Action has extensionAllowed)
- `Label/MyCustomLabel`
- `Age/25 years` (for value tags with #)

**Important**:
- Only extend tags with `extensionAllowed` attribute
- Extended tags are dataset-specific and non-portable
- System warns when extensions are used
- Prefer schema tags over extensions

**How it works**:
1. JSON schema includes `extensionAllowed` attribute
2. Annotation agent knows which tags can be extended
3. Evaluation agent checks if extensions are valid
4. User is warned about portability issues

## Vocabulary Validation

The system validates tags in multiple stages:

### 1. Annotation Agent
- Receives full vocabulary list
- Receives list of extensionAllowed tags
- Uses comprehensive HED rules
- Generates annotations with valid tags

### 2. Validation Agent
- Uses HED Python/JavaScript validators
- Checks syntax and tag validity
- Provides specific error codes
- Feeds errors back to annotation agent

### 3. Evaluation Agent
- **NEW**: Checks each tag against vocabulary
- Suggests closest matches for invalid tags
- Validates tag extensions
- Warns about non-portable extensions

**Example evaluation feedback**:
```
Tag Suggestions:
- 'Circel' not in schema. Did you mean: Circle, Circular?
- 'Action/Grasp' uses extension (dataset-specific, non-portable)
- 'Color' doesn't allow extension. Use schema tag instead.
```

## HED Rules Integration

The system includes comprehensive HED rules covering:

### Syntax Rules
1. Short-form tags only
2. Tag extension with `/`
3. Grouping with parentheses
4. Curly braces for column references
5. Value tags with `#`
6. Comma separators

### Semantic Rules
1. Required classifications (Event + Task-event-role)
2. Sensory-event modality requirements
3. Object property grouping
4. Agent-action-object nesting
5. Spatial relationships
6. Reserved tags (Definition, Onset, etc.)
7. File type semantics
8. Reversibility principle

### Common Patterns
- Simple visual stimulus
- Stimulus with location
- Auditory stimulus
- Participant response
- Multiple stimuli
- Duration specification
- Spatial relationships
- Agent actions

## Closest Match Algorithm

When an invalid tag is detected, the system finds closest matches:

```python
def find_closest_match(invalid_tag: str) -> list[str]:
    # 1. Exact case-insensitive match
    # 2. Substring contains match
    # Returns up to 5 suggestions
```

**Examples**:
- `Circel` → `Circle`, `Circular`
- `Squar` → `Square`
- `Vis` → `Visual-presentation`, `Viseme`, `Visible`

## Extension Allowed Tags

Common extensionAllowed tags include:
- `Action` - For specific actions not in schema
- `Label` - For custom labels
- `Property` - For custom properties
- Value tags (with `#`) - For numeric values

**Checking if a tag is extendable**:
```python
from src.utils.json_schema_loader import load_latest_schema

schema = load_latest_schema()
is_extendable = schema.is_extendable("Action")  # True
is_extendable = schema.is_extendable("Event")   # False
```

## Evaluation/Assessment Feedback Loop

The workflow now properly integrates feedback:

```
1. Annotation Agent generates HED string
2. Validation Agent checks syntax/validity
   → If errors: feedback to Annotation Agent (retry)
3. Evaluation Agent assesses faithfulness
   → Checks tag validity against schema
   → Suggests closest matches
   → If not faithful: feedback to Annotation Agent (refine)
4. Assessment Agent checks completeness
   → If incomplete: feedback to Annotation Agent (optional refinement)
5. Return final annotation with all feedback
```

**Feedback sources**:
- Validation errors (syntax, invalid tags)
- Evaluation feedback (faithfulness, tag suggestions)
- Assessment feedback (completeness, missing elements)

**User options**:
1. **Default**: Validation loop only (automatic)
2. **Optional**: Evaluation refinement loop (configurable)
3. **Report only**: All feedback provided to user without refinement

## Schema Updates

To update to a new schema version:

```bash
# 1. Update HED schemas repository
cd /Users/yahya/Documents/git/HED/hed-schemas
git pull

# 2. Check new schemas are in schemas_latest_json/
ls schemas_latest_json/

# 3. Restart HED-BOT (automatic reload)
docker-compose restart hed-bot
```

## Debugging Schema Issues

### Check schema loading:
```python
from src.utils.json_schema_loader import load_latest_schema

schema = load_latest_schema()
print(f"Version: {schema.get_schema_version()}")
print(f"Vocabulary size: {len(schema.get_vocabulary())}")
print(f"Extendable tags: {len(schema.get_extendable_tags())}")
```

### Test tag validation:
```python
vocabulary = schema.get_vocabulary()
assert "Circle" in vocabulary
assert "Visual-presentation" in vocabulary

matches = schema.find_closest_match("Circel")
print(f"Did you mean: {matches}")
```

### Check extension:
```python
is_extendable = schema.is_extendable("Action")
print(f"Action extensionAllowed: {is_extendable}")
```

## Migration from XML to JSON

**Before** (XML-based):
- Used full-path tags sometimes
- Manual vocabulary extraction
- No extension validation
- No closest match suggestions

**After** (JSON-based):
- Short-form tags only
- Automatic vocabulary from JSON
- Extension validation and warnings
- Intelligent tag suggestions
- Complete HED rules integration

## Performance Considerations

**Schema loading**:
- JSON schemas loaded once at startup
- Cached in memory per agent
- ~1MB per schema (HEDLatest.json is 944KB)
- Fast lookup for tag validation

**Vocabulary lookup**:
- O(1) for existence check (set-based)
- O(n) for closest match (linear scan)
- ~2000-3000 tags in standard schema

## Summary

The new JSON schema system provides:
- ✓ Short-form tag enforcement
- ✓ Tag extension validation
- ✓ Closest match suggestions
- ✓ Comprehensive HED rules
- ✓ Evaluation feedback loop
- ✓ Production-ready validation
- ✓ Portable annotations (with warnings)

This makes HED-BOT truly production-ready for generating valid, portable HED annotations.
