# HEDit Usage Guide

## Overview

HEDit is a multi-agent system that converts natural language event descriptions into valid HED (Hierarchical Event Descriptors) annotations.

## Using the Web Interface

### 1. Open the Frontend

Open `frontend/index.html` in a web browser, or navigate to your deployed URL.

### 2. Enter Event Description

```
Example: A red circle appears on the left side of the screen
```

### 3. Configure Options

- **HED Schema Version**: Select the schema version (default: 8.3.0)
- **Max Validation Attempts**: Number of retry attempts (default: 5)

### 4. Generate Annotation

Click "Generate HED Annotation" and wait for the result.

### 5. Review Results

The system provides:
- **Generated Annotation**: The HED string
- **Status Badges**: Valid/Invalid, Faithful/Needs Refinement, Complete/Incomplete
- **Validation Feedback**: Errors and warnings
- **Evaluation Feedback**: Faithfulness assessment
- **Assessment Feedback**: Completeness analysis

## Using the API

### Base URL

```
http://localhost:38427
```

### Endpoints

#### 1. Generate Annotation

**POST** `/annotate`

```bash
curl -X POST "http://localhost:38427/annotate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A red circle appears on the left side of the screen",
    "schema_version": "8.3.0",
    "max_validation_attempts": 5
  }'
```

**Response**:
```json
{
  "annotation": "Sensory-event, Experimental-stimulus, Visual-presentation, ((Red, Circle), (Left-side-of, Computer-screen))",
  "is_valid": true,
  "is_faithful": true,
  "is_complete": true,
  "validation_attempts": 1,
  "validation_errors": [],
  "validation_warnings": [],
  "evaluation_feedback": "...",
  "assessment_feedback": "...",
  "status": "success"
}
```

#### 2. Validate HED String

**POST** `/validate`

```bash
curl -X POST "http://localhost:38427/validate" \
  -H "Content-Type": application/json" \
  -d '{
    "hed_string": "Sensory-event, Visual-presentation",
    "schema_version": "8.3.0"
  }'
```

**Response**:
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [],
  "parsed_string": "Sensory-event, Visual-presentation"
}
```

#### 3. Health Check

**GET** `/health`

```bash
curl "http://localhost:38427/health"
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "llm_available": true,
  "validator_available": true
}
```

## Python SDK

### Basic Usage

```python
import httpx
import asyncio

async def annotate_event(description: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:38427/annotate",
            json={
                "description": description,
                "schema_version": "8.3.0",
                "max_validation_attempts": 5
            },
            timeout=120.0  # Annotation can take time
        )
        return response.json()

# Run
result = asyncio.run(annotate_event("A red circle appears on screen"))
print(result["annotation"])
```

### Batch Processing

```python
import asyncio
import httpx

async def batch_annotate(descriptions: list[str]):
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            client.post(
                "http://localhost:38427/annotate",
                json={"description": desc}
            )
            for desc in descriptions
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

descriptions = [
    "A red circle appears on the left",
    "The participant presses the mouse button",
    "A green square flashes briefly"
]

results = asyncio.run(batch_annotate(descriptions))
for desc, result in zip(descriptions, results):
    print(f"{desc} -> {result['annotation']}")
```

## Understanding the Workflow

### 1. Annotation Agent

Generates initial HED tags from your description using:
- HED vocabulary constraints
- Semantic grouping rules
- Best practice patterns

### 2. Validation Agent

Validates the annotation using HED validation tools:
- Checks tag validity
- Verifies syntax
- Provides detailed error messages

### 3. Feedback Loop

If validation fails:
- Errors are fed back to the Annotation Agent
- Agent refines the annotation
- Process repeats (up to max attempts)

### 4. Evaluation Agent

Once valid, evaluates:
- Completeness
- Accuracy
- Semantic fidelity
- Missing elements

### 5. Assessment Agent

Final assessment provides:
- Captured elements
- Missing elements
- Optional enhancements
- Annotator guidance

## Best Practices

### Writing Good Descriptions

**Good**:
```
A red circle appears on the left side of the screen and the participant presses the left mouse button
```

**Better**:
```
An experimental visual stimulus consisting of a red circular target appears on the left side of the computer screen. The participant responds by pressing the left mouse button.
```

More detail helps the agent generate complete annotations.

### Understanding Results

#### Status Badges

- **Valid**: Annotation passes HED syntax validation
- **Faithful**: Annotation accurately captures the description
- **Complete**: No major missing elements

#### Validation Errors

Common errors:
- `TAG_INVALID`: Used a tag not in the vocabulary
- `PARENTHESES_MISMATCH`: Incorrect grouping syntax
- `COMMA_MISSING`: Missing required comma

The system automatically attempts to fix these!

#### Evaluation Feedback

Reviews:
- What was captured well
- What might be missing
- Suggestions for refinement

#### Assessment Feedback

Final check for:
- Completeness of annotation
- Optional enhancements
- Guidance for manual review

## Advanced Usage

### Custom Schema Versions

```bash
curl -X POST "http://localhost:38427/annotate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "...",
    "schema_version": "8.4.0"  # Use prerelease
  }'
```

### Adjusting Max Attempts

```bash
curl -X POST "http://localhost:38427/annotate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "...",
    "max_validation_attempts": 10  # More attempts for complex cases
  }'
```

## Troubleshooting

### Slow Response Times

- First request takes longer (model loading)
- Complex descriptions need more time
- Check GPU usage: `nvidia-smi`

### Validation Always Fails

- Description might be too vague
- Try providing more detail
- Check schema version compatibility

### Annotation Not Faithful

- Add more descriptive details
- Specify relationships explicitly
- Mention all relevant attributes

## Example Use Cases

### 1. Simple Visual Stimulus

**Input**: "A blue square appears in the center"

**Output**: `Sensory-event, Experimental-stimulus, Visual-presentation, ((Blue, Square), (Center-of, Computer-screen))`

### 2. Participant Action

**Input**: "The participant presses the spacebar"

**Output**: `Agent-action, Participant-response, ((Human-agent, Experiment-participant), (Press, Keyboard-key/Space))`

### 3. Complex Event

**Input**: "A red triangle target appears on the right side of the screen for 500 milliseconds while a beep sound plays"

**Output**: Complex HED string with multiple sensory modalities, temporal information, and spatial relationships.

## Getting Help

- Check validation errors for specific issues
- Review evaluation feedback for guidance
- Consult HED documentation: https://www.hedtags.org/
- File issues: Project repository

## API Rate Limits

Default: No rate limits (configure as needed for production)

For production deployments, consider:
- API key authentication
- Rate limiting middleware
- Queue system for batch processing
