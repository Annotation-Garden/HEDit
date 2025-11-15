# Multi-Agent Architecture Design

## Agent Workflow

```
Natural Language Input
    ↓
[Annotation Agent] - Generate initial HED tags
    ↓
[Validation Agent] - Check HED compliance
    ↓
If errors → Feed back to Annotation Agent (max N iterations)
If valid ↓
[Evaluation Agent] - Assess faithfulness to original description
    ↓
If refinement needed → Back to Annotation Agent
If good ↓
[Assessment Agent] - Compare tags vs description for missing elements
    ↓
Final HED Annotation
```

## Agent Descriptions

### 1. Annotation Agent
**Role**: Convert natural language to HED tags
**Inputs**:
- Natural language event description
- HED schema vocabulary
- Validation feedback (if iteration)
**Outputs**: HED annotation string
**Context Needed**:
- Full HED schema hierarchy
- Annotation rules and examples
- Previous validation errors (if any)

### 2. Validation Agent
**Role**: Validate HED compliance using HED JavaScript validator
**Inputs**: HED annotation string
**Outputs**:
- Valid/Invalid status
- Detailed error messages with codes
- Warnings
**Tools**: HED JavaScript validator (via subprocess or API)

### 3. Evaluation Agent
**Role**: Assess how faithfully annotation captures the original description
**Inputs**:
- Original natural language description
- Generated HED annotation
**Outputs**:
- Faithfulness score/assessment
- Missing dimensions or elements
- Refinement suggestions
**Context Needed**:
- HED annotation semantics
- Reversibility principle

### 4. Assessment Agent
**Role**: Final comparison to identify still-missing elements
**Inputs**:
- Original description
- Final validated HED tags
**Outputs**:
- Completeness assessment
- List of missing elements/dimensions
- Annotator feedback

## LangGraph Implementation Strategy

### State Definition
```python
class HedAnnotationState(TypedDict):
    input_description: str
    current_annotation: str
    validation_errors: list
    validation_attempts: int
    evaluation_feedback: str
    is_valid: bool
    is_complete: bool
    messages: list
```

### Node Functions
- `annotate`: Generate/refine HED annotation
- `validate`: Run HED validator
- `evaluate`: Assess faithfulness
- `assess`: Final comparison
- `route`: Decision logic for next step

### Conditional Edges
- After validate: If errors and attempts < max → annotate, else → evaluate
- After evaluate: If refinement needed → annotate, else → assess
- After assess: End

## Key Insights from HED-LLM
1. **Validation loop is critical** - feed errors back for correction
2. **Vocabulary constraints** - prevent hallucination of non-existent tags
3. **Session caching** - schema and vocabulary expensive to reload
4. **Max iterations** - prevent infinite loops
5. **Agentic approach** - simple chains don't handle corrections well
