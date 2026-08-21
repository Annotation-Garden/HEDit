# HEDit Multi-Agent Architecture

## Overview

HEDit uses a LangGraph-based multi-agent workflow to convert natural language event descriptions into valid HED annotation strings. The system implements iterative refinement with validation feedback loops.

## Agent Workflow

```
                    +----------+
                    | annotate |<---------+
                    +----+-----+          |
                         |                |
                    +----v-----+     +----+----------+
                    | validate |     | summarize     |
                    +----+-----+     | feedback      |
                         |           +----^----------+
               +---------+---------+      |
               |         |         |      |
          valid     max_attempts  invalid |
               |         |         +------+
          +----v----+    |
          | evaluate|    END
          +----+----+
               |
        +------+------+
        |             |
    faithful    not faithful
        |             |
   +----v----+   summarize
   | assess  |   feedback
   +----+----+   (loop back)
        |
       END
```

## Agents

### 1. AnnotationAgent (`src/agents/annotation_agent.py`)
- **Input**: Natural language description, schema version, prior feedback
- **Process**: Loads JSON schema vocabulary, builds system prompt with HED rules + vocabulary constraints, calls LLM
- **Output**: HED annotation string (short-form only, no parent paths)
- **Key feature**: Vocabulary-constrained generation via comprehensive system prompt from `src/utils/hed_rules.py`

### 2. ValidationAgent (`src/agents/validation_agent.py`)
- **Input**: HED annotation string, schema version
- **Process**: Dual-validator approach (JavaScript preferred, Python fallback)
- **Output**: validation_status, errors, warnings, augmented errors for LLM feedback
- **Key feature**: Error augmentation via `src/utils/error_remediation.py` for better LLM correction

### 3. EvaluationAgent (`src/agents/evaluation_agent.py`)
- **Input**: Original description, current annotation, schema vocabulary
- **Process**: Checks faithfulness, validates tags against vocabulary, suggests corrections
- **Output**: is_faithful, evaluation_feedback, portability warnings
- **Key feature**: Closest-match algorithm for invalid tags

### 4. AssessmentAgent (`src/agents/assessment_agent.py`)
- **Input**: Original description, final annotation, prior feedback
- **Process**: Checks completeness, identifies missing dimensions
- **Output**: is_complete, assessment_feedback
- **Key feature**: Optional (controlled by `run_assessment` flag)

### 5. FeedbackSummarizerAgent (`src/agents/feedback_summarizer.py`)
- **Input**: Validation errors, evaluation feedback
- **Process**: Condenses feedback for LLM consumption
- **Output**: Summarized feedback message for annotation agent
- **Key feature**: Error remediation and augmentation

### 6. VisionAgent (`src/agents/vision_agent.py`)
- **Input**: Base64-encoded image
- **Process**: Uses a Claude model (natively multimodal; no separate vision vendor)
- **Output**: Natural language description of image content
- **Key feature**: Enables image-to-HED annotation pipeline

## State Machine

Defined in `src/agents/state.py` as `HedAnnotationState(TypedDict)`:
- `messages`: Conversation history (LangChain `BaseMessage` list)
- `input_description`: Original natural language input
- `current_annotation`: Current HED string being refined
- `validation_status`: "pending" | "valid" | "invalid" | "max_attempts_reached"
- `validation_errors` / `validation_warnings`: For user display
- `validation_errors_augmented`: Enhanced errors for LLM feedback
- `is_valid`, `is_faithful`, `is_complete`: Status flags
- `max_validation_attempts`: Default 5
- `max_total_iterations`: Default 10

## Routing Logic

### After Validation
- `valid` -> evaluate
- `max_attempts_reached` -> END
- `invalid` -> summarize_feedback -> annotate (loop)

### After Evaluation
- `faithful + valid + assessment requested` -> assess -> END
- `faithful + valid + no assessment` -> END
- `not faithful` -> summarize_feedback -> annotate (loop)
- `max_total_iterations reached` -> END (or assess if requested)

## Configuration

- **LLM Provider**: Anthropic Claude via the Claude Platform on AWS (Anthropic-operated
  Messages API, AWS Marketplace billing; not Amazon Bedrock)
- **Offered models**: `claude-haiku-4-5` (default) and `claude-sonnet-5`
- **Default model**: Configurable via environment/headers
- **BYOK support**: Users can provide their own Anthropic key (`X-Anthropic-Key`),
  which routes to api.anthropic.com
- **Prompt caching**: The annotation system prompt is cached; usage and savings are
  reported per request (see `docs/prompt-caching.md`)
- **Streaming**: LangGraph `astream_events` for real-time progress via SSE
