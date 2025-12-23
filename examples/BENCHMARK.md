# HEDit Model Benchmark

This document describes the benchmarking methodology for comparing LLM models on HED annotation quality.

## Overview

The benchmark evaluates multiple LLM models on their ability to generate valid, faithful, and complete HED annotations from natural language descriptions.

## Methodology

### Fair Comparison Setup

To ensure fair model comparison, the benchmark uses:

1. **Consistent Evaluation Model**: All annotation models are evaluated by the same model (`qwen/qwen3-235b-a22b-2507`). This is configured via the `--eval-model` CLI option.

2. **Separated Concerns**:
   | Agent | Model Used | Purpose |
   |-------|-----------|---------|
   | AnnotationAgent | `--model` (benchmarked) | Generates HED annotation |
   | EvaluationAgent | `--eval-model` (fixed) | Checks faithfulness |
   | AssessmentAgent | `--eval-model` (fixed) | Checks completeness |
   | FeedbackSummarizer | `--eval-model` (fixed) | Condenses errors for retry |

3. **Cache Warm-up**: Before benchmarking each model, a warm-up call is made to ensure all models start with equally "warm" caches for system prompts and schema context.

4. **CLI-based Execution**: All tests run through the `hedit` CLI for reproducibility. Each result includes the exact CLI command that can be re-run.

### Test Domains

The benchmark includes test cases across four domains:

1. **Cognitive Experiments** (`cognitive`): Standard visual/auditory stimuli
2. **Animal Experiments** (`animal`): Monkey/rat reaching, navigation, reward paradigms
3. **Paradigms** (`paradigm`): Oddball, face processing, visual search
4. **Images** (`image`): NSD (Natural Scenes Dataset) images

**Important**: Test cases intentionally avoid examples used in the annotation prompts to prevent "cheating" where models pattern-match from training.

### Metrics

For each annotation, the benchmark measures:

- **is_valid**: HED string passes schema validation
- **is_faithful**: Annotation accurately represents the description
- **is_complete**: All key elements from description are captured
- **validation_attempts**: Number of retry iterations needed
- **execution_time**: Total time to generate annotation

## Models to Benchmark

From [GitHub Issue #64](https://github.com/Annotation-Garden/HEDit/issues/64):

| Model | OpenRouter ID | Category |
|-------|---------------|----------|
| GPT-OSS-120B (baseline) | `openai/gpt-oss-120b` | Baseline |
| GPT-5.2 | `openai/gpt-5.2` | Quality |
| GPT-5.1-Codex-Mini | `openai/gpt-5.1-codex-mini` | Balanced |
| GPT-4o-mini | `openai/gpt-4o-mini` | Balanced |
| Gemini-3-Flash | `google/gemini-3-flash-preview` | Fast |
| Claude-Haiku-4.5 | `anthropic/claude-haiku-4.5` | Balanced |
| Mistral-Small-3.2-24B | `mistralai/mistral-small-3.2-24b-instruct` | Balanced |
| Nemotron-3-Nano-30B | `nvidia/nemotron-3-nano-30b-a3b` | Balanced |

**Evaluation Model**: `qwen/qwen3-235b-a22b-2507` (consistent across all tests)

## Running the Benchmark

### Prerequisites

1. Install HEDit with standalone dependencies:
   ```bash
   pip install hedit[standalone]
   ```

2. Set your OpenRouter API key:
   ```bash
   export OPENROUTER_API_KEY=your-key-here
   # or
   hedit init --api-key your-key-here
   ```

### Run All Tests

```bash
python examples/model_benchmark.py
```

### Run Specific Domains

```bash
# Single domain
python examples/model_benchmark.py cognitive

# Multiple domains
python examples/model_benchmark.py cognitive,animal

# Images only
python examples/model_benchmark.py image
```

### Output

Results are saved to `examples/benchmark_results/`:

- `benchmark_YYYYMMDD_HHMMSS.json`: Raw results in JSON format
- `report_YYYYMMDD_HHMMSS.md`: Human-readable markdown report

## Adding Test Cases

### Text-based Tests

Add new test cases to the appropriate list in `model_benchmark.py`:

```python
TestCase(
    id="cog_06",
    domain="cognitive",
    description="Your natural language event description",
    expected_elements=["Expected", "HED", "Tags"],
    difficulty="easy|medium|hard",
    notes="Optional notes about the test",
)
```

### Image Tests

Simply add images to `examples/images/` directory. The benchmark automatically discovers all `.jpg`, `.jpeg`, and `.png` files.

## Interpreting Results

### Summary Table

The report includes a summary table showing:
- **Valid %**: Percentage of annotations that pass HED validation
- **Faithful %**: Percentage rated as faithful to the description
- **Complete %**: Percentage rated as complete
- **Avg Attempts**: Average validation iterations needed
- **Avg Time**: Average execution time per annotation

### Domain Breakdown

Results are also broken down by domain to identify model strengths/weaknesses in specific areas (e.g., animal experiments vs. cognitive paradigms).

### Detailed Results

Each individual test result includes:
- The generated annotation
- Validation status and messages
- Evaluation feedback
- The exact CLI command for reproducibility

## Related Issues

- [#64: Explore alternative candidates for the default model](https://github.com/Annotation-Garden/HEDit/issues/64)
- [#69: Revisit Agent Prompts (semantic grouping issues)](https://github.com/Annotation-Garden/HEDit/issues/69)
