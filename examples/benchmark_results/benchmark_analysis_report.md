# HED Model Benchmark Analysis Report

**Generated**: 2025-12-23T10:53:45.424512

## Executive Summary

- **Best Quality (Faithful Rate)**: GPT-5.2 (100.0%)
- **Fastest**: Claude-Haiku-4.5 (10.9s avg)
- **Best Efficiency (Quality/Time)**: Claude-Haiku-4.5

## Combined Results (Text + Image)

| Model | Tests | Success | Valid | Faithful | Complete | Avg Time | Avg Attempts |
|-------|-------|---------|-------|----------|----------|----------|--------------|
| GPT-5.2 | 35 | 100% | 100% | 100% | 83% | 79.6s | 1.23 |
| GPT-5.1-Codex-Mini | 35 | 100% | 100% | 100% | 97% | 145.1s | 1.20 |
| Mistral-Small-3.2-24B | 35 | 100% | 100% | 100% | 91% | 13.0s | 1.60 |
| Claude-Haiku-4.5 | 35 | 97% | 97% | 97% | 94% | 10.9s | 1.76 |
| Gemini-3-Flash | 35 | 89% | 89% | 89% | 83% | 11.4s | 1.81 |
| GPT-4o-mini | 35 | 89% | 89% | 89% | 80% | 21.9s | 2.71 |
| GPT-OSS-120B | 35 | 74% | 74% | 74% | 60% | 14.8s | 2.46 |

## Text Benchmark Results

| Model | Tests | Success | Faithful | Avg Time |
|-------|-------|---------|----------|----------|
| GPT-5.2 | 15 | 100% | 100% | 42.5s |
| GPT-5.1-Codex-Mini | 15 | 100% | 100% | 35.9s |
| Gemini-3-Flash | 15 | 100% | 100% | 8.2s |
| GPT-OSS-120B | 15 | 100% | 100% | 8.3s |
| Mistral-Small-3.2-24B | 15 | 100% | 100% | 7.4s |
| GPT-4o-mini | 15 | 100% | 100% | 9.5s |
| Claude-Haiku-4.5 | 15 | 93% | 93% | 8.7s |

## Image Benchmark Results

| Model | Images | Success | Faithful | Avg Time |
|-------|--------|---------|----------|----------|
| GPT-5.2 | 20 | 100% | 100% | 107.3s |
| GPT-5.1-Codex-Mini | 20 | 100% | 100% | 226.9s |
| Claude-Haiku-4.5 | 20 | 100% | 100% | 12.6s |
| Mistral-Small-3.2-24B | 20 | 100% | 100% | 17.2s |
| Gemini-3-Flash | 20 | 80% | 80% | 13.8s |
| GPT-4o-mini | 20 | 80% | 80% | 31.2s |
| GPT-OSS-120B | 20 | 55% | 55% | 19.7s |

## Key Findings

### 1. Quality vs Speed Trade-off

There is a positive correlation (0.47) between execution time and quality, suggesting slower models tend to produce better annotations.

### 2. Text vs Image Performance

- **Gemini-3-Flash**: Performs 20% better on text benchmarks
- **GPT-OSS-120B**: Performs 45% better on text benchmarks
- **GPT-4o-mini**: Performs 20% better on text benchmarks

### 3. First-Pass Success

- **GPT-5.1-Codex-Mini**: 1.20 average attempts (lower is better)
- **GPT-5.2**: 1.23 average attempts (lower is better)
- **Mistral-Small-3.2-24B**: 1.60 average attempts (lower is better)

### 4. Error Analysis

- **Gemini-3-Flash**: 4 errors - recursion_limit(2), json_parse(2)
- **GPT-OSS-120B**: 9 errors - recursion_limit(6), json_parse(3)
- **Claude-Haiku-4.5**: 1 errors - json_parse(1)
- **GPT-4o-mini**: 4 errors - recursion_limit(4)

### 5. Token & Cost Analysis

> **Note**: Token counts below are *estimated* based on ~12K input + 300 output per attempt.
> Actual token usage varies by description complexity and model verbosity.
> These estimates are for relative comparison only.

**Estimated Token Usage per Annotation**:

| Model | Avg Attempts | Est. Tokens/Annotation | Faithful % | Est. Tokens/Quality Point |
|-------|--------------|------------------------|------------|---------------------------|
| GPT-5.1-Codex-Mini | 1.20 | 14,760 | 100% | 148 |
| GPT-5.2 | 1.23 | 15,111 | 100% | 151 |
| Mistral-Small-3.2-24B | 1.60 | 19,680 | 100% | 197 |
| Claude-Haiku-4.5 | 1.76 | 21,706 | 97% | 223 |
| Gemini-3-Flash | 1.81 | 22,219 | 89% | 251 |
| GPT-OSS-120B | 2.46 | 30,277 | 74% | 408 |
| GPT-4o-mini | 2.71 | 33,329 | 89% | 376 |

**Estimated Token Efficiency Ranking** (lower = better):

1. **GPT-5.1-Codex-Mini**: ~148 est. tokens/quality point (100% faithful, 1.20 attempts)
2. **GPT-5.2**: ~151 est. tokens/quality point (100% faithful, 1.23 attempts)
3. **Mistral-Small-3.2-24B**: ~197 est. tokens/quality point (100% faithful, 1.60 attempts)
4. **Claude-Haiku-4.5**: ~223 est. tokens/quality point (97% faithful, 1.76 attempts)
5. **Gemini-3-Flash**: ~251 est. tokens/quality point (89% faithful, 1.81 attempts)
6. **GPT-4o-mini**: ~376 est. tokens/quality point (89% faithful, 2.71 attempts)
7. **GPT-OSS-120B**: ~408 est. tokens/quality point (74% faithful, 2.46 attempts)

**OpenRouter Pricing Reference** ($/M tokens, before caching):

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| Mistral-Small-3.2-24B | $0.060 | $0.18 | |
| GPT-OSS-120B | $0.039 | $0.19 | |
| GPT-4o-mini | $0.150 | $0.60 | |
| GPT-5.1-Codex-Mini | $0.250 | $2.00 | |
| Gemini-3-Flash | $0.500 | $3.00 | |
| Claude-Haiku-4.5 | $0.800 | $4.00 | |
| GPT-5.2 | $1.750 | $14.00 | |

*Note: Anthropic offers ~90% caching discount, OpenAI ~75%. Actual costs depend on caching efficiency.*

**Best Estimated Token Efficiency** (100% quality with lowest est. token usage):

- **GPT-5.1-Codex-Mini**: ~14,760 est. tokens/annotation (most efficient at 100% quality)
- Uses ~**1.3x fewer tokens** than Mistral-Small-3.2-24B at same quality (estimated)

**Cost Efficiency (Quality per Dollar)**:

1. **Mistral-Small-3.2-24B**: 556 quality points per $/M ($0.18/M output)
2. **GPT-OSS-120B**: 391 quality points per $/M ($0.19/M output)
3. **GPT-4o-mini**: 148 quality points per $/M ($0.60/M output)
4. **GPT-5.1-Codex-Mini**: 50 quality points per $/M ($2.00/M output)
5. **Gemini-3-Flash**: 30 quality points per $/M ($3.00/M output)
6. **Claude-Haiku-4.5**: 24 quality points per $/M ($4.00/M output)
7. **GPT-5.2**: 7 quality points per $/M ($14.00/M output)

## Recommendations

Based on the analysis:

1. **For highest quality**: Use GPT-5.2 (100% faithful)
2. **Best value**: Use **Mistral-Small-3.2-24B** (100% faithful at $0.18/M output)
3. **For fastest results**: Use Claude-Haiku-4.5 (10.9s, 97% faithful)
4. **Cost savings**: Mistral-Small-3.2-24B is **78x cheaper** than GPT-5.2 with similar quality

## Figures

- `success_rate_comparison.png`: Text vs Image success rates
- `quality_metrics.png`: Valid/Faithful/Complete rates by model
- `time_vs_quality.png`: Speed-quality trade-off scatter plot
- `cost_quality_comparison.png`: **Cost vs Quality analysis**
- `comprehensive_comparison.png`: All metrics in one view
- `execution_time_distribution.png`: Time distribution box plots
- `validation_attempts.png`: Average iterations needed
- `error_analysis.png`: Error types breakdown
- `domain_performance.png`: Performance by test domain
- `image_complexity_analysis.png`: Description length vs success
