# Prompt Caching and Usage Reporting

Every HEDit annotation re-sends the same large HED vocabulary guide as its system
prompt. Prompt caching serves that prefix at a tenth of the input price, which is
where most of HEDit's per-request cost savings come from. This page records what is
actually cached, what it saves, and how to see the numbers.

Measurements below were taken against the Claude Platform on AWS endpoint
(`claude-haiku-4-5`) on 2026-08-20 using the Messages API `count_tokens` endpoint
and two real annotation runs.

## What caches, and what cannot

Anthropic will not create a cache entry for a prefix shorter than a per-model
minimum: **4096 tokens on Haiku 4.5**, 1024 on Sonnet 5.
A shorter prefix still carries its `cache_control` marker, but no entry is created,
nothing is charged for a write, and `cache_creation_input_tokens` stays 0.
There is no error, which is why these counters are the only trustworthy signal.

Measured system prompts, per agent role:

| Role | System prompt | Caches on Haiku 4.5 |
|---|---:|---|
| Annotation | 21,811 tokens | Yes |
| Evaluation | 623 tokens | No (below the 4096 minimum) |
| Assessment | 266 tokens | No |
| Feedback summarizer | 241 tokens | No |
| Keyword extraction | 186 tokens | No |
| Vision (task prompt, sent as a user message) | 51 tokens | No |

So annotation is the only role whose prefix can cache, and it is also the role that
dominates token spend: it carries the comprehensive guide, and it runs again on every
refinement iteration.
The other five roles send short prompts whose markers are harmless no-ops.

Padding those prompts to reach 4096 tokens would cost more than it saves (each padded
call would pay for thousands of tokens it does not need), so HEDit does not do it.
The markers stay in place because they cost nothing and would start working if a role
ever gained a large static prefix, or if a request runs on a model with a lower
minimum.

## What it saves

A cache read costs 0.1x the base input price; a 5-minute cache write costs 1.25x.
Two consecutive real annotation requests, same prompt prefix:

| | Request 1 (cache write) | Request 2 (cache read) |
|---|---|---|
| LLM calls | 3 | 3 |
| Input tokens | 22,741 | 22,740 |
| From cache | 0 | 21,805 |
| Cost | $0.029917 | $0.004670 |
| Cost without caching | $0.024466 | $0.024295 |
| Saved | none (write premium) | $0.019625 (81%) |

The first request in a cache window is *more* expensive than no caching at all,
because of the 1.25x write premium. Break-even is the second request; after that each
one costs roughly a fifth of the uncached price.

### Cache lifetime

The default entry lives 5 minutes, refreshed on every hit. That fits server traffic,
where requests arrive close together. Interactive use is different: if a user runs one
annotation every ten minutes, every request pays the write premium and never gets a
read, making caching a 25% surcharge instead of a saving.

For that traffic shape, set a 1-hour lifetime:

```bash
export HEDIT_PROMPT_CACHE_TTL=1h   # "5m" (default) or "1h"
```

A 1-hour write costs 2x instead of 1.25x, so it pays off from the third request in the
hour. Choose per deployment: `5m` for a busy server, `1h` for a single user working
through a session.

### Cache lanes

Caching is a prefix match, so any byte change invalidates the entry. Two things
legitimately fork the annotation prefix into separate lanes:

- **Schema version**: the guide embeds the vocabulary of the requested HED schema.
- **`no_extend` mode**: non-extension runs get a different guide.

Per-request content (the description, validation feedback, tag suggestions, semantic
hints, the previous annotation) belongs in the user message and must stay there.
`tests/test_annotation_agent.py::TestCachedPrefixStability` locks that in: it asserts
the system prompt is byte-identical across builds and free of request content. If a
timestamp, request id, or unsorted collection ever leaks into the prefix, cache reads
drop to zero and that test is what catches it.

## Seeing the numbers

Token, cost, and cache figures are collected per request by `src/utils/llm_usage.py`
and surfaced in four places.

**CLI** — printed under every annotation:

```
Usage and cache savings:
  22,740 input / 311 output tokens in 3 LLM calls
  21,805 input tokens read from cache (96% of input)
  $0.004670, saved $0.019625 (81%) by prompt caching
```

**API responses** — `/annotate` and `/annotate-from-image` return a `usage` object
(the streaming endpoints include it in the `result` event):

```json
{
  "calls": 3,
  "input_tokens": 22740,
  "cache_read_tokens": 21805,
  "cache_write_tokens": 0,
  "output_tokens": 311,
  "cache_hit_rate": 0.9589,
  "cost_usd": 0.00467,
  "uncached_cost_usd": 0.024295,
  "savings_usd": 0.019625,
  "savings_pct": 0.8078,
  "models": ["claude-haiku-4-5"],
  "unpriced_calls": 0
}
```

The web UI renders the same figures as a "Usage and Cache Savings" section.

**`GET /metrics`** — server-wide totals since startup, broken down by role and model.
Requires a server API key; BYOK callers get 403 and read their own numbers from their
annotation response instead.

```
annotation   calls=2  input= 43,692  cache_read= 21,805  hit_rate= 50%  cost=$0.029854
evaluation   calls=2  input=  1,393  cache_read=      0  hit_rate=  0%  cost=$0.004123
keyword      calls=2  input=    396  cache_read=      0  hit_rate=  0%  cost=$0.000611
```

**Telemetry** — each event's `performance` block carries `llm_calls`, `input_tokens`,
`output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `cache_hit_rate`,
`cost_usd`, and `uncached_cost_usd`.

## Extended thinking is off on purpose

Related to cost, and easy to mistake for an oversight: HEDit disables extended
thinking on every role that allows it. With thinking on, the agents emit much more
text and tend to circle in reasoning loops instead of converging on an annotation,
which costs latency and tokens without improving the result. On a model where
reasoning cannot be turned off, the factory requests the lowest reasoning effort
instead (`_ALWAYS_THINKING_MODELS` in `src/utils/anthropic_llm.py`).

Haiku 4.5, the default, does not think unless given a token budget, so nothing needs
to be switched off there. Sonnet 5 has adaptive thinking on by default and accepts
`thinking: {"type": "disabled"}`.

## Cost figures

Costs use Anthropic list prices ($1/$5 per MTok for Haiku 4.5, $3/$15 for Sonnet 5)
with the cache multipliers applied. Sonnet 5 has promotional pricing through
2026-08-31; list prices are used deliberately so reported cost is an upper bound
rather than an understatement. Prices cover LLM calls only, not validation or
schema loading. A model with no entry in the price table still contributes token
counts and is reported separately as `unpriced_calls`, so dollar figures never
silently omit a call.

## Verifying against the live endpoint

`tests/test_integration_anthropic.py::TestPromptCachingIntegration` checks both
halves of the claim on the real endpoint (skipped without `ANTHROPIC_API_KEY`):
a repeated 21.8k-token prefix is written and then read back with a measurable saving,
and a short prefix produces no cache activity at all.

```bash
uv run pytest tests/test_integration_anthropic.py -k PromptCaching -v
```
