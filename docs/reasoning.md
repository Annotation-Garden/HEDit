# Extended Thinking: What It Buys and What It Costs

HEDit enables extended thinking on the annotation agent and disables it everywhere else.
This page records the measurement behind that split, because the question had been
settled by recollection in both directions before it was settled by numbers.

## The measurement

Six configurations, each run over the same 15 benchmark descriptions
(`examples/model_benchmark.py` cases: cognitive, animal, and paradigm domains, easy
through hard, deliberately avoiding the examples used in the annotation prompt). Only the
annotation LLM varied; evaluation and keyword extraction stayed on Haiku 4.5 with thinking
off in every arm. Run on 2026-08-20 against the Claude Platform on AWS endpoint, with
`max_validation_attempts=3`.

| Arm | Valid | Valid on 1st attempt | Faithful | Avg attempts | Avg latency | Cost/request | LLM calls |
|---|---|---|---|---|---|---|---|
| Haiku, no thinking (temp 0.1) | 15/15 | **5/15** | 5/15 | 1.87 | 10.3 s | $0.0092 | 71 |
| Haiku, no thinking (temp 1.0) | 14/15 | 6/15 | 5/15 | 1.73 | 10.0 s | $0.0086 | 66 |
| Haiku, 1024-token budget | 15/15 | 11/15 | 6/15 | 1.27 | 22.8 s | $0.0129 | 53 |
| **Haiku, 2048-token budget** | 15/15 | **13/15** | 7/15 | 1.13 | 20.9 s | $0.0114 | 49 |
| Sonnet 5, thinking off | 14/15 | 10/15 | 4/15 | 1.40 | 11.8 s | $0.0261 | 56 |
| Sonnet 5, adaptive | 15/15 | 12/15 | 6/15 | 1.20 | 12.9 s | $0.0267 | 51 |

What the numbers say:

- **Thinking more than doubles first-attempt validity on Haiku**, 5/15 to 13/15. That is
  the metric worth optimizing: each failed attempt costs another full
  annotate/validate/evaluate round, and every refinement pass is another chance to drift
  from the original description.
- **A larger budget is cheaper than a smaller one.** 2048 tokens cost less per request
  than 1024 ($0.0114 vs $0.0129) because it removed more refinement rounds than it added
  in thinking tokens. Input tokens fell from 635k to 387k across the run (fewer re-sends
  of the 22k-token prompt prefix), while output tokens rose from 10.5k to 23.6k.
- **Cost rises 24%** against the no-thinking baseline; total LLM calls fall by a third.
- **Latency roughly doubles**, 10.3 s to 20.9 s. This is the real price: thinking
  generation takes longer than the round trips it saves.
- **No reasoning loops appeared.** Average attempts went *down* with thinking (1.87 to
  1.13). The looping behavior seen previously does not reproduce on Anthropic models with
  a working validator gate; it was observed on open-source models, where thinking was also
  slow enough to erase the prompt-caching savings.
- **Sonnet 5 is not worth its price here.** Adaptive thinking on Sonnet reaches 12/15,
  no better than Haiku at 2048 tokens, for 2.3x the cost per request.
- **Temperature is not the explanation.** Thinking forces `temperature` off (the API
  rejects any value but 1 alongside thinking), so a no-thinking arm at temperature 1.0 was
  run as a control: 6/15 versus 5/15, i.e. no material difference.
- **Faithfulness barely moves** (5/15 to 7/15). Thinking buys first-pass validity, not
  judge-rated faithfulness.

Caveats: one run per arm at n=15, so expect a case or two of noise; the ordering of the
1024 and 2048 arms could flip on a rerun. The headline difference (5 to 13 of 15) is far
outside that range.

These numbers are only meaningful because #161 was fixed first. With the validator
downgrading every error to a warning, `is_valid` was always True and the refinement loop
never fired, so every arm would have scored a perfect first-attempt rate.

## What HEDit does now

| Role | Thinking |
|---|---|
| Annotation | On. 2048-token budget on Haiku 4.5; adaptive on Sonnet 5, which has no budget mode |
| Evaluation, assessment, feedback, keyword extraction | Off. Short structured tasks where reasoning added 5-10 s per call with no quality gain (#150) |
| Vision | Model default (no thinking on Haiku 4.5) |

Non-Anthropic models, should HEDit gain them (#163), default to thinking off: that is
where thinking was slow enough to cost more than it saved.

### Model choice

Haiku 4.5 with thinking is the default everywhere, and Sonnet 5 stays selectable in the
web app, the API (`X-Anthropic-Model`), and the CLI (`--model`). Keeping it visible is
deliberate: the measurement above says it is not the better choice here, and the way to
make that credible is to let people run both rather than to remove the option. The web
app labels it as the larger, 2.3x-cost model instead of the higher-quality one, and links
to this page.

To change or disable it:

```bash
export HEDIT_ANNOTATION_THINKING_BUDGET=4096   # bigger budget
export HEDIT_ANNOTATION_THINKING_BUDGET=off    # disable; also 0, false, none
```

Disabling is the right call for a latency-sensitive deployment: it halves time to first
annotation, at the cost of roughly two thirds of first-attempt validity. The refinement
loop still converges either way, which is why final validity is 15/15 in both arms.

## API constraints worth knowing

Verified against the endpoint, since each one is a 400 at request time rather than a
documented default:

- **Haiku 4.5 has no adaptive mode.** Thinking requires
  `{"type": "enabled", "budget_tokens": N}` with N at least 1024 and below `max_tokens`.
- **Sonnet 5 rejects `"enabled"`**: "`thinking.type.enabled` is not supported for this
  model. Use `thinking.type` adaptive". Adaptive lets the model choose its own depth, so
  the budget value does not apply.
- **Thinking and temperature are mutually exclusive**: "`temperature` may only be set to 1
  when thinking is enabled." `create_anthropic_llm` drops `temperature` when thinking is
  on rather than letting the request fail.
- **Thinking blocks are separate content blocks**, and `extract_text_content` keeps only
  `type == "text"`, so reasoning never reaches an annotation string.

## Reproducing

The harness lives in `examples/thinking_experiment.py`:

```bash
uv run python examples/thinking_experiment.py
```

It needs the server credentials, takes about 25 minutes, and costs a few dollars. Re-run
it when the default model changes (#64) or when a new model is offered.
