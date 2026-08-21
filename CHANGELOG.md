# Changelog

Notable changes to HEDit.
Versions follow [PEP 440](https://peps.python.org/pep-0440/) with a prerelease label per branch
(`.dev` on develop, `a` on main; see [docs/development/versioning.md](docs/development/versioning.md)).

This file starts at 0.7.11.
Earlier releases are described in the [GitHub releases](https://github.com/Annotation-Garden/HEDit/releases),
which are generated from the commit log.

## [0.7.11] - 2026-08-20

The LLM provider migration, plus token and cache accounting on every surface, and extended
thinking on the annotation agent.

### Added

- **Token, cost, and prompt-cache accounting** per request and per agent role
  (`src/utils/llm_usage.py`), reported on five surfaces: a usage section under every CLI
  annotation, a `usage` object on `/annotate` and `/annotate-from-image` responses (in the
  `result` event for the streaming endpoints), a "Usage and Cache Savings" panel in the web
  app, `llm_calls` / `cache_read_tokens` / `cache_write_tokens` / `cache_hit_rate` /
  `cost_usd` / `uncached_cost_usd` in telemetry events, and a new `GET /metrics` endpoint
  with server-wide totals broken down by role and model. Cost figures use Anthropic list
  prices; a model absent from the price table is counted as an `unpriced_calls` call rather
  than silently dropped. Bring-your-own-key callers get their own numbers in the annotation
  response and 403 from `/metrics`. (#155, #157, #164)
- **`HEDIT_PROMPT_CACHE_TTL`** to select a 5-minute (default) or 1-hour prompt cache
  lifetime. A 1-hour entry costs 2x to write instead of 1.25x and pays off from the third
  request in the hour, which suits interactive single-user traffic; `5m` suits a busy
  server. (#164)
- **Extended thinking on the annotation agent**: a 2048-token budget on Haiku 4.5, adaptive
  on Sonnet 5 (which rejects `enabled`), tunable or disableable with
  `HEDIT_ANNOTATION_THINKING_BUDGET`. Measured over 15 benchmark descriptions, this takes
  first-attempt validity from 5/15 to 13/15, cuts average validation attempts from 1.87 to
  1.13 and total LLM calls by a third, for 24% more cost per request and roughly twice the
  latency. Support roles (evaluation, assessment, feedback, keyword extraction) keep
  reasoning off, and any non-Anthropic model added later defaults to off. (#154, #165)
- **`X-Anthropic-*` request headers**: `X-Anthropic-Key`, `-Model`, `-Eval-Model`,
  `-Vision-Model`, and `-Temperature`. The legacy `X-OpenRouter-*` spellings are still
  accepted by the API and forwarded by the Cloudflare worker, so older clients and cached
  frontends keep working. (#164)
- **Documentation and reproducibility**: [docs/prompt-caching.md](docs/prompt-caching.md)
  (what caches, what it saves, how to verify it), [docs/reasoning.md](docs/reasoning.md)
  (the thinking measurement and per-role policy), `examples/thinking_experiment.py` (the
  harness, to re-run when the default model changes), and integration tests that verify a
  cache write and read against the live endpoint.

### Changed

- **All LLM calls now go to the Claude Platform on AWS** (the Anthropic-operated Messages
  API billed through AWS Marketplace, not Bedrock). Offered models are `claude-haiku-4-5`
  (default for annotation, evaluation, and vision) and `claude-sonnet-5`; Opus is
  deliberately not offered, and the evaluation judge stays on Haiku regardless of the
  annotation model. Legacy OpenRouter-style identifiers such as
  `anthropic/claude-haiku-4.5` are accepted as aliases. Multi-provider support is tracked
  in #163. (#155, #156, #158, #162)
- **Sonnet 5 is presented as the larger, 2.3x-cost option rather than the highest-quality
  one**, in the web app, CLI help, `ALLOWED_MODELS`, `.env.example`, and the deployment
  docs. It stays selectable everywhere it was: Haiku 4.5 with thinking matched it on the
  benchmark, and keeping both available is what lets anyone check that. (#165)
- Only the annotation prompt is large enough to cache (21.8k tokens against Haiku 4.5's
  4096-token minimum), so the other roles carry `cache_control` markers that are harmless
  no-ops rather than padded prompts. `tests/test_annotation_agent.py::TestCachedPrefixStability`
  now guards the prefix against request-specific content leaking in and silently zeroing
  cache reads. (#164)

### Fixed

- **Validator severity mapping** (#161): `hedtools` reports issue severity as an
  `ErrorSeverity` IntEnum, which was compared against the string `"error"`. The comparison
  was always false, so every error was recorded as a warning, `is_valid` was always True,
  and the refinement loop could never fire on the Python-validator path, which is what CLI
  standalone runs and servers without Node use. The JavaScript validator path was
  unaffected.
- An invalid model is now rejected with 400 before credentials are checked, so a bad model
  name no longer surfaces as a credential error. Missing server credentials are 503,
  distinct from 400. (#162)
- API error responses no longer quote the provider's error text. A rejected request used to
  forward the first 200 characters of the provider exception, which a provider can populate
  with request details and which the web app renders as HTML. Every failure class now
  returns a fixed message; the provider's wording stays in the server log. (#167)

### Removed

- OpenRouter and Ollama configuration, factory modules, Docker services, and documentation.
  (#162)
