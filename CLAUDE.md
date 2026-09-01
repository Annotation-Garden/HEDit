@AGENTS.md

## Claude Code Specific Instructions

The shared project instructions live in `AGENTS.md`; this file imports them for Claude Code with `@AGENTS.md`.

- Use `uv` for all Python development (never pip/conda) and Bun for JS/TS (never npm/npx).
- Integration tests use real API calls with the `ANTHROPIC_API_KEY` credentials from `.env`; run with `uv run pytest -m integration`.
- Keep `plan.md` updated with progress.
