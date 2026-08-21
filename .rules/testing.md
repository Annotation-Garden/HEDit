# Testing Standards - NO MOCKS Policy

## Core Philosophy: Test Reality, Not Fiction
**Why NO MOCKS?** Mocks test your assumptions, not your code.
**Real bugs** hide in integration points, not unit logic.
**Better approach:** No test is better than a false-confidence mock test.

## [STRICT] NO MOCKS, NO FAKE DATA
Never use mocks, stubs, or fake datasets. If real testing is not possible, don't write tests.
- **No mock objects** - Use real implementations
- **No mock datasets** - Use actual sample data
- **No stub services** - Connect to real test instances
- **Alternative:** Ask user for sample data or test environment setup

## When to Write Tests
- **DO:** Test with real data and actual dependencies
- **DO:** Use the Anthropic test credentials (`ANTHROPIC_API_KEY`, with
  `ANTHROPIC_BASE_URL` and `ANTHROPIC_WORKSPACE_ID` for the Claude Platform on AWS)
- **DO:** Test against actual file systems and real HED schemas
- **DON'T:** Write tests if only mocks would work
- **DON'T:** Create artificial test scenarios

## Test Structure
```
tests/
  conftest.py           # Real test fixtures
  test_agents.py        # Agent behavior tests
  test_validation.py    # Real HED validation
  test_workflow.py      # Full workflow with real API
  test_cli.py           # CLI command tests
```

## Frameworks
- **Python:** `pytest` with real fixtures and `coverage`
- **Integration tests:** Mark with `@pytest.mark.integration`
- **Skip integration:** `pytest -m "not integration"`
- **Only integration:** `pytest -m integration`
- **Coverage:** `pytest --cov=src`

## LLM and Prompt Testing

For prompts, responses, and AI-related functionality, use exemplar scenarios rooted in reality:

### Approaches (in order of preference)
1. **Recorded real conversations**: Capture actual user interactions as test fixtures
2. **Known real-world examples**: Extract cases from HED documentation, GitHub issues
3. **Cached API responses**: Record actual LLM responses and replay them
4. **Ground-truth Q&A pairs**: Domain expert-validated question/answer pairs
5. **Exemplar scenarios**: Write scenarios based on documented real cases

## When Real Testing Seems Impossible
**Think creatively before giving up:**
- Can you use the hedtools.org API for validation?
- Can you record real API responses for replay?
- Can you find real examples in HED documentation or issues?
- Can you use actual HED schemas for test data?

**Skipping is the LAST RESORT.** Before skipping:
1. Search GitHub issues for real examples
2. Check HED mailing list archives for actual user scenarios
3. Look at HED documentation for sample inputs/outputs

---
*NO MOCKS. Real tests build real confidence. Skipping is last resort.*
