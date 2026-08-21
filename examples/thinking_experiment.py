"""Measure whether extended thinking earns its cost on the annotation agent.

Runs the same 15 benchmark descriptions through the full workflow under
several thinking configurations, varying only the annotation LLM. The
evaluation and keyword LLMs stay on Haiku 4.5 with thinking off in every arm,
so any difference is attributable to annotation.

Descriptions are the project's own benchmark cases (recovered from
examples/model_benchmark.py, deleted in 70ccdb8), chosen to avoid the
examples used in the annotation prompt.

Requires ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / ANTHROPIC_WORKSPACE_ID.
Writes a JSON result file and prints a summary table.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

from src.agents.workflow import HedAnnotationWorkflow
from src.utils.anthropic_llm import create_anthropic_llm
from src.utils.llm_usage import usage_scope

CASES = [
    ("cog_01", "easy", "An orange star flashes briefly at the top of the display"),
    (
        "cog_02",
        "medium",
        "A low-frequency buzzer sounds for 500 milliseconds followed by a high-frequency beep",
    ),
    (
        "cog_03",
        "medium",
        "The participant fixates on a central cross while a peripheral distractor appears "
        "in the lower right quadrant",
    ),
    (
        "cog_04",
        "hard",
        "A white noise burst masks the target word which was spoken by a female voice",
    ),
    (
        "cog_05",
        "hard",
        "The go signal consists of a green diamond appearing centrally, prompting a "
        "bimanual key press",
    ),
    (
        "animal_01",
        "medium",
        "A macaque monkey reaches toward a target on a touchscreen and receives a juice reward",
    ),
    (
        "animal_02",
        "medium",
        "The rat navigates through a virtual reality T-maze and turns left at the choice point",
    ),
    (
        "animal_03",
        "hard",
        "A rhesus monkey successfully grasps a pellet with a precision grip using thumb "
        "and index finger",
    ),
    (
        "animal_04",
        "hard",
        "The mouse receives an air puff to the whiskers as an aversive stimulus after "
        "incorrect lever press",
    ),
    (
        "animal_05",
        "hard",
        "A marmoset vocalizes in response to a playback of a conspecific phee call",
    ),
    (
        "para_01",
        "medium",
        "A rare deviant tone at 1200 Hz interrupts a sequence of standard 800 Hz tones",
    ),
    (
        "para_02",
        "medium",
        "An upright neutral face is presented for 200ms followed by a scrambled face mask",
    ),
    (
        "para_03",
        "medium",
        "The participant reaches to grasp a cylinder placed 30 centimeters in front of them",
    ),
    (
        "para_04",
        "hard",
        "A fearful facial expression appears in the left visual field while a happy face "
        "appears on the right",
    ),
    (
        "para_05",
        "hard",
        "Target letters T and L embedded among distractor letters O are searched in a visual array",
    ),
]

# Arms. temperature=None means "do not send one", which thinking requires.
ARMS = [
    ("haiku-nothink", "claude-haiku-4-5", 0.1, None),
    # Temperature control: thinking forces temperature off, so a no-thinking
    # arm at the same effective sampling isolates thinking from sampling.
    ("haiku-nothink-t1", "claude-haiku-4-5", None, None),
    ("haiku-think1024", "claude-haiku-4-5", None, {"type": "enabled", "budget_tokens": 1024}),
    ("haiku-think2048", "claude-haiku-4-5", None, {"type": "enabled", "budget_tokens": 2048}),
    ("sonnet-nothink", "claude-sonnet-5", None, {"type": "disabled"}),
    ("sonnet-adaptive", "claude-sonnet-5", None, {"type": "adaptive"}),
]

MAX_VALIDATION_ATTEMPTS = 3


def build_workflow(model: str, temperature: float | None, thinking: dict | None):
    """Build a workflow whose annotation LLM carries the arm's configuration."""
    annotation_llm = create_anthropic_llm(
        model=model,
        temperature=temperature if temperature is not None else 1.0,
        thinking=thinking,
        role="annotation",
    )
    # Support roles are identical in every arm.
    support_llm = create_anthropic_llm(
        model="claude-haiku-4-5", temperature=0.1, disable_reasoning=True, role="evaluation"
    )
    keyword_llm = create_anthropic_llm(
        model="claude-haiku-4-5",
        temperature=0.1,
        max_tokens=200,
        disable_reasoning=True,
        role="keyword",
    )
    return HedAnnotationWorkflow(
        llm=annotation_llm,
        evaluation_llm=support_llm,
        assessment_llm=support_llm,
        feedback_llm=support_llm,
        keyword_llm=keyword_llm,
        use_js_validator=False,
        lsp_client=None,
    )


async def run_case(workflow, description: str) -> dict:
    """Run one description and collect outcome, usage, and latency."""
    started = time.perf_counter()
    with usage_scope() as ledger:
        state = await workflow.run(
            input_description=description,
            schema_version="8.4.0",
            max_validation_attempts=MAX_VALIDATION_ATTEMPTS,
            run_assessment=False,
        )
    elapsed = time.perf_counter() - started
    totals = ledger.total()

    return {
        "annotation": state["current_annotation"],
        "is_valid": bool(state["is_valid"]) and not state["validation_errors"],
        "is_faithful": bool(state.get("is_faithful")),
        "validation_attempts": state.get("validation_attempts", 0),
        "total_iterations": state.get("total_iterations", 0),
        "errors": state.get("validation_errors", []),
        "warnings": state.get("validation_warnings", []),
        "latency_s": round(elapsed, 2),
        "calls": totals.calls,
        "input_tokens": totals.input_tokens,
        "cache_read_tokens": totals.cache_read_tokens,
        "output_tokens": totals.output_tokens,
        "cost_usd": round(totals.cost_usd, 6),
    }


async def main() -> None:
    results: dict[str, list[dict]] = {}

    for arm_name, model, temperature, thinking in ARMS:
        print(
            f"\n=== {arm_name} ({model}, temp={temperature}, thinking={thinking}) ===", flush=True
        )
        workflow = build_workflow(model, temperature, thinking)
        arm_results = []

        for case_id, difficulty, description in CASES:
            try:
                outcome = await run_case(workflow, description)
            except Exception as exc:  # a failed arm entry must not lose the run
                print(f"  {case_id:<10} ERROR {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                arm_results.append(
                    {"case": case_id, "difficulty": difficulty, "error": str(exc)[:300]}
                )
                continue

            outcome["case"] = case_id
            outcome["difficulty"] = difficulty
            arm_results.append(outcome)
            print(
                f"  {case_id:<10} {difficulty:<6} valid={str(outcome['is_valid']):<5} "
                f"faithful={str(outcome['is_faithful']):<5} "
                f"attempts={outcome['validation_attempts']} "
                f"{outcome['latency_s']:>6.2f}s  ${outcome['cost_usd']:.6f}",
                flush=True,
            )

        results[arm_name] = arm_results

    out = Path(os.environ.get("EXPERIMENT_OUT", "thinking_experiment.json"))
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    print(
        f"\n{'arm':<20} {'valid':>6} {'1st-try':>8} {'faithful':>9} {'attempts':>9} "
        f"{'latency':>9} {'cost':>10}"
    )
    for arm_name, entries in results.items():
        ok = [e for e in entries if "error" not in e]
        if not ok:
            print(f"{arm_name:<20} all runs failed")
            continue
        n = len(ok)
        valid = sum(e["is_valid"] for e in ok)
        first_try = sum(e["is_valid"] and e["validation_attempts"] <= 1 for e in ok)
        faithful = sum(e["is_faithful"] for e in ok)
        attempts = sum(e["validation_attempts"] for e in ok) / n
        latency = sum(e["latency_s"] for e in ok) / n
        cost = sum(e["cost_usd"] for e in ok)
        print(
            f"{arm_name:<20} {valid:>3}/{n:<2} {first_try:>5}/{n:<2} {faithful:>6}/{n:<2} "
            f"{attempts:>9.2f} {latency:>8.1f}s {cost:>10.4f}"
        )


asyncio.run(main())
