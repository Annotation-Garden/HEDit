#!/usr/bin/env python3
"""Model benchmark for HED annotation quality comparison.

This script compares multiple LLM models on HED annotation tasks across
different domains, using the hedit CLI for reproducibility.

Related GitHub Issues:
- #64: Explore alternative candidates for the default model
- #69: Revisit Agent Prompts (semantic grouping issues)

Methodology:
1. Use `hedit annotate` CLI to generate HED annotations (reproducible)
2. Extract is_valid, is_faithful, is_complete from JSON output
3. Measure validation attempts and execution time
4. Generate comparison report

Test Domains (avoiding examples in prompt):
1. Standard cognitive experiments (different stimuli than prompt examples)
2. Animal experiments (monkey/rat reaching, navigation, reward)
3. Image annotations (NSD dataset images)
4. Optional paradigms (oddball, face processing, reaching)

Usage:
    python examples/model_benchmark.py                    # Run all tests
    python examples/model_benchmark.py cognitive          # Run specific domain
    python examples/model_benchmark.py cognitive,animal   # Run multiple domains
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

load_dotenv()

# Suppress LiteLLM debug output
litellm.suppress_debug_info = True

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SCHEMA_VERSION = "8.4.0"
MAX_VALIDATION_ATTEMPTS = 5

# Evaluation model - used consistently across all benchmarks for fair comparison
# This model evaluates annotation quality (is_faithful, is_complete)
EVAL_MODEL = "qwen/qwen3-235b-a22b-2507"
EVAL_PROVIDER = "Cerebras"  # Use Cerebras for fast Qwen inference

# Models to benchmark (from GitHub issue #64)
# https://github.com/Annotation-Garden/HEDit/issues/64#issuecomment-3684641652
MODELS_TO_BENCHMARK = [
    # Baseline: Current default (Cerebras - ultra fast, cheap)
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS-120B",
        "provider": "Cerebras",
        "category": "baseline",
    },
    # GPT-5.2 (OpenAI's latest)
    {
        "id": "openai/gpt-5.2",
        "name": "GPT-5.2",
        "provider": None,
        "category": "quality",
    },
    # GPT 5.1 Codex Mini
    {
        "id": "openai/gpt-5.1-codex-mini",
        "name": "GPT-5.1-Codex-Mini",
        "provider": None,
        "category": "balanced",
    },
    # GPT-4o-mini (OpenAI's cheap option)
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o-mini",
        "provider": None,
        "category": "balanced",
    },
    # Gemini 3 Flash (Google's fast option)
    {
        "id": "google/gemini-3-flash-preview",
        "name": "Gemini-3-Flash",
        "provider": None,
        "category": "fast",
    },
    # Claude Haiku 4.5 (Anthropic's fast option)
    {
        "id": "anthropic/claude-haiku-4.5",
        "name": "Claude-Haiku-4.5",
        "provider": None,
        "category": "balanced",
    },
    # Mistral Small 3.2 24B
    {
        "id": "mistralai/mistral-small-3.2-24b-instruct",
        "name": "Mistral-Small-3.2-24B",
        "provider": None,
        "category": "balanced",
    },
]


# ============================================================================
# TEST CASES
# ============================================================================
# IMPORTANT: These are intentionally DIFFERENT from examples in the prompts
# to avoid "cheating" where models just pattern-match from training.


@dataclass
class TestCase:
    """A single test case for benchmarking."""

    id: str
    domain: str
    description: str
    expected_elements: list[str]  # Key elements that should appear in annotation
    difficulty: str  # easy, medium, hard
    notes: str = ""


# Domain 1: Standard Cognitive Experiments
# (Different from prompt examples: no red circles, blue squares, etc.)
COGNITIVE_TESTS = [
    TestCase(
        id="cog_01",
        domain="cognitive",
        description="An orange star flashes briefly at the top of the display",
        expected_elements=["Sensory-event", "Visual-presentation", "Orange", "Star", "Top"],
        difficulty="easy",
        notes="Simple visual event with color, shape, location",
    ),
    TestCase(
        id="cog_02",
        domain="cognitive",
        description="A low-frequency buzzer sounds for 500 milliseconds followed by a high-frequency beep",
        expected_elements=["Sensory-event", "Auditory-presentation", "Duration", "Frequency"],
        difficulty="medium",
        notes="Auditory sequence with temporal and frequency information",
    ),
    TestCase(
        id="cog_03",
        domain="cognitive",
        description="The participant fixates on a central cross while a peripheral distractor appears in the lower right quadrant",
        expected_elements=[
            "Sensory-event",
            "Visual-presentation",
            "Fixation",
            "Cross",
            "Distractor",
        ],
        difficulty="medium",
        notes="Fixation task with spatial relationships",
    ),
    TestCase(
        id="cog_04",
        domain="cognitive",
        description="A white noise burst masks the target word which was spoken by a female voice",
        expected_elements=["Auditory-presentation", "Noise", "Speech", "Female"],
        difficulty="hard",
        notes="Auditory masking paradigm with voice characteristics",
    ),
    TestCase(
        id="cog_05",
        domain="cognitive",
        description="The go signal consists of a green diamond appearing centrally, prompting a bimanual key press",
        expected_elements=[
            "Agent-action",
            "Visual-presentation",
            "Green",
            "Press",
            "Participant-response",
        ],
        difficulty="hard",
        notes="Go/No-Go task with motor response",
    ),
]

# Domain 2: Animal Experiments
# (Monkey, rat, VR navigation, reaching, reward paradigms)
ANIMAL_TESTS = [
    TestCase(
        id="animal_01",
        domain="animal",
        description="A macaque monkey reaches toward a target on a touchscreen and receives a juice reward",
        expected_elements=["Agent-action", "Animal-agent", "Reach", "Target", "Reward"],
        difficulty="medium",
        notes="Primate reaching task with reward",
    ),
    TestCase(
        id="animal_02",
        domain="animal",
        description="The rat navigates through a virtual reality T-maze and turns left at the choice point",
        expected_elements=["Agent-action", "Animal-agent", "Navigate", "Left"],
        difficulty="medium",
        notes="Rodent VR navigation",
    ),
    TestCase(
        id="animal_03",
        domain="animal",
        description="A rhesus monkey successfully grasps a pellet with a precision grip using thumb and index finger",
        expected_elements=["Agent-action", "Animal-agent", "Grasp"],
        difficulty="hard",
        notes="Fine motor control in primates",
    ),
    TestCase(
        id="animal_04",
        domain="animal",
        description="The mouse receives an air puff to the whiskers as an aversive stimulus after incorrect lever press",
        expected_elements=["Sensory-event", "Animal-agent", "Incorrect-action", "Aversive"],
        difficulty="hard",
        notes="Aversive conditioning paradigm",
    ),
    TestCase(
        id="animal_05",
        domain="animal",
        description="A marmoset vocalizes in response to a playback of a conspecific phee call",
        expected_elements=["Agent-action", "Animal-agent", "Vocalize", "Auditory-presentation"],
        difficulty="hard",
        notes="Vocal communication in primates",
    ),
]

# Domain 3: Optional Paradigms (Oddball, Face Processing, Reaching)
PARADIGM_TESTS = [
    TestCase(
        id="para_01",
        domain="paradigm",
        description="A rare deviant tone at 1200 Hz interrupts a sequence of standard 800 Hz tones",
        expected_elements=["Auditory-presentation", "Frequency", "Oddball"],
        difficulty="medium",
        notes="Auditory oddball paradigm",
    ),
    TestCase(
        id="para_02",
        domain="paradigm",
        description="An upright neutral face is presented for 200ms followed by a scrambled face mask",
        expected_elements=["Sensory-event", "Visual-presentation", "Face", "Duration"],
        difficulty="medium",
        notes="Face processing with masking",
    ),
    TestCase(
        id="para_03",
        domain="paradigm",
        description="The participant reaches to grasp a cylinder placed 30 centimeters in front of them",
        expected_elements=["Agent-action", "Reach", "Grasp", "Distance", "Participant-response"],
        difficulty="medium",
        notes="Reaching and grasping task",
    ),
    TestCase(
        id="para_04",
        domain="paradigm",
        description="A fearful facial expression appears in the left visual field while a happy face appears on the right",
        expected_elements=[
            "Sensory-event",
            "Visual-presentation",
            "Face",
            "Emotion",
            "Left",
            "Right",
        ],
        difficulty="hard",
        notes="Emotional face lateralization",
    ),
    TestCase(
        id="para_05",
        domain="paradigm",
        description="Target letters T and L embedded among distractor letters O are searched in a visual array",
        expected_elements=[
            "Sensory-event",
            "Visual-presentation",
            "Target",
            "Distractor",
            "Search",
        ],
        difficulty="hard",
        notes="Visual search paradigm",
    ),
]


# Note: Image benchmarking is handled separately by image_benchmark.py
# This script focuses on text-based annotation benchmarking only.

ALL_TEST_CASES = COGNITIVE_TESTS + ANIMAL_TESTS + PARADIGM_TESTS


def run_hedit_annotate(
    description: str,
    model_id: str,
    provider: str | None = None,
    eval_model: str | None = None,
    eval_provider: str | None = None,
    schema_version: str = "8.4.0",
    max_attempts: int = 5,
    run_assessment: bool = True,
) -> tuple[dict, str, float]:
    """Run hedit annotate CLI command.

    Args:
        description: Natural language event description
        model_id: Model ID (e.g., "openai/gpt-oss-120b")
        provider: Provider preference (e.g., "Cerebras")
        eval_model: Model for evaluation/assessment (for consistent benchmarking)
        eval_provider: Provider for evaluation model (e.g., "Cerebras")
        schema_version: HED schema version
        max_attempts: Maximum validation attempts
        run_assessment: Whether to run completeness assessment

    Returns:
        Tuple of (parsed JSON result, CLI command string, execution time)
    """
    # Build CLI command
    cmd = [
        "hedit",
        "annotate",
        description,
        "--model",
        model_id,
        "--schema",
        schema_version,
        "--max-attempts",
        str(max_attempts),
        "-o",
        "json",
        "--standalone",
    ]

    if eval_model:
        cmd.extend(["--eval-model", eval_model])

    if eval_provider:
        cmd.extend(["--eval-provider", eval_provider])

    if provider:
        cmd.extend(["--provider", provider])

    if run_assessment:
        cmd.append("--assessment")

    # Command string for logging (quote the description)
    cmd_str = " ".join(cmd[:2]) + f' "{description}"' + " " + " ".join(cmd[3:])

    # Run command
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENROUTER_API_KEY": OPENROUTER_API_KEY or ""},
    )
    execution_time = time.time() - start_time

    # Parse JSON from stdout (filter out debug messages and find JSON block)
    stdout_lines = result.stdout.strip().split("\n")

    # Filter out known noise patterns
    filtered_lines = []
    for line in stdout_lines:
        # Skip workflow debug messages
        if line.startswith("[WORKFLOW]"):
            continue
        # Skip LiteLLM provider warnings (contain ANSI codes)
        if "Provider List" in line or "\x1b[" in line:
            continue
        # Skip empty lines at the start
        if not filtered_lines and not line.strip():
            continue
        filtered_lines.append(line)

    # Find the JSON block (starts with '{')
    json_start = None
    for i, line in enumerate(filtered_lines):
        if line.strip().startswith("{"):
            json_start = i
            break

    if json_start is not None:
        json_str = "\n".join(filtered_lines[json_start:])
    else:
        json_str = "\n".join(filtered_lines)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        parsed = {
            "status": "error",
            "error": f"JSON parse error: {e}",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    return parsed, cmd_str, execution_time


class ModelBenchmark:
    """Benchmark runner with incremental saving per model/domain.

    Design:
    - Saves results after each model completes a domain
    - Captures full JSON response from CLI
    - Resilient to failures - previous results are preserved
    """

    WARMUP_DESCRIPTION = "A visual stimulus appears on screen"

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(__file__).parent / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_domain_results(
        self, model_name: str, domain: str, results: list[dict[str, Any]]
    ) -> Path:
        """Save results for a single model/domain combination.

        Args:
            model_name: Short model name (e.g., "GPT-OSS-120B")
            domain: Domain name (e.g., "cognitive")
            results: List of result dicts with full JSON responses

        Returns:
            Path to saved file
        """
        # Sanitize model name for filename
        safe_name = model_name.replace("/", "-").replace(" ", "_").lower()
        filename = f"{self.session_id}_{safe_name}_{domain}.json"
        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "model_name": model_name,
                    "domain": domain,
                    "schema_version": SCHEMA_VERSION,
                    "eval_model": EVAL_MODEL,
                    "eval_provider": EVAL_PROVIDER,
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"    [SAVED] {output_file.name}")
        return output_file

    def warmup_model(self, model_config: dict) -> None:
        """Run a warm-up call to prime the cache."""
        model_id = model_config["id"]
        model_name = model_config["name"]
        provider = model_config.get("provider")

        print(f"  Warming up cache for {model_name}...")

        try:
            run_hedit_annotate(
                description=self.WARMUP_DESCRIPTION,
                model_id=model_id,
                provider=provider,
                eval_model=EVAL_MODEL,
                eval_provider=EVAL_PROVIDER,
                schema_version=SCHEMA_VERSION,
                max_attempts=1,
                run_assessment=False,
            )
            print("  Cache warmed up successfully")
        except Exception as e:
            print(f"  Warning: Warmup failed: {e}")

    def benchmark_model_domain(
        self,
        model_config: dict,
        test_cases: list[TestCase],
        domain: str,
    ) -> list[dict[str, Any]]:
        """Run benchmark for a single model on a single domain.

        Args:
            model_config: Model configuration dict
            test_cases: List of test cases (already filtered by domain)
            domain: Domain name for logging

        Returns:
            List of result dicts with full JSON responses
        """
        model_id = model_config["id"]
        model_name = model_config["name"]
        provider = model_config.get("provider")

        results = []

        for test_case in test_cases:
            print(f"\n    {test_case.id}: {test_case.description[:60]}...")

            try:
                parsed, cmd_str, exec_time = run_hedit_annotate(
                    description=test_case.description,
                    model_id=model_id,
                    provider=provider,
                    eval_model=EVAL_MODEL,
                    eval_provider=EVAL_PROVIDER,
                    schema_version=SCHEMA_VERSION,
                    max_attempts=MAX_VALIDATION_ATTEMPTS,
                    run_assessment=True,
                )

                # Build result with full JSON response
                result = {
                    "test_id": test_case.id,
                    "domain": domain,
                    "difficulty": test_case.difficulty,
                    "input_description": test_case.description,
                    "expected_elements": test_case.expected_elements,
                    "model_id": model_id,
                    "model_name": model_name,
                    "provider": provider,
                    "cli_command": cmd_str,
                    "execution_time_seconds": exec_time,
                    "full_response": parsed,  # Store full JSON response
                }

                # Extract key metrics for quick access
                metadata = parsed.get("metadata", {})
                annotation = parsed.get("hed_string", "")
                is_valid = parsed.get("is_valid", False)
                is_faithful = metadata.get("is_faithful")
                is_complete = metadata.get("is_complete")

                print(f"      Valid: {is_valid}, Faithful: {is_faithful}, Complete: {is_complete}")
                if annotation:
                    print(f"      HED: {annotation[:60]}...")

                if parsed.get("status") == "error":
                    print(f"      ERROR: {parsed.get('error', 'Unknown')}")

                results.append(result)

            except Exception as e:
                print(f"      EXCEPTION: {e}")
                import traceback

                traceback.print_exc()

                results.append(
                    {
                        "test_id": test_case.id,
                        "domain": domain,
                        "difficulty": test_case.difficulty,
                        "input_description": test_case.description,
                        "expected_elements": test_case.expected_elements,
                        "model_id": model_id,
                        "model_name": model_name,
                        "provider": provider,
                        "cli_command": "",
                        "execution_time_seconds": 0,
                        "full_response": {"status": "exception", "error": str(e)},
                    }
                )

        return results

    def run_benchmark(
        self,
        models: list[dict] | None = None,
        domains: list[str] | None = None,
    ):
        """Run benchmark with incremental saving per model/domain.

        Args:
            models: List of model configs (defaults to MODELS_TO_BENCHMARK)
            domains: Filter to specific domains (defaults to all)
        """
        models = models or MODELS_TO_BENCHMARK
        all_domains = ["cognitive", "animal", "paradigm"]
        domains = domains or all_domains

        # Group test cases by domain
        domain_tests = {d: [tc for tc in ALL_TEST_CASES if tc.domain == d] for d in domains}

        print(f"\n{'#' * 80}")
        print("# HED MODEL BENCHMARK (Incremental Saving)")
        print(f"# Session: {self.session_id}")
        print(f"# Models: {len(models)}")
        print(f"# Domains: {domains}")
        print(f"# Eval Model: {EVAL_MODEL} (via {EVAL_PROVIDER})")
        print(f"{'#' * 80}")

        saved_files = []

        for model_config in models:
            model_name = model_config["name"]
            print(f"\n{'=' * 80}")
            print(f"MODEL: {model_name} ({model_config['id']})")
            print(f"{'=' * 80}")

            # Warm up once per model
            self.warmup_model(model_config)

            for domain in domains:
                test_cases = domain_tests[domain]
                print(f"\n  DOMAIN: {domain} ({len(test_cases)} tests)")

                results = self.benchmark_model_domain(model_config, test_cases, domain)

                # Save immediately after each domain completes
                saved_file = self._save_domain_results(model_name, domain, results)
                saved_files.append(saved_file)

        # Generate final summary report
        self._generate_summary_report(saved_files)

    def _generate_summary_report(self, result_files: list[Path]):
        """Generate summary report from all saved result files."""
        report_file = self.output_dir / f"{self.session_id}_summary.md"

        # Load all results
        all_results = []
        for f in result_files:
            with open(f) as fp:
                data = json.load(fp)
                for r in data["results"]:
                    r["_model_name"] = data["model_name"]
                    r["_domain"] = data["domain"]
                    all_results.append(r)

        # Aggregate stats
        model_stats: dict[str, dict] = {}
        for r in all_results:
            model_name = r["_model_name"]
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "total": 0,
                    "valid": 0,
                    "faithful": 0,
                    "complete": 0,
                    "errors": 0,
                    "total_time": 0.0,
                }

            stats = model_stats[model_name]
            stats["total"] += 1

            resp = r.get("full_response", {})
            metadata = resp.get("metadata", {})

            if resp.get("is_valid"):
                stats["valid"] += 1
            if metadata.get("is_faithful"):
                stats["faithful"] += 1
            if metadata.get("is_complete"):
                stats["complete"] += 1
            if resp.get("status") == "error" or resp.get("status") == "exception":
                stats["errors"] += 1

            stats["total_time"] += r.get("execution_time_seconds", 0)

        # Generate report
        lines = [
            "# HED Model Benchmark Summary",
            "",
            f"**Session**: {self.session_id}",
            f"**Date**: {datetime.now().isoformat()}",
            f"**Eval Model**: {EVAL_MODEL} (via {EVAL_PROVIDER})",
            "",
            "## Results",
            "",
            "| Model | Valid | Faithful | Complete | Avg Time | Errors |",
            "|-------|-------|----------|----------|----------|--------|",
        ]

        for model_name, stats in model_stats.items():
            n = stats["total"]
            valid_pct = stats["valid"] / n * 100 if n else 0
            faithful_pct = stats["faithful"] / n * 100 if n else 0
            complete_pct = stats["complete"] / n * 100 if n else 0
            avg_time = stats["total_time"] / n if n else 0

            lines.append(
                f"| {model_name} | {valid_pct:.0f}% | {faithful_pct:.0f}% | "
                f"{complete_pct:.0f}% | {avg_time:.1f}s | {stats['errors']} |"
            )

        lines.extend(["", "## Result Files", ""])
        for f in result_files:
            lines.append(f"- `{f.name}`")

        with open(report_file, "w") as f:
            f.write("\n".join(lines))

        print(f"\n{'=' * 80}")
        print(f"Summary report: {report_file}")
        print(f"{'=' * 80}")


def main():
    """Run the benchmark."""
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Run 'hedit init' or set the environment variable")
        sys.exit(1)

    # Output directory
    output_dir = Path(__file__).parent / "benchmark_results"

    # Create and run benchmark
    benchmark = ModelBenchmark(output_dir=output_dir)

    # Run with optional domain filter
    domains = None
    if len(sys.argv) > 1:
        domains = sys.argv[1].split(",")
        print(f"Filtering to domains: {domains}")

    benchmark.run_benchmark(domains=domains)


if __name__ == "__main__":
    main()
