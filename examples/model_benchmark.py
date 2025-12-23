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

from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SCHEMA_VERSION = "8.4.0"
MAX_VALIDATION_ATTEMPTS = 5

# Models to benchmark
# Ordered by expected cost (lowest to highest)
MODELS_TO_BENCHMARK = [
    # Current default (Cerebras - ultra fast, cheap)
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS-120B",
        "provider": "Cerebras",
        "category": "fast",
    },
    # Qwen large (Cerebras - fast)
    {
        "id": "qwen/qwen3-235b-a22b-2507",
        "name": "Qwen3-235B",
        "provider": "Cerebras",
        "category": "fast",
    },
    # Claude Haiku (cheap, cacheable)
    {
        "id": "anthropic/claude-3.5-haiku",
        "name": "Claude Haiku 3.5",
        "provider": None,  # OpenRouter auto-routes
        "category": "balanced",
    },
    # GPT-4o-mini (OpenAI's cheap option)
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o-mini",
        "provider": None,
        "category": "balanced",
    },
    # Claude Sonnet (high quality, cacheable)
    {
        "id": "anthropic/claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "provider": None,
        "category": "quality",
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


# Domain 4: Image Annotations (using NSD images)
# Images are dynamically discovered from examples/images/ directory
def _discover_image_tests() -> list[TestCase]:
    """Dynamically discover image test cases from examples/images/ directory.

    Returns:
        List of TestCase objects for each image found
    """
    images_dir = Path(__file__).parent / "images"
    if not images_dir.exists():
        return []

    image_tests = []
    image_files = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.png"))
    )

    for i, img_path in enumerate(image_files, 1):
        image_tests.append(
            TestCase(
                id=f"img_{i:02d}",
                domain="image",
                description=str(img_path),  # Full path for CLI
                expected_elements=["Sensory-event", "Visual-presentation"],
                difficulty="hard",
                notes=f"NSD image: {img_path.stem}",
            )
        )

    return image_tests


IMAGE_TESTS = _discover_image_tests()

ALL_TEST_CASES = COGNITIVE_TESTS + ANIMAL_TESTS + PARADIGM_TESTS + IMAGE_TESTS


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    test_id: str
    model_id: str
    model_name: str
    domain: str
    input_description: str
    annotation: str
    is_valid: bool
    is_faithful: bool | None
    is_complete: bool | None
    validation_attempts: int
    validation_messages: list[str]
    evaluation_feedback: str
    assessment_feedback: str
    execution_time_seconds: float
    cli_command: str  # The exact CLI command used (for reproducibility)
    error: str | None = None


def run_hedit_annotate(
    description: str,
    model_id: str,
    provider: str | None = None,
    schema_version: str = "8.4.0",
    max_attempts: int = 5,
    run_assessment: bool = True,
) -> tuple[dict, str, float]:
    """Run hedit annotate CLI command.

    Args:
        description: Natural language event description
        model_id: Model ID (e.g., "openai/gpt-oss-120b")
        provider: Provider preference (e.g., "Cerebras")
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

    # Parse JSON from stdout (filter out workflow debug messages)
    stdout_lines = result.stdout.strip().split("\n")
    json_lines = [line for line in stdout_lines if not line.startswith("[WORKFLOW]")]
    json_str = "\n".join(json_lines)

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


def run_hedit_annotate_image(
    image_path: str,
    model_id: str,
    provider: str | None = None,
    schema_version: str = "8.4.0",
    max_attempts: int = 5,
    run_assessment: bool = True,
) -> tuple[dict, str, float]:
    """Run hedit annotate-image CLI command.

    Args:
        image_path: Path to image file
        model_id: Model ID
        provider: Provider preference
        schema_version: HED schema version
        max_attempts: Maximum validation attempts
        run_assessment: Whether to run completeness assessment

    Returns:
        Tuple of (parsed JSON result, CLI command string, execution time)
    """
    # Build CLI command
    cmd = [
        "hedit",
        "annotate-image",
        image_path,
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

    if provider:
        cmd.extend(["--provider", provider])

    if run_assessment:
        cmd.append("--assessment")

    # Command string for logging
    cmd_str = " ".join(cmd)

    # Run command
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENROUTER_API_KEY": OPENROUTER_API_KEY or ""},
    )
    execution_time = time.time() - start_time

    # Parse JSON from stdout
    stdout_lines = result.stdout.strip().split("\n")
    json_lines = [line for line in stdout_lines if not line.startswith("[WORKFLOW]")]
    json_str = "\n".join(json_lines)

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
    """Benchmark runner for comparing HED annotation models using CLI."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(__file__).parent / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def benchmark_model(
        self,
        model_config: dict,
        test_cases: list[TestCase],
    ) -> list[BenchmarkResult]:
        """Run benchmark for a single model using CLI.

        Args:
            model_config: Model configuration dict
            test_cases: List of test cases to run

        Returns:
            List of benchmark results
        """
        model_id = model_config["id"]
        model_name = model_config["name"]
        provider = model_config.get("provider")

        print(f"\n{'=' * 80}")
        print(f"Benchmarking: {model_name} ({model_id})")
        print(f"{'=' * 80}")

        results = []

        for test_case in test_cases:
            print(f"\n  Test: {test_case.id} ({test_case.domain})")
            print(f"    Difficulty: {test_case.difficulty}")

            try:
                # Determine if this is an image test
                is_image_test = test_case.domain == "image"

                if is_image_test:
                    # Use annotate-image for image tests
                    image_path = test_case.description
                    print(f"    Image: {Path(image_path).name}")

                    parsed, cmd_str, exec_time = run_hedit_annotate_image(
                        image_path=image_path,
                        model_id=model_id,
                        provider=provider,
                        schema_version=SCHEMA_VERSION,
                        max_attempts=MAX_VALIDATION_ATTEMPTS,
                        run_assessment=True,
                    )
                    # For image tests, get the description from the result
                    description = parsed.get("description", f"[Image: {Path(image_path).name}]")
                else:
                    # Use annotate for text tests
                    description = test_case.description
                    print(f"    Description: {description[:80]}...")

                    parsed, cmd_str, exec_time = run_hedit_annotate(
                        description=description,
                        model_id=model_id,
                        provider=provider,
                        schema_version=SCHEMA_VERSION,
                        max_attempts=MAX_VALIDATION_ATTEMPTS,
                        run_assessment=True,
                    )

                # Extract results from JSON
                metadata = parsed.get("metadata", {})
                annotation = parsed.get("hed_string", "")
                is_valid = parsed.get("is_valid", False)
                is_faithful = metadata.get("is_faithful")
                is_complete = metadata.get("is_complete")
                validation_attempts = metadata.get("validation_attempts", 0)
                validation_messages = parsed.get("validation_messages", [])
                evaluation_feedback = metadata.get("evaluation_feedback", "")
                assessment_feedback = metadata.get("assessment_feedback", "")

                print(
                    f"    Annotation: {annotation[:80]}..."
                    if annotation
                    else "    Annotation: [empty]"
                )
                print(f"    Valid: {is_valid}, Faithful: {is_faithful}, Complete: {is_complete}")
                print(f"    Attempts: {validation_attempts}, Time: {exec_time:.2f}s")

                error = None
                if parsed.get("status") == "error":
                    error = parsed.get("error", "Unknown error")
                    print(f"    ERROR: {error}")

                results.append(
                    BenchmarkResult(
                        test_id=test_case.id,
                        model_id=model_id,
                        model_name=model_name,
                        domain=test_case.domain,
                        input_description=description,
                        annotation=annotation,
                        is_valid=is_valid,
                        is_faithful=is_faithful,
                        is_complete=is_complete,
                        validation_attempts=validation_attempts,
                        validation_messages=validation_messages,
                        evaluation_feedback=evaluation_feedback,
                        assessment_feedback=assessment_feedback,
                        execution_time_seconds=exec_time,
                        cli_command=cmd_str,
                        error=error,
                    )
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback

                traceback.print_exc()

                results.append(
                    BenchmarkResult(
                        test_id=test_case.id,
                        model_id=model_id,
                        model_name=model_name,
                        domain=test_case.domain,
                        input_description=test_case.description,
                        annotation="",
                        is_valid=False,
                        is_faithful=None,
                        is_complete=None,
                        validation_attempts=0,
                        validation_messages=[],
                        evaluation_feedback="",
                        assessment_feedback="",
                        execution_time_seconds=0,
                        cli_command="",
                        error=str(e),
                    )
                )

        return results

    def run_full_benchmark(
        self,
        models: list[dict] | None = None,
        test_cases: list[TestCase] | None = None,
        domains: list[str] | None = None,
    ):
        """Run full benchmark across all models and test cases.

        Args:
            models: List of model configs (defaults to MODELS_TO_BENCHMARK)
            test_cases: List of test cases (defaults to ALL_TEST_CASES)
            domains: Filter to specific domains (e.g., ["cognitive", "animal"])
        """
        models = models or MODELS_TO_BENCHMARK
        test_cases = test_cases or ALL_TEST_CASES

        # Filter by domain if specified
        if domains:
            test_cases = [tc for tc in test_cases if tc.domain in domains]

        print(f"\n{'#' * 80}")
        print("# HED MODEL BENCHMARK (CLI-based)")
        print(f"# Date: {datetime.now().isoformat()}")
        print(f"# Models: {len(models)}")
        print(f"# Test Cases: {len(test_cases)}")
        print(f"# Domains: { {tc.domain for tc in test_cases} }")
        print(f"{'#' * 80}")

        for model_config in models:
            model_results = self.benchmark_model(model_config, test_cases)
            self.results.extend(model_results)

        # Save results
        self._save_results()

        # Generate report
        self._generate_report()

    def _save_results(self):
        """Save benchmark results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{timestamp}.json"

        results_data = []
        for r in self.results:
            results_data.append(
                {
                    "test_id": r.test_id,
                    "model_id": r.model_id,
                    "model_name": r.model_name,
                    "domain": r.domain,
                    "input_description": r.input_description,
                    "annotation": r.annotation,
                    "is_valid": r.is_valid,
                    "is_faithful": r.is_faithful,
                    "is_complete": r.is_complete,
                    "validation_attempts": r.validation_attempts,
                    "validation_messages": r.validation_messages,
                    "evaluation_feedback": r.evaluation_feedback,
                    "assessment_feedback": r.assessment_feedback,
                    "execution_time_seconds": r.execution_time_seconds,
                    "cli_command": r.cli_command,
                    "error": r.error,
                }
            )

        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "schema_version": SCHEMA_VERSION,
                    "results": results_data,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_file}")

    def _generate_report(self):
        """Generate summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"report_{timestamp}.md"

        # Aggregate statistics by model
        model_stats = {}
        for r in self.results:
            if r.model_name not in model_stats:
                model_stats[r.model_name] = {
                    "model_id": r.model_id,
                    "total": 0,
                    "valid": 0,
                    "faithful": 0,
                    "complete": 0,
                    "total_attempts": 0,
                    "total_time": 0.0,
                    "errors": 0,
                    "by_domain": {},
                }

            stats = model_stats[r.model_name]
            stats["total"] += 1
            stats["valid"] += 1 if r.is_valid else 0
            stats["faithful"] += 1 if r.is_faithful else 0
            stats["complete"] += 1 if r.is_complete else 0
            stats["total_attempts"] += r.validation_attempts
            stats["total_time"] += r.execution_time_seconds
            if r.error:
                stats["errors"] += 1

            # Domain stats
            if r.domain not in stats["by_domain"]:
                stats["by_domain"][r.domain] = {
                    "total": 0,
                    "valid": 0,
                    "faithful": 0,
                    "complete": 0,
                }
            domain_stats = stats["by_domain"][r.domain]
            domain_stats["total"] += 1
            domain_stats["valid"] += 1 if r.is_valid else 0
            domain_stats["faithful"] += 1 if r.is_faithful else 0
            domain_stats["complete"] += 1 if r.is_complete else 0

        # Generate markdown report
        report_lines = [
            "# HED Model Benchmark Report",
            "",
            f"**Date**: {datetime.now().isoformat()}",
            f"**Schema Version**: {SCHEMA_VERSION}",
            f"**Total Tests**: {len(self.results)}",
            "",
            "## Summary",
            "",
            "| Model | Valid | Faithful | Complete | Avg Attempts | Avg Time | Errors |",
            "|-------|-------|----------|----------|--------------|----------|--------|",
        ]

        for model_name, stats in model_stats.items():
            valid_rate = stats["valid"] / stats["total"] * 100 if stats["total"] > 0 else 0
            faithful_rate = stats["faithful"] / stats["total"] * 100 if stats["total"] > 0 else 0
            complete_rate = stats["complete"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_attempts = stats["total_attempts"] / stats["total"] if stats["total"] > 0 else 0
            avg_time = stats["total_time"] / stats["total"] if stats["total"] > 0 else 0

            report_lines.append(
                f"| {model_name} | {valid_rate:.0f}% | {faithful_rate:.0f}% | "
                f"{complete_rate:.0f}% | {avg_attempts:.1f} | {avg_time:.1f}s | {stats['errors']} |"
            )

        report_lines.extend(
            [
                "",
                "## By Domain",
                "",
            ]
        )

        domains = {r.domain for r in self.results}
        for domain in sorted(domains):
            report_lines.extend(
                [
                    f"### {domain.title()}",
                    "",
                    "| Model | Valid | Faithful | Complete |",
                    "|-------|-------|----------|----------|",
                ]
            )

            for model_name, stats in model_stats.items():
                if domain in stats["by_domain"]:
                    d = stats["by_domain"][domain]
                    valid_rate = d["valid"] / d["total"] * 100 if d["total"] > 0 else 0
                    faithful_rate = d["faithful"] / d["total"] * 100 if d["total"] > 0 else 0
                    complete_rate = d["complete"] / d["total"] * 100 if d["total"] > 0 else 0
                    report_lines.append(
                        f"| {model_name} | {valid_rate:.0f}% | {faithful_rate:.0f}% | {complete_rate:.0f}% |"
                    )

            report_lines.append("")

        # Detailed results
        report_lines.extend(
            [
                "## Detailed Results",
                "",
            ]
        )

        for r in self.results:
            report_lines.extend(
                [
                    f"### {r.test_id} - {r.model_name}",
                    "",
                    f"**Domain**: {r.domain}",
                    f"**Input**: {r.input_description[:200]}{'...' if len(r.input_description) > 200 else ''}",
                    "",
                    f"**Annotation**: `{r.annotation}`",
                    "",
                    f"**Valid**: {r.is_valid} | **Faithful**: {r.is_faithful} | **Complete**: {r.is_complete}",
                    f"**Attempts**: {r.validation_attempts} | **Time**: {r.execution_time_seconds:.2f}s",
                    "",
                    "**CLI Command**:",
                    "```bash",
                    f"{r.cli_command}",
                    "```",
                    "",
                ]
            )

            if r.evaluation_feedback:
                # Truncate long feedback
                feedback = (
                    r.evaluation_feedback[:500] + "..."
                    if len(r.evaluation_feedback) > 500
                    else r.evaluation_feedback
                )
                report_lines.extend(
                    [
                        "**Evaluation Feedback**:",
                        "```",
                        f"{feedback}",
                        "```",
                        "",
                    ]
                )

            if r.assessment_feedback:
                feedback = (
                    r.assessment_feedback[:500] + "..."
                    if len(r.assessment_feedback) > 500
                    else r.assessment_feedback
                )
                report_lines.extend(
                    [
                        "**Assessment Feedback**:",
                        "```",
                        f"{feedback}",
                        "```",
                        "",
                    ]
                )

            if r.validation_messages:
                report_lines.extend(
                    [
                        "**Validation Messages**:",
                        *[f"- {m}" for m in r.validation_messages[:5]],
                        "",
                    ]
                )

            if r.error:
                report_lines.append(f"**Error**: {r.error}")
                report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Report saved to: {report_file}")


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

    benchmark.run_full_benchmark(domains=domains)


if __name__ == "__main__":
    main()
