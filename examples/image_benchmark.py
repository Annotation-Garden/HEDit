#!/usr/bin/env python3
"""Image annotation benchmark for HED annotation quality comparison.

This script benchmarks multiple LLM models on image annotation tasks using
NSD (Natural Scenes Dataset) images. The same vision model describes all images,
then different annotation models generate HED annotations.

Key Design:
- Same vision model (qwen/qwen3-vl-30b-a3b-instruct) for all image descriptions
- Same evaluation model (qwen/qwen3-235b-a22b-2507) via Cerebras for fair comparison
- Only the annotation model varies between benchmarks

Related GitHub Issues:
- #64: Explore alternative candidates for the default model

Usage:
    python examples/image_benchmark.py                    # Run all models
    python examples/image_benchmark.py --models 2         # Run first 2 models only
    python examples/image_benchmark.py --images 5         # Run first 5 images only
"""

import argparse
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

# Vision model - same for all benchmarks (describes the image)
VISION_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"

# Evaluation model - consistent across all benchmarks for fair comparison
EVAL_MODEL = "qwen/qwen3-235b-a22b-2507"
EVAL_PROVIDER = "Cerebras"

# Models to benchmark (annotation models only - vision is fixed)
MODELS_TO_BENCHMARK = [
    # Baseline: Current default (Cerebras - ultra fast, cheap)
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS-120B (baseline)",
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
    # Nemotron 3 Nano 30B A3B (NVIDIA)
    {
        "id": "nvidia/nemotron-3-nano-30b-a3b",
        "name": "Nemotron-3-Nano-30B",
        "provider": None,
        "category": "balanced",
    },
]


def discover_images() -> list[Path]:
    """Discover all images in examples/images/ directory."""
    images_dir = Path(__file__).parent / "images"
    if not images_dir.exists():
        return []

    image_files = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.png"))
    )
    return image_files


@dataclass
class ImageBenchmarkResult:
    """Result from a single image annotation benchmark."""

    image_name: str
    image_path: str
    model_id: str
    model_name: str
    description: str  # Vision model's description
    annotation: str
    is_valid: bool
    is_faithful: bool | None
    is_complete: bool | None
    validation_attempts: int
    validation_messages: list[str]
    evaluation_feedback: str
    assessment_feedback: str
    execution_time_seconds: float
    cli_command: str
    error: str | None = None


def run_hedit_annotate_image(
    image_path: str,
    model_id: str,
    provider: str | None = None,
) -> tuple[dict, str, float]:
    """Run hedit annotate-image CLI command."""
    cmd = [
        "hedit",
        "annotate-image",
        image_path,
        "--model",
        model_id,
        "--eval-model",
        EVAL_MODEL,
        "--eval-provider",
        EVAL_PROVIDER,
        "--schema",
        SCHEMA_VERSION,
        "--max-attempts",
        str(MAX_VALIDATION_ATTEMPTS),
        "-o",
        "json",
        "--standalone",
        "--assessment",
    ]

    if provider:
        cmd.extend(["--provider", provider])

    cmd_str = " ".join(cmd)

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENROUTER_API_KEY": OPENROUTER_API_KEY or ""},
    )
    execution_time = time.time() - start_time

    # Parse JSON from stdout (filter out debug messages)
    stdout_lines = result.stdout.strip().split("\n")

    filtered_lines = []
    for line in stdout_lines:
        if line.startswith("[WORKFLOW]"):
            continue
        if "Provider List" in line or "\x1b[" in line:
            continue
        if not filtered_lines and not line.strip():
            continue
        filtered_lines.append(line)

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


class ImageBenchmark:
    """Benchmark runner for comparing image annotation models."""

    WARMUP_IMAGE = None  # Will be set to first image

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(__file__).parent / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[ImageBenchmarkResult] = []

    def warmup_model(self, model_config: dict, image_path: Path) -> None:
        """Run a warm-up call to prime the cache."""
        model_id = model_config["id"]
        model_name = model_config["name"]
        provider = model_config.get("provider")

        print(f"  Warming up cache for {model_name}...")

        try:
            run_hedit_annotate_image(
                image_path=str(image_path),
                model_id=model_id,
                provider=provider,
            )
            print("  Cache warmed up successfully")
        except Exception as e:
            print(f"  Warning: Warmup failed: {e}")

    def benchmark_model(
        self,
        model_config: dict,
        images: list[Path],
    ) -> list[ImageBenchmarkResult]:
        """Run benchmark for a single model on all images."""
        model_id = model_config["id"]
        model_name = model_config["name"]
        provider = model_config.get("provider")

        print(f"\n{'=' * 80}")
        print(f"Benchmarking: {model_name} ({model_id})")
        print(f"Images: {len(images)}")
        print(f"{'=' * 80}")

        # Warm up with first image
        if images:
            self.warmup_model(model_config, images[0])

        results = []

        for i, image_path in enumerate(images, 1):
            print(f"\n  [{i}/{len(images)}] {image_path.name}")

            try:
                parsed, cmd_str, exec_time = run_hedit_annotate_image(
                    image_path=str(image_path),
                    model_id=model_id,
                    provider=provider,
                )

                metadata = parsed.get("metadata", {})
                description = parsed.get("description", "")
                annotation = parsed.get("hed_string", "")
                is_valid = parsed.get("is_valid", False)
                is_faithful = metadata.get("is_faithful")
                is_complete = metadata.get("is_complete")
                validation_attempts = metadata.get("validation_attempts", 0)
                validation_messages = parsed.get("validation_messages", [])
                evaluation_feedback = metadata.get("evaluation_feedback", "")
                assessment_feedback = metadata.get("assessment_feedback", "")

                print(
                    f"    Description: {description[:60]}..."
                    if description
                    else "    Description: [empty]"
                )
                print(
                    f"    Annotation: {annotation[:60]}..."
                    if annotation
                    else "    Annotation: [empty]"
                )
                print(f"    Valid: {is_valid}, Faithful: {is_faithful}, Complete: {is_complete}")
                print(f"    Time: {exec_time:.1f}s")

                error = None
                if parsed.get("status") == "error":
                    error = parsed.get("error", "Unknown error")
                    print(f"    ERROR: {error}")

                results.append(
                    ImageBenchmarkResult(
                        image_name=image_path.name,
                        image_path=str(image_path),
                        model_id=model_id,
                        model_name=model_name,
                        description=description,
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
                    ImageBenchmarkResult(
                        image_name=image_path.name,
                        image_path=str(image_path),
                        model_id=model_id,
                        model_name=model_name,
                        description="",
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

    def run_benchmark(
        self,
        models: list[dict] | None = None,
        images: list[Path] | None = None,
        max_models: int | None = None,
        max_images: int | None = None,
    ):
        """Run full benchmark across models and images."""
        models = models or MODELS_TO_BENCHMARK
        images = images or discover_images()

        if max_models:
            models = models[:max_models]
        if max_images:
            images = images[:max_images]

        print(f"\n{'#' * 80}")
        print("# IMAGE ANNOTATION BENCHMARK")
        print(f"# Date: {datetime.now().isoformat()}")
        print(f"# Models: {len(models)}")
        print(f"# Images: {len(images)}")
        print(f"# Vision Model: {VISION_MODEL}")
        print(f"# Eval Model: {EVAL_MODEL} (via {EVAL_PROVIDER})")
        print(f"{'#' * 80}")

        for model_config in models:
            model_results = self.benchmark_model(model_config, images)
            self.results.extend(model_results)

        self._save_results()
        self._generate_report()

    def _save_results(self):
        """Save benchmark results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"image_benchmark_{timestamp}.json"

        results_data = []
        for r in self.results:
            results_data.append(
                {
                    "image_name": r.image_name,
                    "image_path": r.image_path,
                    "model_id": r.model_id,
                    "model_name": r.model_name,
                    "description": r.description,
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
                    "vision_model": VISION_MODEL,
                    "eval_model": EVAL_MODEL,
                    "eval_provider": EVAL_PROVIDER,
                    "results": results_data,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_file}")

    def _generate_report(self):
        """Generate summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"image_report_{timestamp}.md"

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

        # Generate markdown report
        report_lines = [
            "# Image Annotation Benchmark Report",
            "",
            f"**Date**: {datetime.now().isoformat()}",
            f"**Schema Version**: {SCHEMA_VERSION}",
            f"**Vision Model**: {VISION_MODEL}",
            f"**Evaluation Model**: {EVAL_MODEL} (via {EVAL_PROVIDER})",
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
                "## Per-Image Results",
                "",
            ]
        )

        # Group by image
        images = {}
        for r in self.results:
            if r.image_name not in images:
                images[r.image_name] = []
            images[r.image_name].append(r)

        for image_name, results in images.items():
            report_lines.extend(
                [
                    f"### {image_name}",
                    "",
                    "| Model | Valid | Faithful | Complete | Annotation |",
                    "|-------|-------|----------|----------|------------|",
                ]
            )

            for r in results:
                annotation_preview = (
                    r.annotation[:50] + "..." if len(r.annotation) > 50 else r.annotation
                )
                annotation_preview = annotation_preview.replace("|", "\\|")
                report_lines.append(
                    f"| {r.model_name} | {r.is_valid} | {r.is_faithful} | {r.is_complete} | `{annotation_preview}` |"
                )

            report_lines.append("")

        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Report saved to: {report_file}")


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description="Image annotation benchmark")
    parser.add_argument("--models", type=int, help="Limit to first N models")
    parser.add_argument("--images", type=int, help="Limit to first N images")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Run 'hedit init' or set the environment variable")
        sys.exit(1)

    images = discover_images()
    if not images:
        print("ERROR: No images found in examples/images/")
        sys.exit(1)

    print(f"Found {len(images)} images")

    output_dir = Path(__file__).parent / "benchmark_results"
    benchmark = ImageBenchmark(output_dir=output_dir)
    benchmark.run_benchmark(max_models=args.models, max_images=args.images)


if __name__ == "__main__":
    main()
