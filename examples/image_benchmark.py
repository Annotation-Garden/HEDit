#!/usr/bin/env python3
"""Image annotation benchmark for HED annotation quality comparison.

This script benchmarks multiple LLM models on image annotation tasks using
NSD (Natural Scenes Dataset) images.

Design for resilience:
- For each image:
  1. Run vision model once to get description
  2. Run all annotation models on that same description
  3. Save results for that image immediately
- If a model fails, we don't lose results from other models or images
- Full JSON responses are captured

Usage:
    python examples/image_benchmark.py                    # Run all
    python examples/image_benchmark.py --models 2         # First 2 models
    python examples/image_benchmark.py --images 5         # First 5 images
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS-120B",
        "provider": "Cerebras",
        "category": "baseline",
    },
    {
        "id": "openai/gpt-5.2",
        "name": "GPT-5.2",
        "provider": None,
        "category": "quality",
    },
    {
        "id": "openai/gpt-5.1-codex-mini",
        "name": "GPT-5.1-Codex-Mini",
        "provider": None,
        "category": "balanced",
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o-mini",
        "provider": None,
        "category": "balanced",
    },
    {
        "id": "google/gemini-3-flash-preview",
        "name": "Gemini-3-Flash",
        "provider": None,
        "category": "fast",
    },
    {
        "id": "anthropic/claude-haiku-4.5",
        "name": "Claude-Haiku-4.5",
        "provider": None,
        "category": "balanced",
    },
    {
        "id": "mistralai/mistral-small-3.2-24b-instruct",
        "name": "Mistral-Small-3.2-24B",
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


def run_vision_description(image_path: Path) -> tuple[str, dict[str, Any]]:
    """Run vision model to get image description.

    This runs once per image, then all annotation models use the same description.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (description text, full response dict)
    """
    # Read and encode image
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    image_uri = f"data:{mime_type};base64,{image_data}"

    # Use the standalone vision agent
    try:
        from src.agents.vision_agent import VisionAgent
        from src.cli.config import get_machine_id
        from src.utils.openrouter_llm import create_openrouter_llm

        user_id = get_machine_id()

        vision_llm = create_openrouter_llm(
            model=VISION_MODEL,
            api_key=OPENROUTER_API_KEY,
            temperature=0.3,
            provider=None,
            user_id=user_id,
        )

        vision_agent = VisionAgent(llm=vision_llm)

        import asyncio

        async def _run():
            return await vision_agent.describe_image(image_data=image_uri)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import nest_asyncio

            nest_asyncio.apply()
            result = asyncio.get_event_loop().run_until_complete(_run())
        else:
            result = asyncio.run(_run())

        description = result.get("description", "")
        return description, result

    except Exception as e:
        return "", {"error": str(e), "status": "vision_failed"}


def run_hedit_annotate(
    description: str,
    model_id: str,
    provider: str | None = None,
) -> tuple[dict, str, float]:
    """Run hedit annotate CLI command."""
    cmd = [
        "hedit",
        "annotate",
        description,
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

    cmd_str = " ".join(cmd[:2]) + f' "{description[:50]}..."' + " " + " ".join(cmd[3:])

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
    """Benchmark runner with per-image incremental saving.

    Design:
    - For each image: run vision once, then all annotation models
    - Save after each image completes (all models)
    - If one model fails, other models still get saved
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(__file__).parent / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_image_results(
        self, image_name: str, description: str, vision_response: dict, results: list[dict]
    ) -> Path:
        """Save results for a single image (all models).

        Args:
            image_name: Image filename
            description: Vision model's description
            vision_response: Full response from vision model
            results: List of annotation results for each model

        Returns:
            Path to saved file
        """
        safe_name = image_name.replace("/", "-").replace(" ", "_").lower()
        safe_name = safe_name.rsplit(".", 1)[0]  # Remove extension
        filename = f"{self.session_id}_image_{safe_name}.json"
        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "image_name": image_name,
                    "description": description,
                    "vision_model": VISION_MODEL,
                    "vision_response": vision_response,
                    "schema_version": SCHEMA_VERSION,
                    "eval_model": EVAL_MODEL,
                    "eval_provider": EVAL_PROVIDER,
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"  [SAVED] {output_file.name}")
        return output_file

    def benchmark_image(
        self, image_path: Path, models: list[dict]
    ) -> tuple[str, dict, list[dict[str, Any]]]:
        """Benchmark all models on a single image.

        Args:
            image_path: Path to image
            models: List of model configs

        Returns:
            Tuple of (description, vision_response, list of model results)
        """
        print("\n  Getting description from vision model...")

        # Run vision model once
        description, vision_response = run_vision_description(image_path)

        if not description:
            print("    ERROR: Vision model failed")
            return "", vision_response, []

        print(f"    Description: {description[:80]}...")

        # Run all annotation models on this description
        results = []

        for model_config in models:
            model_id = model_config["id"]
            model_name = model_config["name"]
            provider = model_config.get("provider")

            print(f"\n    {model_name}:")

            try:
                parsed, cmd_str, exec_time = run_hedit_annotate(
                    description=description,
                    model_id=model_id,
                    provider=provider,
                )

                result = {
                    "model_id": model_id,
                    "model_name": model_name,
                    "provider": provider,
                    "cli_command": cmd_str,
                    "execution_time_seconds": exec_time,
                    "full_response": parsed,
                }

                # Extract key metrics for logging
                metadata = parsed.get("metadata", {})
                annotation = parsed.get("hed_string", "")
                is_valid = parsed.get("is_valid", False)
                is_faithful = metadata.get("is_faithful")
                is_complete = metadata.get("is_complete")

                print(f"      Valid: {is_valid}, Faithful: {is_faithful}, Complete: {is_complete}")
                if annotation:
                    print(f"      HED: {annotation[:50]}...")

                if parsed.get("status") == "error":
                    print(f"      ERROR: {parsed.get('error', 'Unknown')}")

                results.append(result)

            except Exception as e:
                print(f"      EXCEPTION: {e}")
                import traceback

                traceback.print_exc()

                results.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "provider": provider,
                        "cli_command": "",
                        "execution_time_seconds": 0,
                        "full_response": {"status": "exception", "error": str(e)},
                    }
                )

        return description, vision_response, results

    def run_benchmark(
        self,
        models: list[dict] | None = None,
        images: list[Path] | None = None,
        max_models: int | None = None,
        max_images: int | None = None,
    ):
        """Run benchmark with per-image incremental saving.

        Args:
            models: List of model configs
            images: List of image paths
            max_models: Limit number of models
            max_images: Limit number of images
        """
        models = models or MODELS_TO_BENCHMARK
        images = images or discover_images()

        if max_models:
            models = models[:max_models]
        if max_images:
            images = images[:max_images]

        print(f"\n{'#' * 80}")
        print("# IMAGE BENCHMARK (Per-Image Saving)")
        print(f"# Session: {self.session_id}")
        print(f"# Images: {len(images)}")
        print(f"# Models: {len(models)}")
        print(f"# Vision Model: {VISION_MODEL}")
        print(f"# Eval Model: {EVAL_MODEL} (via {EVAL_PROVIDER})")
        print(f"{'#' * 80}")

        saved_files = []

        for i, image_path in enumerate(images, 1):
            print(f"\n{'=' * 80}")
            print(f"IMAGE [{i}/{len(images)}]: {image_path.name}")
            print(f"{'=' * 80}")

            description, vision_response, results = self.benchmark_image(image_path, models)

            # Save immediately after each image
            if results:  # Only save if we got any results
                saved_file = self._save_image_results(
                    image_name=image_path.name,
                    description=description,
                    vision_response=vision_response,
                    results=results,
                )
                saved_files.append(saved_file)

        # Generate summary
        self._generate_summary_report(saved_files)

    def _generate_summary_report(self, result_files: list[Path]):
        """Generate summary report from all saved result files."""
        report_file = self.output_dir / f"{self.session_id}_image_summary.md"

        # Load all results
        all_results = []
        for f in result_files:
            with open(f) as fp:
                data = json.load(fp)
                image_name = data["image_name"]
                for r in data["results"]:
                    r["_image_name"] = image_name
                    all_results.append(r)

        # Aggregate stats by model
        model_stats: dict[str, dict] = {}
        for r in all_results:
            model_name = r["model_name"]
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
            if resp.get("status") in ("error", "exception"):
                stats["errors"] += 1

            stats["total_time"] += r.get("execution_time_seconds", 0)

        # Generate report
        lines = [
            "# Image Benchmark Summary",
            "",
            f"**Session**: {self.session_id}",
            f"**Date**: {datetime.now().isoformat()}",
            f"**Vision Model**: {VISION_MODEL}",
            f"**Eval Model**: {EVAL_MODEL} (via {EVAL_PROVIDER})",
            f"**Images**: {len(result_files)}",
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
