#!/usr/bin/env python3
"""Benchmark analysis script for HED annotation model comparison.

This script analyzes results from both text-based (model_benchmark.py) and
image-based (image_benchmark.py) benchmarks, producing comprehensive metrics
and visualizations.

Usage:
    python examples/analyze_benchmark.py
    python examples/analyze_benchmark.py --output-dir /path/to/save
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "GPT-OSS-120B": "#1f77b4",
    "GPT-5.2": "#ff7f0e",
    "GPT-5.1-Codex-Mini": "#2ca02c",
    "GPT-4o-mini": "#d62728",
    "Gemini-3-Flash": "#9467bd",
    "Claude-Haiku-4.5": "#8c564b",
    "Mistral-Small-3.2-24B": "#e377c2",
}

# OpenRouter pricing (USD per million tokens) - as of Dec 2025
# Source: https://openrouter.ai/models
MODEL_PRICING = {
    "GPT-OSS-120B": {"input": 0.039, "output": 0.19},
    "GPT-5.2": {"input": 1.75, "output": 14.0},
    "GPT-5.1-Codex-Mini": {"input": 0.25, "output": 2.0},
    "GPT-4o-mini": {"input": 0.15, "output": 0.60},
    "Gemini-3-Flash": {"input": 0.50, "output": 3.0},
    "Claude-Haiku-4.5": {"input": 0.80, "output": 4.0},
    "Mistral-Small-3.2-24B": {"input": 0.06, "output": 0.18},
}


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model."""

    name: str
    total_tests: int = 0
    successful: int = 0
    valid: int = 0
    faithful: int = 0
    complete: int = 0
    errors: int = 0
    total_time: float = 0.0
    total_attempts: int = 0
    times: list = field(default_factory=list)
    attempts_list: list = field(default_factory=list)
    error_types: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_tests * 100 if self.total_tests else 0

    @property
    def valid_rate(self) -> float:
        return self.valid / self.total_tests * 100 if self.total_tests else 0

    @property
    def faithful_rate(self) -> float:
        return self.faithful / self.total_tests * 100 if self.total_tests else 0

    @property
    def complete_rate(self) -> float:
        return self.complete / self.total_tests * 100 if self.total_tests else 0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total_tests if self.total_tests else 0

    @property
    def avg_attempts(self) -> float:
        return self.total_attempts / self.successful if self.successful else 0

    @property
    def median_time(self) -> float:
        return float(np.median(self.times)) if self.times else 0


def load_text_benchmark_results(results_dir: Path) -> dict:
    """Load results from text-based model benchmark."""
    results = defaultdict(lambda: defaultdict(list))

    # Pattern: {session}_{model}_{domain}.json
    for f in results_dir.glob("*_*_*.json"):
        if "image" in f.name or "summary" in f.name:
            continue

        try:
            with open(f) as fp:
                data = json.load(fp)

            model_name = data.get("model_name", "Unknown")
            domain = data.get("domain", "unknown")

            for result in data.get("results", []):
                results[model_name][domain].append(result)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {f.name}: {e}")

    return dict(results)


def load_image_benchmark_results(results_dir: Path) -> dict:
    """Load results from image-based benchmark."""
    results = defaultdict(list)

    # Pattern: {session}_image_{imagename}.json
    for f in results_dir.glob("*_image_*.json"):
        if "summary" in f.name:
            continue

        try:
            with open(f) as fp:
                data = json.load(fp)

            image_name = data.get("image_name", "Unknown")
            description = data.get("description", "")

            for result in data.get("results", []):
                result["_image_name"] = image_name
                result["_description"] = description
                result["_description_length"] = len(description)
                results[result.get("model_name", "Unknown")].append(result)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {f.name}: {e}")

    return dict(results)


def calculate_metrics(results: list) -> ModelMetrics:
    """Calculate metrics from a list of test results."""
    metrics = ModelMetrics(name="")

    for r in results:
        metrics.total_tests += 1
        time_s = r.get("execution_time_seconds", 0)
        metrics.times.append(time_s)
        metrics.total_time += time_s

        response = r.get("full_response", {})

        if response.get("status") == "success":
            metrics.successful += 1

            if response.get("is_valid"):
                metrics.valid += 1

            metadata = response.get("metadata", {})
            if metadata.get("is_faithful"):
                metrics.faithful += 1
            if metadata.get("is_complete"):
                metrics.complete += 1

            attempts = metadata.get("validation_attempts", 1)
            metrics.total_attempts += attempts
            metrics.attempts_list.append(attempts)
        else:
            metrics.errors += 1
            error = response.get("error", "Unknown error")
            if "RateLimitError" in error or "429" in error:
                metrics.error_types.append("rate_limit")
            elif "Recursion limit" in str(response.get("stderr", "")):
                metrics.error_types.append("recursion_limit")
            elif "JSON parse error" in error:
                metrics.error_types.append("json_parse")
            else:
                metrics.error_types.append("other")

    return metrics


def aggregate_model_metrics(text_results: dict, image_results: dict) -> tuple[dict, dict, dict]:
    """Aggregate metrics by model for text, image, and combined."""
    text_metrics = {}
    image_metrics = {}
    combined_metrics = {}

    all_models = set(text_results.keys()) | set(image_results.keys())

    for model in all_models:
        # Text metrics
        if model in text_results:
            all_text_results = []
            for domain_results in text_results[model].values():
                all_text_results.extend(domain_results)
            m = calculate_metrics(all_text_results)
            m.name = model
            text_metrics[model] = m

        # Image metrics
        if model in image_results:
            m = calculate_metrics(image_results[model])
            m.name = model
            image_metrics[model] = m

        # Combined
        all_results = []
        if model in text_results:
            for domain_results in text_results[model].values():
                all_results.extend(domain_results)
        if model in image_results:
            all_results.extend(image_results[model])

        if all_results:
            m = calculate_metrics(all_results)
            m.name = model
            combined_metrics[model] = m

    return text_metrics, image_metrics, combined_metrics


def plot_success_comparison(text_metrics: dict, image_metrics: dict, output_dir: Path):
    """Plot success rate comparison between text and image benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = sorted(
        set(text_metrics.keys()) | set(image_metrics.keys()),
        key=lambda m: combined_metrics.get(m, ModelMetrics(m)).success_rate,
        reverse=True,
    )

    x = np.arange(len(models))
    width = 0.35

    text_rates = [text_metrics.get(m, ModelMetrics(m)).success_rate for m in models]
    image_rates = [image_metrics.get(m, ModelMetrics(m)).success_rate for m in models]

    bars1 = ax.bar(
        x - width / 2, text_rates, width, label="Text Benchmark", color="#2ca02c", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, image_rates, width, label="Image Benchmark", color="#1f77b4", alpha=0.8
    )

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Model Success Rate: Text vs Image Benchmarks", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_comparison.png", dpi=150)
    plt.close()


def plot_quality_metrics(combined_metrics: dict, output_dir: Path):
    """Plot quality metrics (valid, faithful, complete) by model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = sorted(
        combined_metrics.keys(),
        key=lambda m: combined_metrics[m].faithful_rate,
        reverse=True,
    )

    x = np.arange(len(models))
    width = 0.25

    valid_rates = [combined_metrics[m].valid_rate for m in models]
    faithful_rates = [combined_metrics[m].faithful_rate for m in models]
    complete_rates = [combined_metrics[m].complete_rate for m in models]

    ax.bar(x - width, valid_rates, width, label="Valid", color="#2ca02c", alpha=0.8)
    ax.bar(x, faithful_rates, width, label="Faithful", color="#1f77b4", alpha=0.8)
    ax.bar(x + width, complete_rates, width, label="Complete", color="#ff7f0e", alpha=0.8)

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Annotation Quality Metrics by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / "quality_metrics.png", dpi=150)
    plt.close()


def plot_time_vs_quality(combined_metrics: dict, output_dir: Path):
    """Scatter plot: execution time vs quality (faithful rate)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model, m in combined_metrics.items():
        color = COLORS.get(model, "#333333")
        size = m.total_tests * 5  # Size proportional to sample size
        ax.scatter(
            m.avg_time,
            m.faithful_rate,
            s=size,
            c=color,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
            label=f"{model} (n={m.total_tests})",
        )

    ax.set_xlabel("Average Execution Time (seconds)", fontsize=12)
    ax.set_ylabel("Faithful Rate (%)", fontsize=12)
    ax.set_title("Speed vs Quality Trade-off", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 105)

    # Add trend annotation
    times = [m.avg_time for m in combined_metrics.values()]
    rates = [m.faithful_rate for m in combined_metrics.values()]
    if len(times) > 2:
        correlation = np.corrcoef(times, rates)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_quality.png", dpi=150)
    plt.close()


def plot_execution_time_distribution(combined_metrics: dict, output_dir: Path):
    """Box plot of execution time distribution by model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = sorted(
        combined_metrics.keys(),
        key=lambda m: combined_metrics[m].median_time,
    )

    times_data = [combined_metrics[m].times for m in models]
    colors = [COLORS.get(m, "#333333") for m in models]

    bp = ax.boxplot(times_data, labels=models, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title("Execution Time Distribution by Model", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "execution_time_distribution.png", dpi=150)
    plt.close()


def plot_validation_attempts(combined_metrics: dict, output_dir: Path):
    """Plot average validation attempts by model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = sorted(
        combined_metrics.keys(),
        key=lambda m: combined_metrics[m].avg_attempts,
    )

    attempts = [combined_metrics[m].avg_attempts for m in models]
    colors = [COLORS.get(m, "#333333") for m in models]

    bars = ax.barh(models, attempts, color=colors, alpha=0.8)

    ax.set_xlabel("Average Validation Attempts", fontsize=12)
    ax.set_title("Average Iterations to Produce Valid Annotation", fontsize=14, fontweight="bold")
    ax.axvline(x=1, color="green", linestyle="--", alpha=0.5, label="First-pass success")

    # Add value labels
    for bar, val in zip(bars, attempts, strict=False):
        ax.text(
            val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(output_dir / "validation_attempts.png", dpi=150)
    plt.close()


def plot_error_analysis(combined_metrics: dict, output_dir: Path):
    """Plot error types by model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = [m for m in combined_metrics.keys() if combined_metrics[m].errors > 0]

    if not models:
        print("No errors to plot")
        return

    error_types = ["rate_limit", "recursion_limit", "json_parse", "other"]
    error_labels = ["Rate Limit", "Recursion Limit", "JSON Parse", "Other"]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#7f7f7f"]

    x = np.arange(len(models))
    width = 0.2
    offsets = np.arange(len(error_types)) - len(error_types) / 2 + 0.5

    for i, (etype, label, color) in enumerate(zip(error_types, error_labels, colors, strict=False)):
        counts = []
        for m in models:
            count = combined_metrics[m].error_types.count(etype)
            counts.append(count)
        if any(c > 0 for c in counts):
            ax.bar(x + offsets[i] * width, counts, width, label=label, color=color, alpha=0.8)

    ax.set_ylabel("Error Count", fontsize=12)
    ax.set_title("Error Types by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150)
    plt.close()


def plot_cost_quality_comparison(combined_metrics: dict, output_dir: Path):
    """Plot token efficiency vs quality comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Token estimates per attempt (~12K input + 300 output)
    TOKENS_PER_ATTEMPT = 12300

    models = [m for m in combined_metrics.keys() if combined_metrics[m].successful > 0]

    # Calculate tokens per annotation
    tokens = [TOKENS_PER_ATTEMPT * combined_metrics[m].avg_attempts for m in models]
    faithful_rates = [combined_metrics[m].faithful_rate for m in models]
    colors = [COLORS.get(m, "#333333") for m in models]

    # Left plot: Tokens vs Quality scatter
    ax1 = axes[0]
    for i, model in enumerate(models):
        ax1.scatter(
            tokens[i] / 1000,  # Convert to K tokens
            faithful_rates[i],
            s=200,
            c=colors[i],
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
            label=model,
        )
        # Add model name annotation
        ax1.annotate(
            model.split("-")[0],  # Shorter label
            (tokens[i] / 1000, faithful_rates[i]),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=9,
            va="center",
        )

    ax1.set_xlabel("Tokens per Annotation (K)", fontsize=12)
    ax1.set_ylabel("Faithful Rate (%)", fontsize=12)
    ax1.set_title("Token Usage vs Quality", fontsize=14, fontweight="bold")
    ax1.set_ylim(50, 105)
    ax1.axhline(y=95, color="green", linestyle="--", alpha=0.5, label="95% threshold")

    # Right plot: Token efficiency (quality per K tokens)
    ax2 = axes[1]
    # Calculate quality per K tokens (faithful_rate / (tokens/1000))
    efficiency = [
        (faithful_rates[i] / (tokens[i] / 1000)) if tokens[i] > 0 else 0 for i in range(len(models))
    ]

    # Sort by efficiency
    sorted_data = sorted(zip(efficiency, models, colors, strict=False), reverse=True)
    eff_sorted, models_sorted, colors_sorted = zip(*sorted_data, strict=False)

    bars = ax2.barh(models_sorted, eff_sorted, color=colors_sorted, alpha=0.8)
    ax2.set_xlabel("Quality per K Tokens (Faithful% / K tokens)", fontsize=12)
    ax2.set_title("Token Efficiency Ranking", fontsize=14, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, eff_sorted, strict=False):
        ax2.text(
            val + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(output_dir / "cost_quality_comparison.png", dpi=150)
    plt.close()


def plot_comprehensive_comparison(combined_metrics: dict, output_dir: Path):
    """Single comprehensive figure comparing all key metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))

    models = sorted(
        combined_metrics.keys(),
        key=lambda m: combined_metrics[m].faithful_rate,
        reverse=True,
    )

    x = np.arange(len(models))
    width = 0.15

    # Normalize metrics to 0-100 scale for comparison
    max_time = max(combined_metrics[m].avg_time for m in models)
    max_cost = max(MODEL_PRICING.get(m, {"output": 1})["output"] for m in models)

    faithful = [combined_metrics[m].faithful_rate for m in models]
    complete = [combined_metrics[m].complete_rate for m in models]
    # Invert time and cost so higher is better
    time_score = [100 - (combined_metrics[m].avg_time / max_time * 100) for m in models]
    cost_score = [
        100 - (MODEL_PRICING.get(m, {"output": max_cost})["output"] / max_cost * 100)
        for m in models
    ]
    first_pass = [
        100 / combined_metrics[m].avg_attempts if combined_metrics[m].avg_attempts > 0 else 0
        for m in models
    ]

    ax.bar(x - 2 * width, faithful, width, label="Faithful %", color="#2ca02c", alpha=0.8)
    ax.bar(x - width, complete, width, label="Complete %", color="#1f77b4", alpha=0.8)
    ax.bar(x, time_score, width, label="Speed Score", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width, cost_score, width, label="Cost Score", color="#9467bd", alpha=0.8)
    ax.bar(x + 2 * width, first_pass, width, label="First-Pass Score", color="#d62728", alpha=0.8)

    ax.set_ylabel("Score (higher is better)", fontsize=12)
    ax.set_title("Comprehensive Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_comparison.png", dpi=150)
    plt.close()


def plot_domain_performance(text_results: dict, output_dir: Path):
    """Plot model performance by domain."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    domains = ["cognitive", "animal", "paradigm"]
    domain_titles = ["Cognitive", "Animal", "Paradigm"]

    for ax, domain, title in zip(axes, domains, domain_titles, strict=False):
        model_rates = []
        model_names = []

        for model, domain_results in text_results.items():
            if domain in domain_results:
                metrics = calculate_metrics(domain_results[domain])
                model_rates.append(metrics.faithful_rate)
                model_names.append(model)

        if model_rates:
            # Sort by rate
            sorted_pairs = sorted(zip(model_rates, model_names, strict=False), reverse=True)
            rates, names = zip(*sorted_pairs, strict=False)

            colors = [COLORS.get(n, "#333333") for n in names]
            ax.barh(names, rates, color=colors, alpha=0.8)

            ax.set_xlabel("Faithful Rate (%)")
            ax.set_title(f"{title} Domain", fontweight="bold")
            ax.set_xlim(0, 105)

    plt.suptitle("Model Performance by Domain", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "domain_performance.png", dpi=150)
    plt.close()


def plot_image_complexity_analysis(image_results: dict, output_dir: Path):
    """Analyze if description length affects performance."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model, results in image_results.items():
        desc_lengths = []
        success_flags = []

        for r in results:
            desc_lengths.append(r.get("_description_length", 0))
            response = r.get("full_response", {})
            success = response.get("status") == "success" and response.get("is_valid", False)
            success_flags.append(1 if success else 0)

        if desc_lengths:
            color = COLORS.get(model, "#333333")
            # Bin by description length
            bins = np.linspace(min(desc_lengths), max(desc_lengths), 5)
            bin_indices = np.digitize(desc_lengths, bins)

            bin_success_rates = []
            bin_centers = []
            for i in range(1, len(bins)):
                mask = bin_indices == i
                if np.any(mask):
                    rate = (
                        np.mean([success_flags[j] for j in range(len(success_flags)) if mask[j]])
                        * 100
                    )
                    bin_success_rates.append(rate)
                    bin_centers.append((bins[i - 1] + bins[i]) / 2)

            if bin_centers:
                ax.plot(bin_centers, bin_success_rates, "o-", label=model, color=color, alpha=0.7)

    ax.set_xlabel("Description Length (characters)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Success Rate vs Image Description Complexity", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / "image_complexity_analysis.png", dpi=150)
    plt.close()


def generate_summary_report(
    text_metrics: dict,
    image_metrics: dict,
    combined_metrics: dict,
    output_dir: Path,
):
    """Generate a comprehensive markdown summary report."""
    lines = [
        "# HED Model Benchmark Analysis Report",
        "",
        f"**Generated**: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## Executive Summary",
        "",
    ]

    # Find best models
    best_faithful = max(combined_metrics.items(), key=lambda x: x[1].faithful_rate)
    best_speed = min(
        combined_metrics.items(),
        key=lambda x: x[1].avg_time if x[1].successful > 0 else float("inf"),
    )
    best_efficiency = max(
        combined_metrics.items(),
        key=lambda x: x[1].faithful_rate / x[1].avg_time if x[1].avg_time > 0 else 0,
    )

    lines.extend(
        [
            f"- **Best Quality (Faithful Rate)**: {best_faithful[0]} ({best_faithful[1].faithful_rate:.1f}%)",
            f"- **Fastest**: {best_speed[0]} ({best_speed[1].avg_time:.1f}s avg)",
            f"- **Best Efficiency (Quality/Time)**: {best_efficiency[0]}",
            "",
            "## Combined Results (Text + Image)",
            "",
            "| Model | Tests | Success | Valid | Faithful | Complete | Avg Time | Avg Attempts |",
            "|-------|-------|---------|-------|----------|----------|----------|--------------|",
        ]
    )

    for model in sorted(
        combined_metrics.keys(), key=lambda m: combined_metrics[m].faithful_rate, reverse=True
    ):
        m = combined_metrics[model]
        lines.append(
            f"| {model} | {m.total_tests} | {m.success_rate:.0f}% | {m.valid_rate:.0f}% | "
            f"{m.faithful_rate:.0f}% | {m.complete_rate:.0f}% | {m.avg_time:.1f}s | {m.avg_attempts:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Text Benchmark Results",
            "",
            "| Model | Tests | Success | Faithful | Avg Time |",
            "|-------|-------|---------|----------|----------|",
        ]
    )

    for model in sorted(
        text_metrics.keys(), key=lambda m: text_metrics[m].faithful_rate, reverse=True
    ):
        m = text_metrics[model]
        lines.append(
            f"| {model} | {m.total_tests} | {m.success_rate:.0f}% | {m.faithful_rate:.0f}% | {m.avg_time:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## Image Benchmark Results",
            "",
            "| Model | Images | Success | Faithful | Avg Time |",
            "|-------|--------|---------|----------|----------|",
        ]
    )

    for model in sorted(
        image_metrics.keys(), key=lambda m: image_metrics[m].faithful_rate, reverse=True
    ):
        m = image_metrics[model]
        lines.append(
            f"| {model} | {m.total_tests} | {m.success_rate:.0f}% | {m.faithful_rate:.0f}% | {m.avg_time:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "### 1. Quality vs Speed Trade-off",
            "",
        ]
    )

    # Analyze correlation
    times = [m.avg_time for m in combined_metrics.values() if m.successful > 0]
    rates = [m.faithful_rate for m in combined_metrics.values() if m.successful > 0]
    if len(times) > 2:
        corr = np.corrcoef(times, rates)[0, 1]
        if corr > 0.3:
            lines.append(
                f"There is a positive correlation ({corr:.2f}) between execution time and quality, "
                "suggesting slower models tend to produce better annotations."
            )
        elif corr < -0.3:
            lines.append(
                f"There is a negative correlation ({corr:.2f}) between execution time and quality, "
                "meaning faster models can still produce high-quality annotations."
            )
        else:
            lines.append(
                f"Weak correlation ({corr:.2f}) between speed and quality, "
                "indicating these factors are largely independent."
            )

    lines.extend(
        [
            "",
            "### 2. Text vs Image Performance",
            "",
        ]
    )

    # Compare text vs image
    for model in combined_metrics.keys():
        text_rate = text_metrics.get(model, ModelMetrics(model)).faithful_rate
        image_rate = image_metrics.get(model, ModelMetrics(model)).faithful_rate
        diff = text_rate - image_rate
        if abs(diff) > 10:
            better = "text" if diff > 0 else "image"
            lines.append(f"- **{model}**: Performs {abs(diff):.0f}% better on {better} benchmarks")

    lines.extend(
        [
            "",
            "### 3. First-Pass Success",
            "",
        ]
    )

    first_pass = [
        (m, combined_metrics[m].avg_attempts)
        for m in combined_metrics
        if combined_metrics[m].successful > 0
    ]
    first_pass.sort(key=lambda x: x[1])

    for model, attempts in first_pass[:3]:
        lines.append(f"- **{model}**: {attempts:.2f} average attempts (lower is better)")

    lines.extend(
        [
            "",
            "### 4. Error Analysis",
            "",
        ]
    )

    for model, m in combined_metrics.items():
        if m.errors > 0:
            error_summary = ", ".join(
                [f"{e}({m.error_types.count(e)})" for e in set(m.error_types)]
            )
            lines.append(f"- **{model}**: {m.errors} errors - {error_summary}")

    # Token usage analysis section
    lines.extend(
        [
            "",
            "### 5. Token & Cost Analysis",
            "",
            "> **Note**: Token counts below are *estimated* based on ~12K input + 300 output per attempt.",
            "> Actual token usage varies by description complexity and model verbosity.",
            "> These estimates are for relative comparison only.",
            "",
            "**Estimated Token Usage per Annotation**:",
            "",
            "| Model | Avg Attempts | Est. Tokens/Annotation | Faithful % | Est. Tokens/Quality Point |",
            "|-------|--------------|------------------------|------------|---------------------------|",
        ]
    )

    # Token estimates per attempt
    INPUT_TOKENS_PER_ATTEMPT = 12000
    OUTPUT_TOKENS_PER_ATTEMPT = 300
    TOKENS_PER_ATTEMPT = INPUT_TOKENS_PER_ATTEMPT + OUTPUT_TOKENS_PER_ATTEMPT

    token_efficiency = []
    for model in sorted(combined_metrics.keys(), key=lambda m: combined_metrics[m].avg_attempts):
        m = combined_metrics[model]
        if m.successful > 0:
            total_tokens = TOKENS_PER_ATTEMPT * m.avg_attempts
            tokens_per_quality = (
                total_tokens / m.faithful_rate if m.faithful_rate > 0 else float("inf")
            )
            token_efficiency.append(
                (model, tokens_per_quality, m.faithful_rate, m.avg_attempts, total_tokens)
            )
            lines.append(
                f"| {model} | {m.avg_attempts:.2f} | {total_tokens:,.0f} | {m.faithful_rate:.0f}% | {tokens_per_quality:.0f} |"
            )
        else:
            lines.append(f"| {model} | N/A | N/A | N/A | N/A |")

    # Token efficiency ranking
    lines.extend(
        [
            "",
            "**Estimated Token Efficiency Ranking** (lower = better):",
            "",
        ]
    )

    token_efficiency.sort(key=lambda x: x[1])
    for i, (model, tpq, faithful, attempts, _total) in enumerate(token_efficiency, 1):
        lines.append(
            f"{i}. **{model}**: ~{tpq:.0f} est. tokens/quality point ({faithful:.0f}% faithful, {attempts:.2f} attempts)"
        )

    # Pricing reference
    lines.extend(
        [
            "",
            "**OpenRouter Pricing Reference** ($/M tokens, before caching):",
            "",
            "| Model | Input | Output | Notes |",
            "|-------|-------|--------|-------|",
        ]
    )

    for model in sorted(MODEL_PRICING.keys(), key=lambda m: MODEL_PRICING[m]["output"]):
        pricing = MODEL_PRICING[model]
        lines.append(f"| {model} | ${pricing['input']:.3f} | ${pricing['output']:.2f} | |")

    lines.append("")
    lines.append(
        "*Note: Anthropic offers ~90% caching discount, OpenAI ~75%. Actual costs depend on caching efficiency.*"
    )

    # Best token-efficient models
    lines.extend(
        [
            "",
            "**Best Estimated Token Efficiency** (100% quality with lowest est. token usage):",
            "",
        ]
    )

    # Find models with 100% faithful rate, sort by tokens
    perfect_models = [
        (m, tpq, faithful, attempts, total)
        for m, tpq, faithful, attempts, total in token_efficiency
        if faithful >= 99.9
    ]
    if perfect_models:
        perfect_models.sort(key=lambda x: x[4])  # Sort by total tokens
        best_token = perfect_models[0]
        worst_token = perfect_models[-1] if len(perfect_models) > 1 else None
        lines.append(
            f"- **{best_token[0]}**: ~{best_token[4]:,.0f} est. tokens/annotation (most efficient at 100% quality)"
        )
        if worst_token and worst_token[0] != best_token[0]:
            savings = worst_token[4] / best_token[4]
            lines.append(
                f"- Uses ~**{savings:.1f}x fewer tokens** than {worst_token[0]} at same quality (estimated)"
            )

    # Cost efficiency ranking
    lines.extend(
        [
            "",
            "**Cost Efficiency (Quality per Dollar)**:",
            "",
        ]
    )

    cost_efficiency = []
    for model, m in combined_metrics.items():
        if m.successful > 0:
            pricing = MODEL_PRICING.get(model, {"output": 1})
            efficiency = m.faithful_rate / pricing["output"]
            cost_efficiency.append((model, efficiency, pricing["output"]))

    cost_efficiency.sort(key=lambda x: x[1], reverse=True)
    for i, (model, eff, cost) in enumerate(cost_efficiency, 1):
        lines.append(f"{i}. **{model}**: {eff:.0f} quality points per $/M (${cost:.2f}/M output)")

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "Based on the analysis:",
            "",
        ]
    )

    # Find best cost-efficient model with good quality
    best_cost_efficient = None
    for model, _eff, _cost in cost_efficiency:
        if combined_metrics[model].faithful_rate >= 95:
            best_cost_efficient = model
            break

    # Recommendations based on findings
    if best_faithful[1].faithful_rate >= 95:
        lines.append(
            f"1. **For highest quality**: Use {best_faithful[0]} ({best_faithful[1].faithful_rate:.0f}% faithful)"
        )

    if best_cost_efficient:
        pricing = MODEL_PRICING.get(best_cost_efficient, {"output": 0})
        lines.append(
            f"2. **Best value**: Use **{best_cost_efficient}** (100% faithful at ${pricing['output']:.2f}/M output)"
        )

    if best_speed[1].avg_time < 20 and best_speed[1].faithful_rate >= 80:
        lines.append(
            f"3. **For fastest results**: Use {best_speed[0]} ({best_speed[1].avg_time:.1f}s, {best_speed[1].faithful_rate:.0f}% faithful)"
        )

    # Compare cheapest high-quality to expensive option
    if cost_efficiency:
        cheapest_good = cost_efficiency[0]
        most_expensive = min(cost_efficiency, key=lambda x: x[1])
        if cheapest_good[0] != most_expensive[0]:
            savings = (most_expensive[2] / cheapest_good[2]) if cheapest_good[2] > 0 else 0
            lines.append(
                f"4. **Cost savings**: {cheapest_good[0]} is **{savings:.0f}x cheaper** than {most_expensive[0]} with similar quality"
            )

    # Identify problematic models
    low_performers = [m for m, metrics in combined_metrics.items() if metrics.faithful_rate < 70]
    if low_performers:
        lines.append(f"5. **Avoid**: {', '.join(low_performers)} (below 70% faithful rate)")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `success_rate_comparison.png`: Text vs Image success rates",
            "- `quality_metrics.png`: Valid/Faithful/Complete rates by model",
            "- `time_vs_quality.png`: Speed-quality trade-off scatter plot",
            "- `cost_quality_comparison.png`: **Cost vs Quality analysis**",
            "- `comprehensive_comparison.png`: All metrics in one view",
            "- `execution_time_distribution.png`: Time distribution box plots",
            "- `validation_attempts.png`: Average iterations needed",
            "- `error_analysis.png`: Error types breakdown",
            "- `domain_performance.png`: Performance by test domain",
            "- `image_complexity_analysis.png`: Description length vs success",
            "",
        ]
    )

    report_path = output_dir / "benchmark_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze HED benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "benchmark_results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save analysis outputs (defaults to results-dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(exist_ok=True)

    print("Loading benchmark results...")
    text_results = load_text_benchmark_results(args.results_dir)
    image_results = load_image_benchmark_results(args.results_dir)

    print(f"  Text benchmark: {len(text_results)} models")
    print(f"  Image benchmark: {len(image_results)} models")

    print("\nCalculating metrics...")
    global combined_metrics  # For use in sorting functions
    text_metrics, image_metrics, combined_metrics = aggregate_model_metrics(
        text_results, image_results
    )

    print("\nGenerating visualizations...")
    plot_success_comparison(text_metrics, image_metrics, output_dir)
    print("  - success_rate_comparison.png")

    plot_quality_metrics(combined_metrics, output_dir)
    print("  - quality_metrics.png")

    plot_time_vs_quality(combined_metrics, output_dir)
    print("  - time_vs_quality.png")

    plot_execution_time_distribution(combined_metrics, output_dir)
    print("  - execution_time_distribution.png")

    plot_validation_attempts(combined_metrics, output_dir)
    print("  - validation_attempts.png")

    plot_error_analysis(combined_metrics, output_dir)
    print("  - error_analysis.png")

    plot_cost_quality_comparison(combined_metrics, output_dir)
    print("  - cost_quality_comparison.png")

    plot_comprehensive_comparison(combined_metrics, output_dir)
    print("  - comprehensive_comparison.png")

    if text_results:
        plot_domain_performance(text_results, output_dir)
        print("  - domain_performance.png")

    if image_results:
        plot_image_complexity_analysis(image_results, output_dir)
        print("  - image_complexity_analysis.png")

    print("\nGenerating report...")
    generate_summary_report(text_metrics, image_metrics, combined_metrics, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
