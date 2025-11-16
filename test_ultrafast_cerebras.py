#!/usr/bin/env python3
"""Benchmark ultra-fast models with Cerebras provider."""

import asyncio
import os
import time
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.agents.workflow import HedAnnotationWorkflow
from src.utils.openrouter_llm import create_openrouter_llm

# Get OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found in environment")
    print("Make sure you have a .env file with OPENROUTER_API_KEY set")
    exit(1)

# Test description
TEST_DESCRIPTION = "A red circle appears on the left side of the screen and the participant presses the left mouse button"

async def test_configuration(
    annotation_model: str,
    config_name: str,
    provider: str = "Cerebras",
):
    """Test a specific ultra-fast model configuration with Cerebras.

    Args:
        annotation_model: Model for annotation agent
        config_name: Name of this configuration
        provider: Provider to use (default: Cerebras)
    """
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"  Annotation Model: {annotation_model} (Cerebras)")
    print(f"  Evaluation Model: qwen/qwen3-235b-a22b-2507 (Cerebras)")
    print(f"  Feedback/Assessment: {annotation_model} (Cerebras)")
    print(f"  ALL MODELS RUNNING ON CEREBRAS FOR MAXIMUM SPEED!")
    print(f"{'='*80}\n")

    # Create LLMs with Cerebras provider
    annotation_llm = create_openrouter_llm(
        model=annotation_model,
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
        provider=provider,
    )

    # Use Qwen 3 235B for evaluation (it's fast and reliable)
    # Keep everything on Cerebras for maximum speed
    evaluation_llm = create_openrouter_llm(
        model="qwen/qwen3-235b-a22b-2507",
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
        provider=provider,
    )

    # Use annotation model for assessment and feedback
    assessment_llm = annotation_llm
    feedback_llm = annotation_llm

    # Create workflow
    schema_dir = Path.home() / "git/hed-schemas/schemas_latest_json"

    workflow = HedAnnotationWorkflow(
        llm=annotation_llm,
        evaluation_llm=evaluation_llm,
        assessment_llm=assessment_llm,
        feedback_llm=feedback_llm,
        schema_dir=schema_dir,
        validator_path=None,
        use_js_validator=False,
    )

    # Run test
    start_time = time.time()

    result = await workflow.run(
        input_description=TEST_DESCRIPTION,
        schema_version="8.4.0",
        max_validation_attempts=3,
        run_assessment=False,  # Skip assessment for maximum speed
    )

    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n✅ RESULTS:")
    print(f"  ⏱️  Time: {elapsed_time:.2f}s")
    print(f"  ✅ Valid: {result['is_valid']}")
    print(f"  ✅ Faithful: {result.get('is_faithful', 'N/A')}")
    print(f"  📝 Annotation: {result['current_annotation'][:150]}...")
    print(f"  🔄 Attempts: {result.get('validation_attempts', 'N/A')}")

    if result['validation_errors']:
        print(f"  ❌ Errors: {len(result['validation_errors'])}")
    else:
        print(f"  ✅ No validation errors!")

    return elapsed_time, result


async def main():
    """Run benchmarks for ultra-fast models."""
    print("\n" + "="*80)
    print("Ultra-Fast Models Benchmark (Cerebras Provider)")
    print("="*80)
    print("\nThese models achieve >1000 tokens/second with Cerebras:")
    print("- openai/gpt-oss-120b: 2,700-3,045 tokens/s")
    print("- qwen/qwen3-235b-a22b-2507: Ultra-fast Qwen model")

    configs = [
        {
            "name": "GPT-OSS-120B (Cerebras)",
            "model": "openai/gpt-oss-120b",
            "provider": "Cerebras",
        },
        {
            "name": "Qwen 3 235B (Cerebras)",
            "model": "qwen/qwen3-235b-a22b-2507",
            "provider": "Cerebras",
        },
    ]

    results = []

    for config in configs:
        try:
            elapsed, result = await test_configuration(
                annotation_model=config["model"],
                config_name=config["name"],
                provider=config["provider"],
            )
            results.append({
                "name": config["name"],
                "time": elapsed,
                "valid": result['is_valid'],
                "faithful": result.get('is_faithful'),
                "annotation": result['current_annotation'],
            })
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": config["name"],
                "time": None,
                "error": str(e),
            })

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for result in results:
        if result.get('time'):
            print(f"\n{result['name']}")
            print(f"  ⏱️  Time: {result['time']:.2f}s")
            print(f"  ✅ Valid: {result['valid']}")
            print(f"  ✅ Faithful: {result.get('faithful', 'N/A')}")
            if result.get('annotation'):
                print(f"  📝 Annotation: {result['annotation'][:100]}...")
        else:
            print(f"\n{result['name']}")
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

    print("\n" + "="*80)

    # Show speed comparison
    successful_results = [r for r in results if r.get('time')]
    if len(successful_results) > 0:
        print("\nSPEED COMPARISON:")
        for result in sorted(successful_results, key=lambda x: x['time']):
            print(f"  {result['name']}: {result['time']:.2f}s")

        fastest = min(successful_results, key=lambda x: x['time'])
        print(f"\n🏆 Fastest: {fastest['name']} at {fastest['time']:.2f}s")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
