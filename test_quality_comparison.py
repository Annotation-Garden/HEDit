#!/usr/bin/env python3
"""Quality comparison test for ultra-fast Cerebras models."""

import asyncio
import os
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
    exit(1)

# Test descriptions of varying complexity
TEST_DESCRIPTIONS = [
    {
        "name": "Simple Visual Event",
        "description": "A red circle appears on the screen",
    },
    {
        "name": "Complex Interaction",
        "description": "A red circle appears on the left side of the screen and the participant presses the left mouse button",
    },
    {
        "name": "Multiple Objects",
        "description": "A blue square and a yellow triangle appear simultaneously on the right side while a beep sound plays",
    },
    {
        "name": "Temporal Sequence",
        "description": "First a warning tone sounds, then after 2 seconds a green arrow pointing upward appears in the center",
    },
    {
        "name": "Error Event",
        "description": "The participant presses the wrong key and receives negative feedback with a red X appearing",
    },
]


async def test_model(model_name: str, description: str, test_name: str):
    """Test a single model with a description.

    Args:
        model_name: Model to test
        description: Description to annotate
        test_name: Name of the test

    Returns:
        Result dictionary
    """
    # Create LLMs
    annotation_llm = create_openrouter_llm(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
        provider="Cerebras",
    )

    # Use Qwen for evaluation (reliable)
    evaluation_llm = create_openrouter_llm(
        model="qwen/qwen3-235b-a22b-2507",
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
        provider="Cerebras",
    )

    # Create workflow
    schema_dir = Path.home() / "git/hed-schemas/schemas_latest_json"

    workflow = HedAnnotationWorkflow(
        llm=annotation_llm,
        evaluation_llm=evaluation_llm,
        assessment_llm=annotation_llm,
        feedback_llm=annotation_llm,
        schema_dir=schema_dir,
        validator_path=None,
        use_js_validator=False,
    )

    # Run workflow
    import time
    start = time.time()

    result = await workflow.run(
        input_description=description,
        schema_version="8.4.0",
        max_validation_attempts=3,
        run_assessment=False,
    )

    elapsed = time.time() - start

    return {
        "model": model_name,
        "test": test_name,
        "description": description,
        "annotation": result["current_annotation"],
        "is_valid": result["is_valid"],
        "is_faithful": result["is_faithful"],
        "attempts": result["validation_attempts"],
        "time": elapsed,
        "errors": result.get("validation_errors", []),
        "eval_feedback": result.get("evaluation_feedback", ""),
    }


async def main():
    """Run quality comparison tests."""
    print("\n" + "="*100)
    print("QUALITY COMPARISON: Cerebras Ultra-Fast Models")
    print("="*100)
    print("\nComparing:")
    print("  1. openai/gpt-oss-120b (GPT-OSS-120B)")
    print("  2. qwen/qwen3-235b-a22b-2507 (Qwen 3 235B)")
    print("\nBoth models running on Cerebras provider")

    models = [
        "openai/gpt-oss-120b",
        "qwen/qwen3-235b-a22b-2507",
    ]

    for test_case in TEST_DESCRIPTIONS:
        print("\n" + "="*100)
        print(f"TEST: {test_case['name']}")
        print("="*100)
        print(f"Description: \"{test_case['description']}\"")
        print()

        results = []

        for model in models:
            model_short = "GPT-OSS-120B" if "gpt-oss" in model else "Qwen 3 235B"
            print(f"\n{'-'*100}")
            print(f"Testing: {model_short}")
            print(f"{'-'*100}")

            try:
                result = await test_model(
                    model_name=model,
                    description=test_case["description"],
                    test_name=test_case["name"],
                )

                results.append(result)

                print(f"⏱️  Time: {result['time']:.2f}s")
                print(f"✅ Valid: {result['is_valid']}")
                print(f"✅ Faithful: {result['is_faithful']}")
                print(f"🔄 Attempts: {result['attempts']}")
                print(f"\n📝 Annotation:")
                print(f"   {result['annotation']}")

                if result['errors']:
                    print(f"\n❌ Validation Errors:")
                    for error in result['errors']:
                        print(f"   - {error}")

            except Exception as e:
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Compare results
        if len(results) == 2:
            print(f"\n{'='*100}")
            print(f"COMPARISON SUMMARY for: {test_case['name']}")
            print(f"{'='*100}")

            print(f"\nSpeed:")
            print(f"  GPT-OSS-120B: {results[0]['time']:.2f}s")
            print(f"  Qwen 3 235B:  {results[1]['time']:.2f}s")

            fastest = results[0] if results[0]['time'] < results[1]['time'] else results[1]
            fastest_name = "GPT-OSS-120B" if "gpt-oss" in fastest['model'] else "Qwen 3 235B"
            print(f"  → Faster: {fastest_name}")

            print(f"\nQuality:")
            print(f"  GPT-OSS-120B: Valid={results[0]['is_valid']}, Faithful={results[0]['is_faithful']}")
            print(f"  Qwen 3 235B:  Valid={results[1]['is_valid']}, Faithful={results[1]['is_faithful']}")

            # Check if annotations are similar
            ann1_tags = set(results[0]['annotation'].split(', '))
            ann2_tags = set(results[1]['annotation'].split(', '))

            common_tags = ann1_tags & ann2_tags
            unique_to_1 = ann1_tags - ann2_tags
            unique_to_2 = ann2_tags - ann1_tags

            print(f"\nAnnotation Overlap:")
            print(f"  Common tags: {len(common_tags)}")
            print(f"  Unique to GPT-OSS-120B: {len(unique_to_1)}")
            print(f"  Unique to Qwen 3 235B: {len(unique_to_2)}")

            if unique_to_1:
                print(f"\n  Tags only in GPT-OSS-120B:")
                for tag in sorted(unique_to_1)[:5]:  # Show first 5
                    print(f"    - {tag}")

            if unique_to_2:
                print(f"\n  Tags only in Qwen 3 235B:")
                for tag in sorted(unique_to_2)[:5]:  # Show first 5
                    print(f"    - {tag}")

    print("\n" + "="*100)
    print("QUALITY COMPARISON COMPLETE")
    print("="*100)


if __name__ == "__main__":
    asyncio.run(main())
