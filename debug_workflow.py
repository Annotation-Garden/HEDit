"""Debug script to find where the workflow hangs."""
import asyncio
import sys
from pathlib import Path

# Add detailed logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from langchain_community.chat_models import ChatOllama
from src.agents.workflow import HedAnnotationWorkflow

async def debug_workflow():
    """Test workflow with debug logging."""
    print("=" * 80)
    print("STARTING WORKFLOW DEBUG")
    print("=" * 80)

    # Initialize LLM
    print("\n[1] Initializing LLM...")
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.2:1b",  # Small model for faster testing
        temperature=0.1,
    )
    print("✓ LLM initialized")

    # Test LLM
    print("\n[2] Testing LLM connection...")
    try:
        response = await llm.ainvoke("Say 'hello' and nothing else")
        print(f"✓ LLM response: {response.content[:100]}")
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        return

    # Initialize workflow
    print("\n[3] Initializing workflow...")
    try:
        workflow = HedAnnotationWorkflow(
            llm=llm,
            schema_dir=Path("/home/yahya/git/hed-schemas/schemas_latest_json"),  # Local path
            validator_path=None,  # No JS validator for local testing
            use_js_validator=False,  # Disable JS validator for local testing
        )
        print("✓ Workflow initialized")
    except Exception as e:
        print(f"✗ Workflow initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run workflow with timeout
    print("\n[4] Running workflow...")
    print("Input: 'The participant pressed a button'")

    try:
        # Set a timeout and recursion limit
        result = await asyncio.wait_for(
            workflow.run(
                input_description="The participant pressed a button",
                schema_version="8.4.0",  # Updated schema version
                max_validation_attempts=2,
                config={"recursion_limit": 50},  # Increase for debugging
            ),
            timeout=60.0  # 60 second timeout for debugging
        )
        print("\n✓ Workflow completed!")
        print(f"Result: {result}")
    except asyncio.TimeoutError:
        print("\n✗ WORKFLOW TIMED OUT after 30 seconds")
        print("This indicates the workflow is hanging somewhere")
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting async debug...")
    asyncio.run(debug_workflow())
