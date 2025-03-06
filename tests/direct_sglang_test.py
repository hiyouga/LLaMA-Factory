#!/usr/bin/env python3
"""
Direct test of the SGLang engine using imports from src directly.
This bypasses the package import system and tests the engine in ultra minimal mode.
"""

import asyncio
import logging
import os
import sys


# Set ultra minimal mode to bypass actual model loading
os.environ["SGLANG_ULTRA_MINIMAL"] = "1"
os.environ["SGLANG_ULTRA_DEBUG"] = "1"
os.environ["DISABLE_VERSION_CHECK"] = "1"  # Disable version checks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


async def run_async_test():
    """Run the actual async test"""
    try:
        logger.info("Starting async SGLang engine test...")

        # Import directly from src
        from src.llamafactory.chat.sglang_engine import SGLangEngine
        from src.llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

        # Create minimal arguments
        model_args = ModelArguments(model_name_or_path="dummy/model", infer_dtype="float16", infer_backend="sglang")

        data_args = DataArguments(template="llama3")

        finetuning_args = FinetuningArguments(stage="sft")

        generating_args = GeneratingArguments(do_sample=False, max_new_tokens=1)

        # Create the engine with ultra minimal mode
        logger.info("Initializing SGLangEngine in ultra minimal mode...")
        engine = SGLangEngine(model_args, data_args, finetuning_args, generating_args)

        # Dummy messages
        messages = [{"role": "user", "content": "Hi"}]

        # Test chat method (async)
        logger.info("Testing chat method...")
        response = await engine.chat(messages)
        logger.info(f"Chat response: {response}")

        # Test streaming chat method (async)
        logger.info("Testing streaming chat method...")
        stream_output = ""
        async for token in engine.stream_chat(messages):
            stream_output += token
            logger.info(f"Received token: {token}")
        logger.info(f"Complete streamed response: {stream_output}")

        # Test get_scores method (async)
        logger.info("Testing get_scores method...")
        scores = await engine.get_scores(["Test input"])
        logger.info(f"Scores: {scores}")

        logger.info("All async tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Async test failed: {e}", exc_info=True)
        return False


def main():
    """Run the async test using the event loop"""
    try:
        # Run the async test
        result = asyncio.run(run_async_test())
        return 0 if result else 1

    except Exception as e:
        logger.error(f"Failed to run async test: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
