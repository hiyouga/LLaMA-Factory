#!/usr/bin/env python3
"""
Debug script for testing SGLang engine initialization in isolation.
This helps identify issues before running the full test suite.
"""

import logging
import os
import sys
import time

import torch


# Add the parent directory to the path to find the llamafactory module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variables for debugging
os.environ["SGLANG_ULTRA_DEBUG"] = "1"  # Enable ultra debug mode

# Optionally enable ultra minimal mode to bypass actual model loading
# os.environ["SGLANG_ULTRA_MINIMAL"] = "1"

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run a minimal test of SGLang engine initialization"""
    try:
        # Import inside the function to catch import errors
        import sglang

        from llamafactory.chat import ChatModel

        logger.info(f"SGLang version: {getattr(sglang, '__version__', 'unknown')}")

        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA available, cleared cache")

        # Basic arguments for a minimal test
        INFER_ARGS = {
            "model_name_or_path": "llamafactory/tiny-random-Llama-3",  # Use tiny model for testing
            "template": "llama3",
            "infer_dtype": "float16",
            "infer_backend": "sglang",  # Use the SGLang backend
            "do_sample": False,
            "max_new_tokens": 1,
            # Conservative memory settings
            "sglang_maxlen": 1024,  # Small context window
            "sglang_mem_fraction": 0.6,  # Conservative memory usage
            "sglang_config": {
                "device": "cuda",
                "chunked_prefill_size": 512,
                "max_running_requests": 1,
                "allow_auto_truncate": True,
            },
        }

        logger.info("Initializing ChatModel with SGLang backend...")
        start_time = time.time()
        chat_model = ChatModel(INFER_ARGS)
        init_time = time.time() - start_time
        logger.info(f"ChatModel initialized in {init_time:.2f} seconds")

        # Check that the model was initialized with the right backend
        logger.info(f"Engine type: {chat_model.engine_type}")

        # Try a simple chat query
        logger.info("Testing simple chat query...")
        message = [{"role": "user", "content": "Hi"}]
        response = chat_model.chat(message)[0]
        logger.info(f"Response: {response.response_text}")

        logger.info("All tests passed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in debug test: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
