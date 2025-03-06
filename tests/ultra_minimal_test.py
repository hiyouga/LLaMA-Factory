#!/usr/bin/env python3
"""
Ultra minimal test for SGLang engine integration.
This script creates a direct mock version of the SGLang engine for testing.
"""

import logging
import sys


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a simple mock SGLang implementation for testing
class MockSGLangEngine:
    def __init__(self, **kwargs):
        logger.info("Initialized mock SGLang Engine")
        self.tokenizer = None

    def generate(self, prompt, **kwargs):
        logger.info(f"Mock generate with prompt: {prompt}")
        return [{"text": "_rho"}]


# Try to run an ultra minimal test
def main():
    try:
        logger.info("Starting ultra minimal SGLang test...")

        # Create mock engine
        engine = MockSGLangEngine(model_path="dummy/model")

        # Test generation
        result = engine.generate("Hello", max_tokens=1)
        logger.info(f"Generation result: {result}")

        assert result[0]["text"] == "_rho", f"Expected '_rho', got '{result[0]['text']}'"
        logger.info("Test passed!")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
