#!/usr/bin/env python3
"""
Dependency checker script that tries to import all necessary dependencies.
"""

import logging
import sys


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_import(module_name):
    """Try to import a module and report if it's available."""
    try:
        __import__(module_name)
        logger.info(f"✓ {module_name} is available")
        return True
    except ImportError as e:
        logger.error(f"✗ {module_name} is NOT available: {e}")
        return False


def main():
    logger.info("Checking dependencies...")

    # Core dependencies
    core_deps = ["torch", "transformers", "peft", "datasets", "accelerate", "tqdm", "numpy", "pandas", "psutil"]

    # SGLang specific
    sglang_deps = ["sglang"]

    # Check core dependencies
    logger.info("Checking core dependencies...")
    core_ok = all(check_import(dep) for dep in core_deps)

    # Check SGLang dependencies
    logger.info("\nChecking SGLang dependencies...")
    sglang_ok = all(check_import(dep) for dep in sglang_deps)

    # Report summary
    logger.info("\nDependency check summary:")
    logger.info(f"Core dependencies: {'OK' if core_ok else 'MISSING'}")
    logger.info(f"SGLang dependencies: {'OK' if sglang_ok else 'MISSING'}")

    # Overall status
    if core_ok and sglang_ok:
        logger.info("\nAll dependencies are available!")
        return 0
    else:
        logger.error("\nSome dependencies are missing. Please install them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
