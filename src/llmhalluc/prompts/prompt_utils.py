"""
Prompt management module for LLM applications.

This module provides functionality to load and manage prompt templates
stored as separate Python files in the prompts directory.
"""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_prompt_template(prompt_name: str) -> str:
    """Load a prompt template by variable name from the prompts directory.

    Args:
        prompt_name: The name of the prompt variable to load (e.g., "QA_PROMPT_TEMPLATE")

    Returns:
        The prompt content as a string
    """
    # Get the directory containing this module (prompts directory)
    prompts_dir = Path(__file__).parent

    # Find all Python files in the prompts directory
    prompt_files = list(prompts_dir.glob("*.py"))

    # Remove __init__.py and this file from the search
    prompt_files = [
        f for f in prompt_files if f.name not in ["__init__.py", "prompt_utils.py"]
    ]

    # Search through each prompt file for the specified variable
    for prompt_file in prompt_files:
        # Get module name from filename (without .py extension)
        module_name = prompt_file.stem

        # Import the prompt module dynamically using relative import
        prompt_module = importlib.import_module(f".{module_name}", package=__package__)

        # Check if the module has the requested prompt variable
        if hasattr(prompt_module, prompt_name):
            return getattr(prompt_module, prompt_name)

    # If not found, raise an error
    raise AttributeError(
        f"Prompt variable '{prompt_name}' not found in any prompt module"
    )


def list_available_prompts() -> list[str]:
    """List all available prompt files in the prompts directory.

    Returns:
        A list of prompt names (without .py extension) that are available

    Example:
        >>> prompts = list_available_prompts()
        >>> print(prompts)
        ['BasePrompt', 'SystemPrompt', 'UserPrompt']
    """
    prompts_dir = Path(__file__).parent
    prompt_files = []

    for file_path in prompts_dir.glob("*.py"):
        # Skip __init__.py and this file itself
        if file_path.name not in ["__init__.py", "prompts.py", "prompt_utils.py"]:
            prompt_name = file_path.stem
            prompt_files.append(prompt_name)

    logger.info("Found %d available prompts: %s", len(prompt_files), prompt_files)
    return sorted(prompt_files)


def validate_prompt_structure(prompt_name: str) -> bool:
    """Validate that a prompt file has the correct structure.

    Args:
        prompt_name: The name of the prompt to validate

    Returns:
        True if the prompt structure is valid, False otherwise

    Example:
        >>> is_valid = validate_prompt_structure("BasePrompt")
        >>> print(is_valid)
        True
    """
    try:
        prompt_content = get_prompt(prompt_name)
        return isinstance(prompt_content, str) and len(prompt_content.strip()) > 0
    except Exception as e:
        logger.warning("Prompt validation failed for '%s': %s", prompt_name, e)
        return False
