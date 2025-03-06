# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time

import pytest


# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "")))

# Check if SGLang is installed
try:
    import importlib.util

    SGLANG_AVAILABLE = importlib.util.find_spec("sglang") is not None
except ImportError:
    SGLANG_AVAILABLE = False

# Import directly from src
from src.llamafactory.chat import ChatModel


TINY_LLAMA = os.getenv("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

# Basic inference args for SGLang testing
INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
    "infer_backend": "sglang",  # Use the SGLang backend
    "do_sample": False,
    "max_new_tokens": 1,
}

MESSAGES = [
    {"role": "user", "content": "Hi"},
]

EXPECTED_RESPONSE = "_rho"  # This should match the expected response for tiny-random model


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "../../src/llamafactory/chat/sglang_engine.py"))
    or not SGLANG_AVAILABLE,
    reason="SGLang engine file not found or SGLang is not installed",
)
def test_sglang_chat():
    """Test the SGLang engine's chat functionality"""
    try:
        # Initialize chat model with SGLang backend
        chat_model = ChatModel(INFER_ARGS)
        # Verify that the engine type is correctly set to "sglang"
        assert chat_model.engine_type == "sglang", f"Expected engine type 'sglang', got '{chat_model.engine_type}'"
        # Test basic chat functionality
        response = chat_model.chat(MESSAGES)[0]
        assert (
            response.response_text == EXPECTED_RESPONSE
        ), f"Expected '{EXPECTED_RESPONSE}', got '{response.response_text}'"
        print(f"Chat test passed! Response: {response.response_text}")
    except Exception as e:
        pytest.fail(f"SGLang chat test failed with error: {str(e)}")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "../../src/llamafactory/chat/sglang_engine.py"))
    or not SGLANG_AVAILABLE,
    reason="SGLang engine file not found or SGLang is not installed",
)
def test_sglang_stream_chat():
    """Test the SGLang engine's streaming chat functionality"""
    try:
        # Initialize chat model with SGLang backend
        chat_model = ChatModel(INFER_ARGS)
        # Test streaming chat functionality
        response = ""
        for token in chat_model.stream_chat(MESSAGES):
            response += token
            print(f"Token received: {token}")

        assert response == EXPECTED_RESPONSE, f"Expected '{EXPECTED_RESPONSE}', got '{response}'"
        print(f"Stream chat test passed! Complete response: {response}")
    except Exception as e:
        pytest.fail(f"SGLang stream chat test failed with error: {str(e)}")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "../../src/llamafactory/chat/sglang_engine.py"))
    or not SGLANG_AVAILABLE,
    reason="SGLang engine file not found or SGLang is not installed",
)
def test_sglang_batch_inference():
    """Test the SGLang engine's batch inference capability"""
    try:
        # Initialize chat model with SGLang backend
        chat_model = ChatModel(INFER_ARGS)

        # Create a batch of messages
        batch_messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi there"}],
            [{"role": "user", "content": "How are you?"}],
        ]

        # Test batch inference
        start_time = time.time()
        responses = []
        for msg in batch_messages:
            resp = chat_model.chat(msg)[0]
            responses.append(resp.response_text)
        batch_time = time.time() - start_time

        # Verify we got responses for all inputs
        assert len(responses) == len(batch_messages), f"Expected {len(batch_messages)} responses, got {len(responses)}"
        print(f"Batch inference test passed! Responses: {responses}")
        print(f"Batch inference time: {batch_time:.2f}s")
    except Exception as e:
        pytest.fail(f"SGLang batch inference test failed with error: {str(e)}")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "../../src/llamafactory/chat/sglang_engine.py"))
    or not SGLANG_AVAILABLE,
    reason="SGLang engine file not found or SGLang is not installed",
)
def test_sglang_get_scores():
    """Test the SGLang engine's scoring functionality"""
    try:
        # Initialize chat model with SGLang backend
        chat_model = ChatModel(INFER_ARGS)

        # Create test inputs for scoring
        test_inputs = [
            "This is a test sentence.",
            "Another test sentence for scoring.",
            "SGLang should be able to score this.",
        ]

        # Get scores for these inputs
        scores = chat_model.get_scores(test_inputs)

        # Verify we got scores for all inputs
        assert len(scores) == len(test_inputs), f"Expected {len(test_inputs)} scores, got {len(scores)}"
        # Scores should be floats
        assert all(isinstance(score, float) for score in scores), "All scores should be floats"

        print(f"Scoring test passed! Scores: {scores}")
    except Exception as e:
        pytest.fail(f"SGLang scoring test failed with error: {str(e)}")


# Run tests if executed directly
if __name__ == "__main__":
    if not SGLANG_AVAILABLE:
        print("SGLang is not available. Please install it with 'pip install sglang'")
        sys.exit(1)

    print("Testing SGLang engine...")
    test_sglang_chat()
    test_sglang_stream_chat()
    test_sglang_batch_inference()
    test_sglang_get_scores()
    print("All SGLang tests passed!")
