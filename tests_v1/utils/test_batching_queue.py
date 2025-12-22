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

import torch

from llamafactory.v1.utils.batching_queue import DynamicBatchSizeBuffer, TextBatchingQueue


def create_sample(length: int):
    """Helper to create a mock sample with a specific token length."""
    return {"input_ids": torch.ones(length), "attention_mask": torch.ones(length)}


class TestDynamicBatchSizeBuffer:
    def test_append_and_token_count(self):
        buffer = DynamicBatchSizeBuffer()
        buffer.append(create_sample(10))
        buffer.append(create_sample(20))

        assert len(buffer) == 2
        assert buffer.total_token_count == 30

    def test_get_samples_within_budget(self):
        buffer = DynamicBatchSizeBuffer()
        buffer.append(create_sample(10))
        buffer.append(create_sample(10))
        buffer.append(create_sample(50))  # This one is large

        # Request 25 tokens. Should get the first two (20 tokens total)
        samples = buffer.get_samples(max_tokens_per_iteration=25)
        assert len(samples) == 2

    def test_force_return_first_sample(self):
        buffer = DynamicBatchSizeBuffer()
        buffer.append(create_sample(100))

        # Even though budget is 50, force=True (default) should return the 100-token sample
        samples = buffer.get_samples(max_tokens_per_iteration=50, force=True)
        assert len(samples) == 1
        assert len(samples[0]["input_ids"]) == 100

    def test_flush_removes_used_samples(self):
        buffer = DynamicBatchSizeBuffer()
        buffer.append(create_sample(10))
        buffer.append(create_sample(20))

        # Take the first sample
        buffer.get_samples(max_tokens_per_iteration=15)
        buffer.flush()

        assert len(buffer) == 1
        assert buffer.total_token_count == 20
        # The remaining sample should now be at the start
        remaining = buffer.get_samples(max_tokens_per_iteration=50)
        assert len(remaining[0]["input_ids"]) == 20


class TestTextBatchingQueue:
    def test_is_full_filled(self):
        queue = TextBatchingQueue(token_micro_bsz=100, buffer_size=2)

        queue.put_item(create_sample(10))
        assert not queue.is_full_filled()  # Only 1 sample, buffer_size=2

        queue.put_item(create_sample(10))
        assert not queue.is_full_filled()  # 2 samples, but only 20 tokens (min 100)

        queue.put_item(create_sample(90))
        assert queue.is_full_filled()  # Meets both conditions

    def test_warmup_logic(self):
        # token_micro_bsz=1000, starts at 200, reaches 1000 at step 10
        queue = TextBatchingQueue(token_micro_bsz=1000, bsz_warmup_steps=10, bsz_warmup_init_mbtoken=200)

        # Step 0: should be init value
        assert queue.get_cur_token_micro_bsz() == 200

        # Step 5: halfway through warmup (200 + (800 * 5/10)) = 600
        queue._step = 5
        assert queue.get_cur_token_micro_bsz() == 600

        # Step 11: past warmup
        queue._step = 11
        assert queue.get_cur_token_micro_bsz() == 1000

    def test_get_micro_batch_integration(self):
        queue = TextBatchingQueue(token_micro_bsz=50, buffer_size=1)
        queue.put_item(create_sample(20))
        queue.put_item(create_sample(20))
        queue.put_item(create_sample(20))

        # At step 0 (warmup not triggered as bsz_warmup_steps is -1 default),
        # it should take samples up to 50 tokens.
        batch = queue.get_micro_batch(step=0)

        assert len(batch) == 2
        assert queue.empty() is False

        batch_2 = queue.get_micro_batch(step=1)
        assert len(batch_2) == 1
        assert queue.empty() is True
