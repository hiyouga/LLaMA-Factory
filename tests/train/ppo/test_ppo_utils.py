# Copyright 2026 the LlamaFactory team.
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

import json

import pytest
import requests
import torch

from llamafactory.train.ppo import ppo_utils


def _response(status_code, body):
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode()
    return response


def test_get_rewards_from_server_rejects_http_error(monkeypatch):
    monkeypatch.setattr(ppo_utils.requests, "post", lambda *args, **kwargs: _response(500, {"scores": [0.5]}))

    with pytest.raises(requests.HTTPError):
        ppo_utils.get_rewards_from_server("http://reward.test", ["message"])


def test_get_rewards_from_server_rejects_score_count_mismatch(monkeypatch):
    monkeypatch.setattr(ppo_utils.requests, "post", lambda *args, **kwargs: _response(200, {"scores": [0.5]}))

    with pytest.raises(ValueError, match="one score for each message"):
        ppo_utils.get_rewards_from_server("http://reward.test", ["first", "second"])


def test_get_rewards_from_server_returns_scores(monkeypatch):
    def post(url, json, headers):
        assert url == "http://reward.test"
        assert json == {"model": "model", "messages": ["first", "second"]}
        assert headers == {"Content-Type": "application/json"}
        return _response(200, {"scores": [0.5, -0.25]})

    monkeypatch.setattr(ppo_utils.requests, "post", post)

    rewards = ppo_utils.get_rewards_from_server("http://reward.test", ["first", "second"])

    torch.testing.assert_close(rewards, torch.tensor([0.5, -0.25]))
