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

import pytest
from fastapi import HTTPException

from llamafactory.api import chat


class FakeResponse:
    def __init__(self, content: bytes = b"media", is_redirect: bool = False, error: Exception | None = None):
        self.closed = False
        self.content = content
        self.is_redirect = is_redirect
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


def test_fetch_remote_media_returns_buffer_and_closes_response(monkeypatch):
    response = FakeResponse(content=b"image")

    monkeypatch.setattr(chat, "check_ssrf_url", lambda url: None)
    monkeypatch.setattr(chat.requests, "get", lambda *args, **kwargs: response)

    media = chat._fetch_remote_media("https://example.com/image.png")

    assert media.read() == b"image"
    assert response.closed


def test_fetch_remote_media_rejects_redirect_and_closes_response(monkeypatch):
    response = FakeResponse(is_redirect=True)

    monkeypatch.setattr(chat, "check_ssrf_url", lambda url: None)
    monkeypatch.setattr(chat.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(HTTPException) as exc_info:
        chat._fetch_remote_media("https://example.com/image.png")

    assert exc_info.value.status_code == 403
    assert response.closed


def test_fetch_remote_media_maps_request_errors(monkeypatch):
    monkeypatch.setattr(chat, "check_ssrf_url", lambda url: None)

    def fake_get(*args, **kwargs):
        raise chat.requests.Timeout("timed out")

    monkeypatch.setattr(chat.requests, "get", fake_get)

    with pytest.raises(HTTPException) as exc_info:
        chat._fetch_remote_media("https://example.com/image.png")

    assert exc_info.value.status_code == 400


def test_fetch_remote_media_maps_http_errors_and_closes_response(monkeypatch):
    response = FakeResponse(error=chat.requests.HTTPError("not found"))

    monkeypatch.setattr(chat, "check_ssrf_url", lambda url: None)
    monkeypatch.setattr(chat.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(HTTPException) as exc_info:
        chat._fetch_remote_media("https://example.com/image.png")

    assert exc_info.value.status_code == 400
    assert response.closed
