"""Tests for the custom OpenAI-compatible provider (9Router, LM Studio, vLLM, etc.)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUGHUNTER_DIR = ROOT / "bughunter"
if str(BUGHUNTER_DIR) not in sys.path:
    sys.path.insert(0, str(BUGHUNTER_DIR))

from brain import LLMClient  # noqa: E402
import engine  # noqa: E402


def test_custom_provider_unavailable_without_base_url(monkeypatch):
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("CUSTOM_API_BASE", raising=False)
    client = LLMClient("custom")
    assert client.available is False


def test_custom_provider_init_normalizes_url_and_optional_key(monkeypatch):
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://localhost:20128/v1/chat/completions/")
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    client = LLMClient("custom")
    assert client.available is True
    assert client._api_base == "http://localhost:20128/v1"
    assert "Authorization" not in client._http.headers


def test_custom_provider_sets_bearer_key_when_provided(monkeypatch):
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://127.0.0.1:1234/v1/")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-local-test")

    client = LLMClient("custom")
    assert client.available is True
    assert client._api_base == "http://127.0.0.1:1234/v1"
    assert client._http.headers["Authorization"] == "Bearer sk-local-test"


def test_custom_provider_list_models_and_chat(monkeypatch):
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://localhost:20128/v1")
    client = LLMClient("custom")

    mock_models_resp = MagicMock()
    mock_models_resp.json.return_value = {"data": [{"id": "9router/gpt-4o"}, {"id": "local-qwen"}]}
    mock_models_resp.raise_for_status.return_value = None

    mock_chat_resp = MagicMock()
    mock_chat_resp.status_code = 200
    mock_chat_resp.json.return_value = {"choices": [{"message": {"content": "READY"}}]}
    mock_chat_resp.raise_for_status.return_value = None

    client._http = MagicMock()
    client._http.get.return_value = mock_models_resp
    client._http.post.return_value = mock_chat_resp

    models = client.list_models()
    assert models == ["9router/gpt-4o", "local-qwen"]

    reply = client.chat(None, "sys", "hi", max_tokens=10)
    assert reply == "READY"
    client._http.post.assert_called_once()
    post_url = client._http.post.call_args.args[0]
    post_body = json.loads(client._http.post.call_args.kwargs["data"])
    assert post_url == "http://localhost:20128/v1/chat/completions"
    assert post_body["model"] == "9router/gpt-4o"


def test_setup_persists_custom_provider(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(engine, "CONFIG", cfg_path)
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    monkeypatch.delenv("BRAIN_MODEL", raising=False)

    fake_client = MagicMock()
    fake_client.available = True
    fake_client.description = "Custom OpenAI-compatible (http://localhost:20128/v1)"
    fake_client.list_models.return_value = ["9router/claude-sonnet-4-6"]
    fake_client.chat.return_value = "READY"

    monkeypatch.setattr(engine, "_import_brain", lambda: (MagicMock(), lambda p: fake_client))

    args = SimpleNamespace(
        setup_provider="custom",
        setup_model="9router/claude-sonnet-4-6",
        setup_base_url="http://localhost:20128/v1",
        setup_api_key="my-secret",
        provider=None,
        model=None,
    )
    engine.cmd_setup(args)

    saved = json.loads(cfg_path.read_text())
    assert saved["provider"] == "custom"
    assert saved["CUSTOM_BASE_URL"] == "http://localhost:20128/v1"
    assert saved["CUSTOM_API_KEY"] == "my-secret"
    assert saved["models"]["custom"] == "9router/claude-sonnet-4-6"


def test_custom_provider_parses_trailing_sse_done_and_sse_stream(monkeypatch):
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://localhost:20128/v1")
    client = LLMClient("custom")

    # Case 1: JSON object immediately followed by "data: [DONE]\n\n" (9Router behavior)
    resp_trailing = MagicMock()
    resp_trailing.status_code = 200
    resp_trailing.raise_for_status.return_value = None
    resp_trailing.json.side_effect = json.JSONDecodeError("Extra data", "x", 10)
    resp_trailing.text = (
        '{"id":"1","choices":[{"message":{"content":"READY","role":"assistant"}}]}'
        'data: [DONE]\n\n'
    )
    client._http = MagicMock()
    client._http.post.return_value = resp_trailing
    assert client.chat("zro-bro/deepseek-v4.1-flash", "sys", "hi") == "READY"

    # Case 2: Standard SSE delta stream
    resp_sse = MagicMock()
    resp_sse.status_code = 200
    resp_sse.raise_for_status.return_value = None
    resp_sse.json.side_effect = json.JSONDecodeError("Expecting value", "x", 0)
    resp_sse.text = (
        'data: {"choices":[{"delta":{"content":"REA"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"DY"}}]}\n\n'
        'data: [DONE]\n\n'
    )
    client._http.post.return_value = resp_sse
    assert client.chat("zro-bro/deepseek-v4.1-flash", "sys", "hi") == "READY"
