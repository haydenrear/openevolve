from types import SimpleNamespace

import pytest

from openevolve.config import LLMModelConfig
from openevolve.llm.openai import OpenAILLM


class _FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **params):
        self.parent.followup_params = params
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))
        self.greeting_path = None
        self.greeting_body = None
        self.followup_params = None

    def post(self, path, *, cast_to, body):
        self.greeting_path = path
        self.greeting_body = body
        return {"conversation_id": "ak:test-conversation"}


@pytest.mark.asyncio
async def test_openai_llm_harness_greeting_pins_followup_conversation():
    llm = OpenAILLM(
        LLMModelConfig(
            name="OPEN_AI_test",
            api_base="http://127.0.0.1:8000/v1",
            api_key="test",
            temperature=0,
            max_tokens=100,
            timeout=10,
            retries=0,
            retry_delay=0,
        )
    )
    fake_client = _FakeClient()
    llm.client = fake_client

    response = await llm.generate_with_context(
        system_message="system",
        messages=[{"role": "user", "content": "research"}],
        harness_greeting={
            "working_directory": "/tmp/research",
            "env": {"CODEX_HOME": "/tmp/codex"},
            "mcp_servers": [{"type": "stdio", "name": "tools", "command": "tool"}],
        },
    )

    assert response == '{"ok": true}'
    assert fake_client.greeting_path == "/harness-greeting"
    assert [message["role"] for message in fake_client.greeting_body["messages"]] == [
        "system",
        "user",
    ]
    assert fake_client.greeting_body["working_directory"] == "/tmp/research"
    assert fake_client.greeting_body["env"] == {"CODEX_HOME": "/tmp/codex"}
    assert fake_client.followup_params["extra_headers"] == {
        "X-Conversation-Id": "ak:test-conversation"
    }
    assert fake_client.followup_params["messages"][-1]["content"] == "research"
