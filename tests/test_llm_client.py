from paper_agent.config import LLMConfig
from paper_agent.llm import ChatMessage, LLMClient


def test_endpoint_builder_accepts_base_url():
    client = LLMClient(
        LLMConfig(
            api_key="test-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="doubao-seed-1-8-251228",
        )
    )

    assert client.endpoint == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"


def test_payload_uses_messages_and_model():
    client = LLMClient(
        LLMConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="doubao-seed-1-8-251228",
            max_tokens=123,
        )
    )

    payload = client._payload(
        [ChatMessage(role="user", content="hello")],
        temperature=None,
        max_tokens=None,
        response_format={"type": "json_object"},
    )

    assert payload["model"] == "doubao-seed-1-8-251228"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 123
    assert payload["response_format"] == {"type": "json_object"}
