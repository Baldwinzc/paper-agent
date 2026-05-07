from paper_agent.config import LLMConfig, load_llm_config
from paper_agent.llm import ChatMessage, LLMClient


def test_endpoint_builder_accepts_base_url():
    client = LLMClient(
        LLMConfig(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
        )
    )

    assert client.endpoint == "https://api.deepseek.com/chat/completions"


def test_payload_uses_messages_and_model():
    client = LLMClient(
        LLMConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="deepseek-v4-pro",
            max_tokens=123,
        )
    )

    payload = client._payload(
        [ChatMessage(role="user", content="hello")],
        temperature=None,
        max_tokens=None,
        response_format={"type": "json_object"},
    )

    assert payload["model"] == "deepseek-v4-pro"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 123
    assert payload["response_format"] == {"type": "json_object"}


def test_config_defaults_to_deepseek_v4_pro(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("TEXT_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    config = load_llm_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-pro"


def test_config_accepts_deepseek_base_alias(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.invalid")

    config = load_llm_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://api.deepseek.com"
