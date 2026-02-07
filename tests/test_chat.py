#!/usr/bin/env python3
"""Quick smoke-test for the chatbot backend modules."""

import sys
sys.path.insert(0, ".")

from src.services.llm_service import LLMService, LLMConfig, LLMProvider, LLMError
from src.services.data_context_builder import DataContextBuilder
from src.services.conversation_manager import ConversationManager, ChatMessage
from src.services.chat_service import ChatService


def test_llm_config(monkeypatch):
    print("=== LLMConfig defaults ===")
    monkeypatch.setenv('LLM_API_KEY', '')
    cfg = LLMConfig(api_key="test_key_123")
    assert cfg.provider == LLMProvider.GROQ
    assert cfg.model == "llama-3.3-70b-versatile"
    assert "groq.com" in cfg.base_url
    assert cfg.is_configured
    print(f"  Provider: {cfg.provider.value}")
    print(f"  Model: {cfg.model}")
    print(f"  Base URL: {cfg.base_url}")

    cfg2 = LLMConfig(api_key="gkey", provider=LLMProvider.GEMINI)
    assert cfg2.model == "gemini-2.0-flash"
    print(f"  Gemini model: {cfg2.model}")

    cfg3 = LLMConfig(api_key='')  # no key
    assert not cfg3.is_configured
    print("  ✅ PASSED")


def test_data_context_builder():
    print("\n=== DataContextBuilder ===")
    builder = DataContextBuilder()
    builder.ingest_inventory_results({
        "meta": {"completed_at": "2026-02-07", "total_time_seconds": 65},
        "behavior": {"summary": {
            "temporal_insights": {
                "peak_hour_label": "16:00",
                "peak_day": "Friday",
                "weekend_pct": 26.9,
                "avg_orders_per_day": 399.8,
            },
            "purchase_insights": {
                "avg_items_per_order": 1.7,
                "avg_quantity_per_order": 2.3,
                "avg_order_value": 445.44,
                "median_order_value": 80.0,
            },
        }},
        "demand_model": {
            "metrics": {"mae": 2.23, "rmse": 6.77, "r2": 0.622},
            "feature_importance": [
                {"feature": "lag_14d_avg", "importance": 0.461},
            ],
        },
        "inventory": {
            "alerts": None,
            "summary": {
                "parameters": {"lead_time_days": 2, "service_level": 0.95},
                "total_items_analyzed": 17273,
            },
        },
    })
    prompt = builder.build_system_prompt()
    assert len(prompt) > 200
    assert "FlavorFlow" in prompt
    assert "16:00" in prompt
    assert "Friday" in prompt
    print(f"  System prompt length: {len(prompt)} chars")
    print(f"  Sections: {builder.get_section_keys()}")
    print("  ✅ PASSED")


def test_conversation_manager():
    print("\n=== ConversationManager ===")
    mgr = ConversationManager(system_prompt="You are a test bot.")
    sid = mgr.create_session()
    assert mgr.active_session_count == 1
    mgr.add_user_message(sid, "Hello")
    mgr.add_assistant_message(sid, "Hi there!")
    msgs = mgr.get_llm_messages(sid)
    assert len(msgs) == 3  # system + user + assistant
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    history = mgr.get_history(sid)
    assert len(history) == 2  # user + assistant (no system)
    mgr.reset_session(sid)
    assert len(mgr.get_history(sid)) == 0
    assert len(mgr.get_llm_messages(sid)) == 1  # system preserved
    print(f"  Session: {sid}")
    print(f"  Messages after reset: {len(mgr.get_llm_messages(sid))}")
    print("  ✅ PASSED")


def test_chat_service(monkeypatch):
    print("\n=== ChatService ===")
    monkeypatch.setenv('LLM_API_KEY', '')
    chat = ChatService(api_key='')
    assert not chat.is_configured
    assert not chat.is_context_loaded

    status = chat.configure(api_key="test_key", provider="groq")
    assert chat.is_configured
    assert status["provider"] == "groq"
    assert status["model"] == "llama-3.3-70b-versatile"
    print(f"  Provider: {chat.provider}")
    print(f"  Model: {chat.model}")

    # Load context
    prompt = chat.load_analysis_context(
        inventory_results={
            "meta": {"completed_at": "2026-02-07", "total_time_seconds": 65},
            "behavior": {"summary": {
                "temporal_insights": {"peak_hour_label": "16:00", "peak_day": "Friday", "weekend_pct": 26.9, "avg_orders_per_day": 399.8},
                "purchase_insights": {"avg_items_per_order": 1.7, "avg_quantity_per_order": 2.3, "avg_order_value": 445.44, "median_order_value": 80.0},
            }},
            "demand_model": {"metrics": {"mae": 2.23, "rmse": 6.77, "r2": 0.622}, "feature_importance": []},
            "inventory": {"alerts": None, "summary": {"parameters": {"lead_time_days": 2, "service_level": 0.95}, "total_items_analyzed": 17273}},
        }
    )
    assert chat.is_context_loaded
    assert "FlavorFlow" in prompt
    print(f"  Context loaded: {len(prompt)} chars")

    # Session management
    sid = chat.create_session()
    assert sid
    sessions = chat.list_sessions()
    assert len(sessions) == 1
    chat.delete_session(sid)
    assert len(chat.list_sessions()) == 0
    print("  ✅ PASSED")


def test_chat_api_import():
    print("\n=== Chat API Router ===")
    from src.api.chat import router, get_chat_service, set_chat_service
    assert router.prefix == "/chat"
    routes = [r.path for r in router.routes]
    assert "/chat/configure" in routes
    assert "/chat/message" in routes
    assert "/chat/stream" in routes
    assert "/chat/history" in routes
    assert "/chat/health" in routes
    assert "/chat/usage" in routes
    print(f"  Routes: {len(routes)}")
    for r in routes:
        print(f"    {r}")
    print("  ✅ PASSED")


if __name__ == "__main__":
    test_llm_config()
    test_data_context_builder()
    test_conversation_manager()
    test_chat_service()
    test_chat_api_import()
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 50)
