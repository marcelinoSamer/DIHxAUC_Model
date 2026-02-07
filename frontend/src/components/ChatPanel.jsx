import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Trash2, RefreshCw, Zap, AlertCircle } from "lucide-react";
import { sendChatMessage, getChatConfig, resetChatSession, streamChatMessage } from "../services/api";

export default function ChatPanel() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! I'm your FlavorFlow AI assistant 👋\n\nI can help you with menu analysis, inventory insights, pricing suggestions, and more. Ask me anything about your restaurant data!" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [config, setConfig] = useState({ isConfigured: false, provider: "", model: "" });
  const [useStreaming, setUseStreaming] = useState(true);
  const [backendError, setBackendError] = useState("");
  const [initializing, setInitializing] = useState(true);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check backend readiness on mount (LLM is pre-configured by the server)
  useEffect(() => {
    let retries = 0;
    const maxRetries = 5;

    const checkBackend = async () => {
      try {
        const status = await getChatConfig();
        if (status.is_configured) {
          setConfig({ isConfigured: true, provider: status.provider, model: status.model });
          setBackendError("");
          setInitializing(false);
          return;
        }
        // Backend is up but key not loaded yet (still starting?)
        if (retries < maxRetries) {
          retries++;
          setTimeout(checkBackend, 2000);
          return;
        }
        setBackendError("LLM API key not found. Please set LLM_API_KEY in the .env file and restart the backend.");
        setInitializing(false);
      } catch {
        if (retries < maxRetries) {
          retries++;
          setTimeout(checkBackend, 2000);
          return;
        }
        setBackendError("Cannot reach the backend. Make sure the FastAPI server is running on port 8000.");
        setInitializing(false);
      }
    };
    checkBackend();
  }, []);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    if (useStreaming) {
      // Streaming mode
      let streamedContent = "";
      setMessages((prev) => [...prev, { role: "assistant", content: "", streaming: true }]);

      const cancel = streamChatMessage({
        message: text,
        sessionId,
        onToken: (token) => {
          streamedContent += token;
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: "assistant", content: streamedContent, streaming: true };
            return updated;
          });
        },
        onDone: (sid) => {
          if (sid) setSessionId(sid);
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...updated[updated.length - 1], streaming: false };
            return updated;
          });
          setLoading(false);
        },
        onError: (err) => {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "assistant",
              content: `❌ Error: ${err}`,
              error: true,
              streaming: false,
            };
            return updated;
          });
          setLoading(false);
        },
      });

      // Store cancel function for potential cleanup
      return () => cancel();
    } else {
      // Non-streaming mode
      try {
        const data = await sendChatMessage({ message: text, sessionId });
        setSessionId(data.session_id);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.reply,
            model: data.model,
            usage: data.usage,
          },
        ]);
      } catch (err) {
        const errorMsg = err.response?.data?.detail || err.message || "Something went wrong";
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `❌ ${errorMsg}`, error: true },
        ]);
      }
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (sessionId) {
      try { await resetChatSession(sessionId); } catch { /* ignore */ }
    }
    setSessionId(null);
    setMessages([
      { role: "assistant", content: "Conversation reset. How can I help you? 🔄" },
    ]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-panel">
      <div className="page-header">
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <h1>AI Assistant</h1>
          {config.isConfigured && (
            <span className="config-badge online">
              <Zap size={12} /> {config.provider} / {config.model}
            </span>
          )}
          {initializing && (
            <span className="config-badge" style={{ background: "rgba(245,158,11,0.12)", color: "#f59e0b" }}>
              <Loader2 size={12} className="spin" /> Connecting…
            </span>
          )}
        </div>
        <div className="chat-actions">
          <button className="btn-icon" onClick={() => setUseStreaming(!useStreaming)} title={useStreaming ? "Streaming ON" : "Streaming OFF"}>
            <Zap size={16} className={useStreaming ? "text-green" : "text-muted"} />
          </button>
          <button className="btn-icon" onClick={handleReset} title="Reset conversation">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {backendError && (
        <div className="card config-panel" style={{ borderColor: "rgba(239,68,68,0.3)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--red)" }}>
            <AlertCircle size={18} />
            <span>{backendError}</span>
          </div>
        </div>
      )}

      {/* Chat Window */}
      <div className="chat-window">
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role} ${m.error ? "error" : ""}`}>
            <div className="msg-avatar">
              {m.role === "assistant" ? <Bot size={18} /> : <User size={18} />}
            </div>
            <div className="msg-body">
              <div className="msg-content">{m.content}</div>
              {m.usage && (
                <div className="msg-meta">
                  {m.model} · {m.usage.total_tokens || 0} tokens
                </div>
              )}
              {m.streaming && <span className="streaming-cursor" />}
            </div>
          </div>
        ))}
        {loading && !useStreaming && (
          <div className="chat-msg assistant">
            <div className="msg-avatar"><Bot size={18} /></div>
            <div className="msg-body">
              <Loader2 size={18} className="spin" />
              <span className="thinking-text">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-bar">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={config.isConfigured ? "Ask about menu performance, inventory, pricing..." : initializing ? "Connecting to AI…" : "Backend not available — check that the server is running"}
          disabled={!config.isConfigured || loading}
          rows={1}
        />
        <button
          className="btn-send"
          onClick={handleSend}
          disabled={!input.trim() || loading || !config.isConfigured}
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}
