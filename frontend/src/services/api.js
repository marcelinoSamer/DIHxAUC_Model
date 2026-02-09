import axios from "axios";

const API_BASE = "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ── Chat endpoints ──────────────────────────────────────────────────────────

export async function configureLLM({ apiKey, provider = "groq", model, temperature }) {
  const payload = {
    provider,
    model: model || undefined,
    temperature: temperature || undefined,
  };
  if (apiKey) payload.api_key = apiKey;
  const { data } = await api.post("/chat/configure", payload);
  return data;
}

export async function getChatConfig() {
  const { data } = await api.get("/chat/config");
  return data;
}

export async function sendChatMessage({ message, sessionId, temperature, maxTokens }) {
  const { data } = await api.post("/chat/message", {
    message,
    session_id: sessionId || undefined,
    temperature,
    max_tokens: maxTokens,
  });
  return data;
}

export function streamChatMessage({ message, sessionId, onToken, onDone, onError }) {
  const controller = new AbortController();

  fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId || undefined,
    }),
    signal: controller.signal,
  })
    .then(async (response) => {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.token) onToken(payload.token);
            if (payload.done) onDone?.(payload.session_id);
            if (payload.error) onError?.(payload.error);
          } catch {
            // skip malformed
          }
        }
      }
      onDone?.();
    })
    .catch((err) => {
      if (err.name !== "AbortError") onError?.(err.message);
    });

  return () => controller.abort();
}

export async function getChatHistory(sessionId) {
  const { data } = await api.get("/chat/history", { params: { session_id: sessionId } });
  return data;
}

export async function resetChatSession(sessionId) {
  const { data } = await api.post("/chat/reset", { session_id: sessionId });
  return data;
}

export async function getChatHealth() {
  const { data } = await api.get("/chat/health");
  return data;
}

export async function getChatUsage() {
  const { data } = await api.get("/chat/usage");
  return data;
}

// ── Recommendations endpoints ───────────────────────────────────────────────

export async function getMonthlySuggestions() {
  const { data } = await api.get("/recommendations/weekly");
  return data;
}

export async function getDashboardData() {
  const { data } = await api.get("/dashboard/data");
  return data;
}

export async function submitInventoryData(file) {
  // Upload CSV file directly to the inventory ingest endpoint
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await api.post("/inventory/ingest", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

// ── Analysis endpoints ──────────────────────────────────────────────────────

export async function initializeSystem(dataDir = "data/") {
  const { data } = await api.post("/initialize", null, { params: { data_dir: dataDir } });
  return data;
}

export async function runAnalysis({ includePredictions = true, includeClustering = true } = {}) {
  const { data } = await api.post("/analyze", {
    include_predictions: includePredictions,
    include_clustering: includeClustering,
  });
  return data;
}

export async function getMenuItems({ category, limit = 100, minOrders = 0 } = {}) {
  const { data } = await api.get("/items", {
    params: { category, limit, min_orders: minOrders },
  });
  return data;
}

export async function getHealthStatus() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
