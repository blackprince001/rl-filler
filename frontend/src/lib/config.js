function defaultWsUrl() {
  if (typeof window === "undefined") return "ws://localhost:8000/ws/game";
  const { protocol, hostname } = window.location;
  const wsProto = protocol === "https:" ? "wss:" : "ws:";
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "ws://localhost:8000/ws/game";
  }
  return `${wsProto}//${hostname}/ws/game`;
}

export const WS_URL = import.meta.env.VITE_WS_URL || defaultWsUrl();
