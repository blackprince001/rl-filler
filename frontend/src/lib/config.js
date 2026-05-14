function defaults() {
  if (typeof window === "undefined") {
    return { ws: "ws://localhost:8000/ws/game", api: "http://localhost:8000" };
  }
  const { protocol, hostname } = window.location;
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return { ws: "ws://localhost:8000/ws/game", api: "http://localhost:8000" };
  }
  const wsProto = protocol === "https:" ? "wss:" : "ws:";
  const httpProto = protocol === "https:" ? "https:" : "http:";
  return {
    ws: `${wsProto}//${hostname}/ws/game`,
    api: `${httpProto}//${hostname}`,
  };
}

const d = defaults();
export const WS_URL = import.meta.env.VITE_WS_URL || d.ws;
export const API_URL =
  import.meta.env.VITE_API_URL ||
  // Best-effort: derive HTTP base from WS URL if only WS was supplied.
  (import.meta.env.VITE_WS_URL
    ? import.meta.env.VITE_WS_URL.replace(/^wss:/, "https:").replace(/^ws:/, "http:").replace(/\/ws\/game$/, "")
    : d.api);
