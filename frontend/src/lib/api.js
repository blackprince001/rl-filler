import { API_URL } from "./config";

export function apiUrl(path) {
  return `${API_URL}${path}`;
}

async function request(path, init = {}) {
  const res = await fetch(apiUrl(path), {
    credentials: "include",
    ...init,
  });
  if (!res.ok) {
    throw new Error(`${init.method || "GET"} ${path} → ${res.status}`);
  }
  return res.json();
}

export async function ensureSession() {
  try {
    await request("/session", { method: "POST" });
  } catch {
    // Non-fatal: if the cookie endpoint is unreachable, the WS still works,
    // but server-side history won't be linked to this client.
  }
}

export async function fetchGames({ scope = "mine", page = 1 } = {}) {
  return request(`/games?scope=${scope}&page=${page}&page_size=24`);
}

export async function fetchGame(id, { scope = "mine" } = {}) {
  try {
    return await request(`/games/${id}?scope=${scope}`);
  } catch {
    return null;
  }
}

export async function fetchStats({ scope = "mine" } = {}) {
  return request(`/stats?scope=${scope}`);
}
