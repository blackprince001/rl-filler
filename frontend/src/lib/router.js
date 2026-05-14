import { useEffect, useState } from "react";

function parseHash() {
  const h = window.location.hash.replace(/^#\/?/, "");
  if (!h) return { page: "play", params: {} };
  const [pathRaw, queryRaw] = h.split("?");
  const segments = pathRaw.split("/").filter(Boolean);
  const params = {};
  if (queryRaw) {
    for (const pair of queryRaw.split("&")) {
      const [k, v] = pair.split("=");
      if (k) params[k] = decodeURIComponent(v || "");
    }
  }
  return { page: segments[0] || "play", segments, params };
}

export function useHashRoute() {
  const [route, setRoute] = useState(parseHash);
  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}

export function navigate(path) {
  window.location.hash = path.startsWith("#") ? path : `#${path}`;
}
