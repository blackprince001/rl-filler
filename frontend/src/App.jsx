import { useEffect } from "react";
import "./App.css";
import PlayPage from "./pages/PlayPage";
import StatsPage from "./pages/StatsPage";
import HistoryPage from "./pages/HistoryPage";
import ReplayPage from "./pages/ReplayPage";
import { useHashRoute } from "./lib/router";
import { ensureSession } from "./lib/api";

export default function App() {
  const route = useHashRoute();

  // Establish the client cookie before any data fetches or the WS connect,
  // so every request thereafter is scoped to this browser.
  useEffect(() => {
    ensureSession();
  }, []);

  switch (route.page) {
    case "stats":   return <StatsPage scope={route.params.scope || "mine"} />;
    case "history": return <HistoryPage scope={route.params.scope || "mine"} />;
    case "replay":  return <ReplayPage matchId={route.params.id} scope={route.params.scope || "mine"} />;
    case "play":
    default:        return <PlayPage />;
  }
}
