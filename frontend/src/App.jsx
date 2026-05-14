import "./App.css";
import PlayPage from "./pages/PlayPage";
import StatsPage from "./pages/StatsPage";
import HistoryPage from "./pages/HistoryPage";
import ReplayPage from "./pages/ReplayPage";
import { useHashRoute } from "./lib/router";

export default function App() {
  const route = useHashRoute();
  switch (route.page) {
    case "stats":   return <StatsPage />;
    case "history": return <HistoryPage />;
    case "replay":  return <ReplayPage matchId={route.params.id} />;
    case "play":
    default:        return <PlayPage />;
  }
}
