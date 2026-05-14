const KEY = "rl-filler.matches.v1";
const MAX_MATCHES = 200;

export function loadMatches() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveMatch(match) {
  const matches = loadMatches();
  matches.unshift(match);
  if (matches.length > MAX_MATCHES) matches.length = MAX_MATCHES;
  try {
    localStorage.setItem(KEY, JSON.stringify(matches));
  } catch {
    // localStorage is full or unavailable — best-effort only.
  }
  return matches;
}

export function getMatch(id) {
  return loadMatches().find((m) => m.id === id) || null;
}

export function deleteMatch(id) {
  const matches = loadMatches().filter((m) => m.id !== id);
  localStorage.setItem(KEY, JSON.stringify(matches));
  return matches;
}

export function clearMatches() {
  localStorage.removeItem(KEY);
}

export function computeStats(matches) {
  const totals = {
    games: matches.length,
    human_wins: 0,
    ai_wins: 0,
    ties: 0,
    avg_turns: 0,
    avg_human_score: 0,
    avg_ai_score: 0,
    avg_margin: 0,
  };
  if (!matches.length) return totals;

  let turnSum = 0;
  let humanScoreSum = 0;
  let aiScoreSum = 0;
  let marginSum = 0;
  for (const m of matches) {
    if (m.winner === "human") totals.human_wins++;
    else if (m.winner === "ai") totals.ai_wins++;
    else totals.ties++;
    turnSum += m.turns || 0;
    humanScoreSum += m.final_scores?.[0] || 0;
    aiScoreSum += m.final_scores?.[1] || 0;
    marginSum += (m.final_scores?.[0] || 0) - (m.final_scores?.[1] || 0);
  }
  const n = matches.length;
  totals.avg_turns = +(turnSum / n).toFixed(2);
  totals.avg_human_score = +(humanScoreSum / n).toFixed(2);
  totals.avg_ai_score = +(aiScoreSum / n).toFixed(2);
  totals.avg_margin = +(marginSum / n).toFixed(2);
  totals.win_rate = +((totals.human_wins / n) * 100).toFixed(1);
  return totals;
}

export function newMatchId() {
  return `m_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
