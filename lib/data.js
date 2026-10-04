import dataset from "../data/dataset.json";
import stats from "../data/stats.json";
import analyzed from "../data/analyzed.json";
import themesFile from "../data/themes.json";

const byId = new Map(dataset.map((r) => [r.id, r]));
const analysis = new Map((analyzed.records || []).map((a) => [a.id, a]));

export const getStats = () => stats;
export const hasAnalysis = () => (analyzed.records || []).length > 0;
export const analysisMeta = () => analyzed.meta;

export function themeList() {
  return (analyzed.meta?.themes || themesFile.themes || []);
}

/** Aggregates for the analyzer window. Off-topic records (relevant=false) are excluded from every chart. */
export function aggregate() {
  const themes = themeList();
  const counts = Object.fromEntries(themes.map((t) => [t.id, 0]));
  const terms = new Map();
  let relevant = 0, offTopic = 0;
  for (const a of analysis.values()) {
    if (!a.relevant) { offTopic++; continue; }
    relevant++;
    for (const t of a.themes) if (t in counts) counts[t]++;
    for (const k of a.keywords || []) { const w = k.trim().toLowerCase(); if (w) terms.set(w, (terms.get(w) || 0) + 1); }
  }
  const themeRows = themes.map((t) => ({ ...t, count: counts[t.id] })).sort((a, b) => b.count - a.count);
  const termRows = [...terms.entries()].sort((a, b) => b[1] - a[1]).slice(0, 60).map(([term, count]) => ({ term, count }));
  return { themes: themeRows, terms: termRows, analysed: analysis.size, relevant, offTopic, total: dataset.length };
}

/** Random quotes for a category. category: {theme} or {term}; optional platform_group / year filters. */
export function randomEvidence({ theme, term, platform, year, n = 5, exclude = [] }) {
  const ex = new Set(exclude);
  const pool = [];
  for (const a of analysis.values()) {
    if (!a.relevant || ex.has(a.id)) continue;
    if (theme && !a.themes.includes(theme)) continue;
    if (term && !(a.keywords || []).some((k) => k.trim().toLowerCase() === term.toLowerCase())) continue;
    const r = byId.get(a.id); if (!r) continue;
    if (platform && r.platform_group !== platform) continue;
    if (year && String(r.year ?? "undated") !== String(year)) continue;
    pool.push(r);
  }
  // partial Fisher-Yates: n random items without bias
  for (let i = 0; i < Math.min(n, pool.length); i++) {
    const j = i + Math.floor(Math.random() * (pool.length - i)); [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const picked = pool.slice(0, n).map((r) => ({
    id: r.id, text: r.text, platform: r.platform_group, date: r.date, year: r.year, url: r.url, title: r.thread_title,
    themes: analysis.get(r.id).themes,
  }));
  return { quotes: picked, matching: pool.length };
}

export const platformGroups = () => Object.keys(stats.by_platform);
export const years = () => Object.keys(stats.by_year);
