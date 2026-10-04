"use client";
import { useEffect, useMemo, useRef, useState } from "react";

const fmt = (n) => n.toLocaleString("en-US");
const CLAMP = 420;

/** biggest items in the middle, smaller towards the edges */
function centerOut(items) {
  const out = [];
  items.forEach((it, i) => (i % 2 === 0 ? out.push(it) : out.unshift(it)));
  return out;
}

export default function Analyzer({ themes, terms, relevant, offTopic, platforms, years, model, generatedAt }) {
  const live = themes.filter((t) => t.count > 0);
  const [view, setView] = useState("themes");
  const [sel, setSel] = useState(live[0] ? { type: "theme", value: live[0].id } : null);
  const [platform, setPlatform] = useState("");
  const [year, setYear] = useState("");
  const [quotes, setQuotes] = useState([]);
  const [matching, setMatching] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState({});
  const seq = useRef(0);

  const themeById = useMemo(() => Object.fromEntries(themes.map((t) => [t.id, t])), [themes]);
  const maxT = Math.max(1, ...live.map((t) => t.count));
  const cloud = view === "themes"
    ? live.map((t) => ({ key: t.id, label: t.label, count: t.count }))
    : terms.map((t) => ({ key: t.term, label: t.term, count: t.count }));
  const maxC = Math.max(1, ...cloud.map((c) => c.count));
  const minC = Math.min(...cloud.map((c) => c.count), maxC);
  const rel = (c) => (c - minC) / Math.max(1, maxC - minC);
  const size = (c) => 14 + 30 * Math.sqrt(rel(c));
  const tier = (c) => (rel(c) > 0.66 ? 3 : rel(c) > 0.33 ? 2 : 1);

  async function fetchQuotes(params) {
    const r = await fetch("/api/evidence?" + params.toString());
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "request failed");
    return j;
  }

  async function load(keepSeen) {
    if (!sel) return;
    const my = ++seq.current;
    setBusy(true); setError("");
    const q = new URLSearchParams({ n: "5" });
    q.set(sel.type, sel.value);
    if (platform) q.set("platform", platform);
    if (year) q.set("year", year);
    try {
      let j;
      if (keepSeen && quotes.length) {
        const q2 = new URLSearchParams(q); q2.set("exclude", quotes.map((x) => x.id).join(","));
        j = await fetchQuotes(q2);
        if (j.quotes.length === 0) { setError("You have seen every matching quote. Showing a fresh draw."); j = await fetchQuotes(q); }
      } else j = await fetchQuotes(q);
      if (my !== seq.current) return;
      setQuotes(j.quotes); setMatching(j.matching); setOpen({});
    } catch (e) {
      if (my === seq.current) { setError(e.message); setQuotes([]); }
    } finally {
      if (my === seq.current) setBusy(false);
    }
  }

  useEffect(() => { load(false); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [sel, platform, year]);

  const selLabel = !sel ? "" : sel.type === "theme" ? themeById[sel.value]?.label : sel.value;
  const selDef = sel?.type === "theme" ? themeById[sel.value]?.definition : "";
  const isSel = (key) => sel && sel.value.toLowerCase() === key.toLowerCase() && sel.type === (view === "themes" ? "theme" : "term");

  if (!live.length) return <div className="empty"><p>Analysis ran but produced no themes. Re-run <code>npm run analyze:classify</code>.</p></div>;

  return (
    <div className="analyzer">
      <p className="note top">
        {fmt(relevant)} on-topic records analysed{offTopic ? `, ${fmt(offTopic)} set aside as off-topic` : ""}.
        A record can carry up to two themes, so theme totals add up to more than the record count.
        {model ? ` Coded by ${model}` : ""}{generatedAt ? ` on ${generatedAt.slice(0, 10)}` : ""}.
      </p>

      <div className="grid2">
        <div className="card">
          <div className="card-h">
            <h3>Themes by frequency</h3>
            <div className="seg" role="tablist" aria-label="Cloud type">
              <button role="tab" aria-selected={view === "themes"} onClick={() => setView("themes")}>Themes</button>
              <button role="tab" aria-selected={view === "terms"} onClick={() => setView("terms")}>Key terms</button>
            </div>
          </div>
          <div className="cloud" aria-label="Word cloud sized by frequency">
            {centerOut(cloud).map((c) => (
              <button
                key={c.key}
                className={`w t${tier(c.count)} ${isSel(c.key) ? "on" : ""}`}
                style={{ fontSize: size(c.count) }}
                aria-pressed={!!isSel(c.key)}
                title={`${c.label}: ${c.count}`}
                onClick={() => setSel({ type: view === "themes" ? "theme" : "term", value: c.key })}
              >
                {c.label}
              </button>
            ))}
          </div>
          <p className="hint">Size = number of records. Click to pull quotes for it.</p>
        </div>

        <div className="card">
          <h3>Themes by volume</h3>
          <ol className="bars">
            {live.map((t) => (
              <li key={t.id}>
                <button
                  className={`bar ${sel?.type === "theme" && sel.value === t.id ? "on" : ""}`}
                  onClick={() => { setView("themes"); setSel({ type: "theme", value: t.id }); }}
                >
                  <span className="bl">{t.label}</span>
                  <span className="bt"><span className="bf" style={{ width: `${(100 * t.count) / maxT}%` }} /></span>
                  <span className="bn">{fmt(t.count)} <small>{((100 * t.count) / relevant).toFixed(1)}%</small></span>
                </button>
              </li>
            ))}
          </ol>
        </div>
      </div>

      <div className="card ev">
        <div className="card-h">
          <div>
            <h3>Evidence window</h3>
            <p className="note">Random verbatim quotes for: <b>{selLabel}</b>{matching ? ` (${fmt(matching)} matching)` : ""}</p>
            {selDef ? <p className="note def">{selDef}</p> : null}
          </div>
          <div className="filters">
            <label>Category
              <select value={sel?.type === "theme" ? sel.value : ""} onChange={(e) => e.target.value && setSel({ type: "theme", value: e.target.value })}>
                {sel?.type === "term" ? <option value="">Term: {sel.value}</option> : null}
                {live.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
              </select>
            </label>
            <label>Platform
              <select value={platform} onChange={(e) => setPlatform(e.target.value)}>
                <option value="">All</option>{platforms.map((p) => <option key={p}>{p}</option>)}
              </select>
            </label>
            <label>Year
              <select value={year} onChange={(e) => setYear(e.target.value)}>
                <option value="">All</option>{years.map((y) => <option key={y}>{y}</option>)}
              </select>
            </label>
            <button className="primary" onClick={() => load(true)} disabled={busy}>{busy ? "Loading" : "Shuffle"}</button>
          </div>
        </div>
        {error ? <p className="err" role="status">{error}</p> : null}
        <ul className="quotes">
          {quotes.map((q) => {
            const long = q.text.length > CLAMP;
            const shown = open[q.id] || !long ? q.text : q.text.slice(0, CLAMP).trimEnd() + "…";
            return (
              <li key={q.id}>
                <blockquote>{shown}</blockquote>
                {long ? <button className="link" onClick={() => setOpen({ ...open, [q.id]: !open[q.id] })}>{open[q.id] ? "Show less" : "Show more"}</button> : null}
                <div className="meta">
                  <span>{q.platform}</span>
                  <span>{/^\d{4}/.test(q.date || "") ? q.date.slice(0, 10) : "undated"}</span>
                  {q.themes.map((t) => <span key={t} className="chip">{themeById[t]?.label || t}</span>)}
                  {q.url ? <a href={q.url} target="_blank" rel="noopener noreferrer">source</a> : null}
                </div>
              </li>
            );
          })}
        </ul>
        {!busy && !quotes.length && !error ? <p className="note">No quotes match these filters.</p> : null}
      </div>
    </div>
  );
}
