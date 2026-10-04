import { getStats, aggregate, hasAnalysis, analysisMeta, platformGroups, years } from "../lib/data";
import Analyzer from "./Analyzer";

export const dynamic = "force-static";

const fmt = (n) => n.toLocaleString("en-US");

export default function Page() {
  const s = getStats();
  const analysed = hasAnalysis();
  const agg = analysed ? aggregate() : null;
  const meta = analysed ? analysisMeta() : null;
  const rm = s.removed;

  return (
    <main className="wrap">
      <header className="head">
        <h1>Photo retrieval evidence</h1>
        <p className="sub">What people say when they cannot find a photo they remember. Public posts, comments and reviews, plus primary research.</p>
      </header>

      <section aria-labelledby="a-h">
        <h2 id="a-h" className="sec"><span>A</span> Corpus</h2>
        <div className="kpis">
          <div className="kpi">
            <div className="kpi-n">{fmt(s.scraped_total)}</div>
            <div className="kpi-l">Total corpus</div>
            <div className="kpi-d">Complete data scraped based on the basic set of filters.</div>
            <div className="kpi-s">Before validation</div>
          </div>
          <div className="kpi">
            <div className="kpi-n">{fmt(s.unique_sources)}</div>
            <div className="kpi-l">Unique sources</div>
            <div className="kpi-d">Distinct threads, videos and pages the records came from.</div>
            <div className="kpi-s">Across {s.unique_platforms} platforms</div>
          </div>
          <div className="kpi kpi-hi">
            <div className="kpi-n">{fmt(s.retained)}</div>
            <div className="kpi-l">Retained after validation</div>
            <div className="kpi-d">Individual comments, posts, reviews and survey answers.</div>
            <div className="kpi-s">{s.retained_pct}% of the corpus. {fmt(s.retained_unique_sources)} sources remain</div>
          </div>
        </div>
        <details className="why">
          <summary>What validation removed ({fmt(s.scraped_total - s.retained)} records)</summary>
          <ul>
            <li><b>{fmt(rm.exact_duplicates)}</b> exact duplicates (identical text and source link). Similar comments are kept.</li>
            <li><b>{fmt(rm.content_free)}</b> content-free fragments (emoji only, single words, bare thanks).</li>
            <li><b>{fmt(rm.outside_2021_2026)}</b> dated outside {s.date_window}.</li>
            <li><b>{fmt(rm.empty)}</b> empty.</li>
          </ul>
          <p className="note">{fmt(s.undated_retained)} retained records carry no date and are kept. Data window: {s.date_window}.</p>
        </details>
      </section>

      <section aria-labelledby="b-h">
        <h2 id="b-h" className="sec"><span>B</span> Analyzer</h2>
        {analysed ? (
          <Analyzer
            themes={agg.themes}
            terms={agg.terms}
            relevant={agg.relevant}
            offTopic={agg.offTopic}
            platforms={platformGroups().slice(0, 12)}
            years={years()}
            model={meta?.model}
            generatedAt={meta?.generated_at}
          />
        ) : (
          <div className="empty">
            <h3>Analysis has not been run yet</h3>
            <p>Themes are produced offline by Gemini, then shipped with the app. To generate them:</p>
            <ol>
              <li>Copy <code>.env.example</code> to <code>.env.local</code> and add your <code>GEMINI_API_KEY</code>.</li>
              <li>Run <code>npm run analyze:themes</code>, then review <code>data/themes.json</code>.</li>
              <li>Run <code>npm run analyze:classify</code>, then redeploy.</li>
            </ol>
          </div>
        )}
      </section>

      <footer className="foot">
        Source data: public posts, comments and reviews; author names are not included. Themes are model-generated and should be read as
        a starting map of the evidence, not as findings.
      </footer>
    </main>
  );
}
