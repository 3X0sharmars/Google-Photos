// DEV ONLY: a fake Gemini endpoint to test analyze.mjs plumbing (batching, retry, validation, resume).
// Its themes/labels are meaningless placeholders and must never be used as real analysis.
//   node scripts/dev/mock_gemini.mjs 8787
import http from "node:http";

const port = Number(process.argv[2] || 8787);
let calls = 0; const perModel = {}; const CAP = Number(process.env.MODEL_CAP || 0);
const hash = (s) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);

http.createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const send = (code, obj) => { res.writeHead(code, { "content-type": "application/json" }); res.end(JSON.stringify(obj)); };
    if (req.method === "GET") return send(200, { name: "models/mock" });
    calls++;
    const model = (req.url.match(/models\/([^:]+):/) || [])[1] || "x"; perModel[model] = (perModel[model] || 0) + 1;
    if (CAP && perModel[model] > CAP) return send(429, { error: { message: "Quota exceeded for metric: generate_content_free_tier_requests, limit: " + CAP + ", model: " + model + "\nPlease retry in 5h19m14.6s." } });
    if (!CAP && calls === 3) return send(429, { error: "mock rate limit" });                 // exercise retry/backoff
    const text = JSON.parse(body).contents[0].parts[0].text;
    const wrap = (o) => send(200, { candidates: [{ finishReason: "STOP", content: { parts: [{ text: JSON.stringify(o) }] } }] });
    if (text.includes("propose 10 to 14")) {
      return wrap({ themes: Array.from({ length: 10 }, (_, i) => ({ id: `mock_theme_${i + 1}`, label: `Mock Theme ${i + 1}`, definition: `Placeholder definition ${i + 1}.` })) });
    }
    const ids = [...text.matchAll(/^\[(A\d+)\] /gm)].map((m) => m[1]);
    const themeIds = [...text.matchAll(/^(mock_theme_\d+):/gm)].map((m) => m[1]);
    const results = ids.map((id) => {
      const h = hash(id), relevant = h % 7 !== 0;
      return { id, relevant, themes: relevant ? [themeIds[h % themeIds.length], ...(h % 3 === 0 ? [themeIds[(h + 3) % themeIds.length]] : [])] : [], keywords: relevant ? [`mock term ${h % 15}`] : [] };
    });
    if (calls === 5) results.pop();                                                   // exercise validation failure + retry
    wrap({ results });
  });
}).listen(port, () => console.log("mock gemini on " + port));
