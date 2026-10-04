import { randomEvidence, hasAnalysis } from "../../../lib/data";

export const dynamic = "force-dynamic";

export function GET(req) {
  if (!hasAnalysis()) return Response.json({ error: "Analysis has not been run yet (npm run analyze)." }, { status: 409 });
  const p = new URL(req.url).searchParams;
  const n = Math.min(Math.max(parseInt(p.get("n") || "5", 10) || 5, 1), 12);
  const theme = p.get("theme") || undefined, term = p.get("term") || undefined;
  if (!theme && !term) return Response.json({ error: "theme or term is required" }, { status: 400 });
  const exclude = (p.get("exclude") || "").split(",").filter(Boolean).slice(0, 200);
  return Response.json(randomEvidence({ theme, term, platform: p.get("platform") || undefined, year: p.get("year") || undefined, n, exclude }),
    { headers: { "Cache-Control": "no-store" } });
}
