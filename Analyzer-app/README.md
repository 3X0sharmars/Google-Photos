# Photo retrieval evidence: analyzer app

Next.js app (Vercel or Render). Shows corpus health (section A) and a Gemini-coded analyzer (section B).

## How it works
* **Gemini runs offline**, not in the browser or the deployed server. `scripts/analyze.mjs` codes every record once and writes
  `data/analyzed.json`. The deployed app only reads JSON, so it needs **no API key**, has no per-visit cost, and never exposes one.
* **Section A**: scraped total (5,870), unique sources (454 distinct threads/videos/pages, 30 platforms), retained after validation (3,982).
* **Section B**: word cloud sized by frequency (Themes / Key terms toggle), themes by volume (bars), and an evidence window that
  draws random verbatim quotes for the selected theme or term, with platform and year filters and a Shuffle button.

## Run the analysis (once)
```
npm install
copy .env.example .env.local        # then put your key in GEMINI_API_KEY
npm run analyze:themes              # Gemini proposes 10-14 themes from 300 random records -> data/themes.json
                                    # REVIEW / EDIT data/themes.json (rename, merge, delete) before continuing
npm run analyze:classify            # codes all 3,982 records (resumable, ~25 min at the default 8 req/min) -> data/analyzed.json
npm run dev                         # http://localhost:3000
```
`npm run analyze:status` shows progress. If it stops (rate limit, network), run `analyze:classify` again; it resumes from
`data/analysis_checkpoint.jsonl`. Each record gets `relevant` (on-topic or not), up to 2 theme ids, and up to 3 key terms.
Off-topic records are excluded from every chart. Comments are passed to the model as data with an instruction to ignore any
instructions inside them, and output is constrained by a JSON schema and re-validated; records that cannot be validated are
reported as failed, never guessed.

Model defaults to `gemini-3.8-flash`; set `GEMINI_MODEL` to change. The script checks the model exists and lists valid ones if not.

## Deploy
Commit `data/analyzed.json` and `data/themes.json` after the analysis (the app ships with empty placeholders and shows setup
instructions until then).
* **Vercel**: import the repo, root = `analyzer-app`, no environment variables needed.
* **Render**: `render.yaml` is included (Node web service: `npm install && npm run build`, `npm start`).

## Refreshing the data
`python scripts/prepare_data.py` regenerates `data/dataset.json` and `data/stats.json` from `../analyzer_ready/`. Re-run the
analysis afterwards (delete `data/analysis_checkpoint.jsonl` first if the records changed).

## Notes
* Author names are not in the data. Quotes are shown verbatim (including odd characters or HTML entities from the source).
* Public posts and comments from YouTube/Reddit/forums are shown with links back to the source. Check each platform's terms
  before making a deployment public; a password or private link is the safe default for coursework.
* `scripts/dev/mock_gemini.mjs` is a fake endpoint used only to test the pipeline. Never use its output as analysis.
