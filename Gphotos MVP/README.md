# Moments v1.1: metadata first, then Gemini-expanded visual categories

Search returns **moments in time**, not a list. v1.1 changes the corpus and adds query understanding.

## What changed from v1.0
* **Corpus:** 3,007 real Flickr photos (Creative Commons, via Openverse) replace the 3,200 Unsplash stock photos, so the library looks like
  ordinary people's photos. Creator, title, licence and a Flickr link are in `data/attribution.csv` and shown when a photo is opened.
  Licences: CC BY 1,100 / BY-SA 636 / BY-NC 454 / BY-NC-SA 817 (attribution required; NC = non-commercial use only).
* **Search logic** (`app/main.py`):
  1. **Metadata.** Query words are matched against each photo's Flickr title and tags (`app/metadata.py`). 5 or more photos matched
     -> shown as moments, route `metadata`.
  2. **Gemini expansion.** Otherwise (including 1 to 4 weak tag hits, which are shown first) Gemini says what the query *means* as 4 to 6 visual
     categories ("Goa" -> beaches, Portuguese architecture, seafood, nightlife, churches). Each category's `visual_query` is matched against
     the **pixels** with CLIP and grouped into moments, route `expanded`. Categories with no convincing visual match are shown as
     "Nothing in this library looks like that" rather than padded with junk.
  3. **Fallbacks, always visible.** Gemini down or out of quota -> a notice plus a direct CLIP search on the query (route `visual`).
     Gemini says the query is not a photo subject -> a "nothing to suggest" message (route `none`).
* Gemini runs **server-side**; the key never reaches the browser. Answers are cached in `data/suggestion_cache.json`; models rotate when one hits
  its daily free-tier cap (`GEMINI_MODELS`). Pre-cache common places with `python scripts/prewarm.py`.

## Run
```
pip install flask pandas numpy scikit-learn sentence-transformers pillow requests
copy .env.example .env             # add GEMINI_API_KEY (this app's own .env is git-ignored)
python scripts/embed.py            # only if data/embeddings.npy is missing (~14 min on CPU)
python scripts/build_metadata.py   # only if data/metadata.json is missing
python app/main.py                 # http://localhost:5000
```

## Mobile app shell (added after first review)
* **Mobile-first.** The page is a phone layout; on a desktop it shows as a centred phone-width window. Bottom tabs: **Photos**, **Collections**, **Search**.
  A paintbrush **Create** button sits top-right and is deliberately inactive in this MVP (it shows "Create is coming soon").
* **Photos** (landing): every photo as a thumbnail grid, newest to oldest, grouped by day, endless scroll.
* **Collections:** Albums (automatic, from trips and events), On this device, People, Documents, Places, Moments. People, Documents and Places are
  *smart collections* from CLIP text prompts over the pixels (no face recognition, no map data yet), so they are approximate; each screen says so.
* **Search suggestions:** recent searches, "Try searching" text ideas, typeahead from photo tags and saved AI answers, and **Search by picture**:
  topic tiles with a real cover photo (a tile is hidden if its best match is weak) plus **Find similar photos** inside the photo viewer
  (image-to-image similarity; an uploaded-picture query is not built).

## Relevance fix (Goa returned an icy/Nordic shoreline)
Cause: each AI category kept the top-100 CLIP matches however weak (similarity down to ~0.19 when the best was 0.32), and the
"4 visually different thumbnails" rule then surfaced the weakest, most unlike members. Fixes in `app/retrieval.py` and `app/gemini_expand.py`:
a photo must be within 0.06 of the category's best match and above 0.225; Gemini also returns an `avoid_query` (what would be a *wrong* match, e.g.
"snowy mountains, ice, cold grey northern coastline") and a photo that resembles that more than the target is dropped; categories with no
temporal cluster show a flat "closest matches" grid instead of padding. Clusters are now smaller but cleaner.

## Timeline fix (a Goa day showed snow) and your own photos
**Cause.** The first timeline filled episodes in Flickr upload order, so a "trip" was a random mix of subjects, and episodes closer together than the
+/-2 day browse window bled into each other (one 5-day window held 4 episodes: food, a bookshelf, a forest and snowy fjords).
**Fix** (`scripts/retimeline.py`, deterministic): episodes are built from visually similar photos (trip seeds from travel-looking photos, event seeds from
people/celebration photos, documents from the most document-like), every simulated episode is at least 3 days from any other, and a repair step swaps any
snow photo out of a non-snow episode. The Flickr-ID rule is kept at episode level (older upload era -> earlier dates). Measured: photo-to-episode similarity
trips 0.715 -> 0.818, events 0.734 -> 0.768; 0 of 179 episodes have another episode inside their browse window; 0 episodes with a lone snow photo.
**Your photos** (`scripts/ingest_user_photos.py`, folder `Photo Dwonload`, 100 files): phone camera, WhatsApp, screenshots (LinkedIn, Instagram, Samsung
Browser), downloaded images. Originals are untouched; the app serves resized copies without EXIF/GPS. Dates are **real where known** (EXIF capture date, or
the date in the file name) and labelled so in the photo viewer; the rest get simulated dates. Scanner EXIF (the 2018 book scans) is a scan date, not a
capture date, so those are simulated. On this device = Camera, Screenshots, WhatsApp Images, Downloads, Flickr library; scans join Documents.
**Source and licence of downloaded images are not recorded** (guessed groups only). Fill `data/user_sources_TEMPLATE.csv`, save it as `data/user_sources.csv`
(columns filename,source,title,tags,creator,licence,source_url,date) and re-run the ingest, `build_metadata.py` and `retimeline.py`. Do this before any public
deploy: Wikimedia/archive.org items need attribution, and screenshots of other people's posts carry copyright and privacy concerns.

## Auto-scrolling moments
Every moment card (search results, Collections > Moments) is a filmstrip of **all** its matching photos (up to 60 are sent; the card shows the true count),
scrolling left to right on its own, then pausing and looping back to the first photo. The speed scales with the number of photos (the whole strip takes
8-30 s). It pauses while you touch, swipe or hover, resumes a few seconds later, only runs for cards on screen, and is off when the device asks for reduced motion
(you can still swipe). Tap a photo to open it; tap the card header ("View all") to browse the whole stretch +/-2 days.

## Deploy
* **Render** (works): `render.yaml` is included. Root directory = `Gphotos MVP`. Needs about **2 GB RAM** (CLIP + PyTorch), i.e. the *Standard* plan; free and
  Starter instances run out of memory. Set `GEMINI_API_KEY` in the Render dashboard (never commit it). First request after idle can take 30-60 s while the model loads.
* **Vercel** (not suitable as-is): this is a Flask app that loads PyTorch at start-up, which exceeds Vercel's serverless size and memory limits. It would need the CLIP
  text encoder moved to a hosted embedding service or an ONNX build first.
* If Gemini is unavailable (no key or quota) search still works: it falls back to a direct visual match with a visible notice.

## About this repository's content
* Photos: 3,007 Creative Commons photos from Flickr via Openverse (creator, licence and link in `data/attribution.csv`; licences are BY, BY-SA, BY-NC, BY-NC-SA, so
  non-commercial use only for some) **plus 100 photos added by the project owner**: phone camera, WhatsApp images, screenshots (including screenshots of LinkedIn and
  Instagram posts by other people) and downloaded Wikimedia Commons / archive.org images whose source and licence are not recorded (`data/user_sources_TEMPLATE.csv`).
  The owner chose to include them in this public repository. If you are a rights holder or appear in any image and want it removed, open an issue.
* Dates are simulated except where the photo carries its own (labelled in the app).
* Secrets: `.env` is git-ignored. No API key is stored in this repository.

## Rules that changed, and why it matters for the study
* v1.0 forbade using provided keywords for retrieval, so measured success could not be inflated by human-written labels. **v1.1 deliberately
  uses titles and tags as a metadata layer** (like captions/labels in a real library) and says which route answered. When reporting test
  results, report metadata-route and visual-route outcomes separately; metadata hits are not evidence of visual retrieval quality.
* The CLIP image index itself is still pixels only.

## Limits and known issues
* Dates are synthetic (assigned along Flickr upload-ID order into 177 episodes), and episodes are **not thematically coherent** (a "trip" can
  mix subjects). Browse stretches may therefore look random. Grouping by visual similarity would fix this but changes the ID-ordering rule.
* No hand-collected document/screenshot layer yet, so document retrieval cannot be tested.
* Gemini suggestions are model opinions: they can include things this library does not contain (handled by the relevance floor in
  `app/retrieval.py`, `MIN_MEAN_SIM`) and can be culturally or regionally generic.
* The free Gemini tier is small (about 20 requests per model per day); uncached queries use quota.

## Stated limitation (unchanged)
The test library is a constructed corpus, not the participants' own photos. This inflates measured success: searching an unfamiliar library is
**target matching**, not **memory retrieval**. Read results as evidence the mechanism works, not of real-world retrieval improvement.
