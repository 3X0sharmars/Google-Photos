"""Moments v1.1 - mobile photo library with moment search.

Screens (see templates/index.html): Photos timeline, Collections, Search.
Search = metadata first, then Gemini-expanded visual categories, then CLIP on the pixels (always says which route answered):
  1. metadata  : query words in photo titles/tags. >= EXACT_ENOUGH hits -> route "metadata".
  2. expansion : Gemini says what the query MEANS as 4-6 visual categories (+ what would be a wrong match); each is matched against
                 the pixels with CLIP, keeping only photos that genuinely match -> route "expanded".
  3. fallback  : Gemini unavailable -> direct CLIP search with a visible notice (route "visual"); not a photo subject -> route "none".
Picture search: /api/similar/<id> returns photos that look like a given photo.
"""
import json
import os
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import gemini_expand
from metadata import MetadataIndex
from retrieval import ABS_FLOOR, browse, diverse_thumbs, metadata_moments, visual_moments

ROOT = Path(__file__).resolve().parent.parent
EXACT_ENOUGH = 5
app = Flask(__name__, static_folder="static", template_folder="templates")

manifest = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
attr = pd.read_csv(ROOT / "data" / "attribution.csv", keep_default_na=False).set_index("photo_id")
E = np.load(ROOT / "data" / "embeddings.npy")
if len(manifest) != E.shape[0]: sys.exit(f"FATAL: manifest rows {len(manifest)} != embeddings rows {E.shape[0]}")
if (manifest.assigned_date == "").any(): sys.exit("FATAL: photos without assigned_date")
IDS = manifest.photo_id.tolist(); DATES = manifest.assigned_date.tolist(); N = len(IDS)
SRC = manifest.source.tolist()
DSRC = (manifest.date_source if "date_source" in manifest.columns else pd.Series(["synthetic"] * N)).tolist()
TAGS = [set(t for t in str(x).split("|") if t) for x in manifest.provided_tags]
ID2IDX = {p: i for i, p in enumerate(IDS)}
META = MetadataIndex(IDS)
EPISODES = json.loads((ROOT / "data" / "episodes.json").read_text(encoding="utf-8"))["episodes"]
NEWEST_FIRST = sorted(range(N), key=lambda i: (DATES[i], IDS[i]), reverse=True)

from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL = SentenceTransformer("clip-ViT-B-32", device="cpu")


def encode(text): return MODEL.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
def card(i): return dict(id=IDS[i], date=DATES[i], thumb=f"/thumb/{IDS[i]}.jpg", full=f"/photo/{IDS[i]}.jpg")
STRIP_MAX = 60      # photos sent per moment for the auto-scrolling strip (count still shows the true total)
def moment_json(m): return dict(start=m["start"], end=m["end"], count=m["count"], thumbs=[card(i) for i in m["thumbs"]],
                                photos=[card(i) for i in m.get("members", m["thumbs"])[:STRIP_MAX]])
def newest(idx): return sorted(idx, key=lambda i: (DATES[i], IDS[i]), reverse=True)


# ------------------------------------------------------------------ smart collections (CLIP text prompts over the pixels)
def smart_select(prompt, cap=400, floor=ABS_FLOOR, margin=0.07):
    s = E @ encode(prompt); order = np.argsort(-s)[:cap]; best = float(s[order[0]])
    lim = max(floor, best - margin)
    return [int(i) for i in order if s[i] >= lim], int(order[0])


SMART = {
    "people": dict(title="People", prompt="a portrait photo of a person or a group of people posing for the camera",
                   note="Face grouping is not in this MVP. These are photos that show people."),
    "documents": dict(title="Documents", prompt="a photo of a paper document, a receipt, a form or a screenshot of text",
                      note="Document-like photos. Few appear until the hand-collected document layer is added."),
}
PLACES = {
    "beaches": ("Beaches", "a beach with sea and sand"),
    "mountains": ("Mountains", "mountains and hills landscape"),
    "cities": ("Cities", "a city street with buildings"),
    "parks": ("Parks & gardens", "a park or garden with trees and flowers"),
    "water": ("Lakes & rivers", "a lake or river with calm water"),
}
for k, v in SMART.items(): v["idx"], v["cover"] = smart_select(v["prompt"], floor=0.20 if k == "documents" else 0.215)
_scans = [i for i in range(N) if "scan" in TAGS[i]]
if _scans:                      # scanned pages are documents by definition (scanner data), not a CLIP guess
    SMART["documents"]["idx"] = sorted(set(SMART["documents"]["idx"]) | set(_scans)); SMART["documents"]["cover"] = _scans[0]
for k, (title, prompt) in list(PLACES.items()):
    idx, cover = smart_select(prompt, floor=0.23)
    PLACES[k] = dict(title=title, prompt=prompt, idx=idx, cover=cover)

FOLDERS = {
    "camera": ("Camera", lambda i: SRC[i] == "phone" and "camera" in TAGS[i]),
    "screenshots": ("Screenshots", lambda i: "screenshot" in TAGS[i]),
    "whatsapp": ("WhatsApp Images", lambda i: "whatsapp" in TAGS[i]),
    "downloads": ("Downloads", lambda i: SRC[i] in ("wikimedia", "archive.org", "unknown")),
    "flickr": ("Flickr library (Creative Commons)", lambda i: SRC[i] == "openverse/flickr"),
}
FOLDER_IDX = {k: newest([i for i in range(N) if f(i)]) for k, (t, f) in FOLDERS.items()}
FOLDER_IDX = {k: v for k, v in FOLDER_IDX.items() if v}

# albums = trips and events (episodes); moments = every episode with 3+ photos
ALBUMS = []
for ep in EPISODES:
    if ep["type"] in ("trip", "event") and ep["n_photos"] >= 3:
        idx = [ID2IDX[p] for p in ep["photo_ids"] if p in ID2IDX]
        if len(idx) >= 3:
            cen = E[idx].mean(axis=0); cover = idx[int(np.argmax(E[idx] @ cen))]
            ALBUMS.append(dict(key=ep["episode_id"], type=ep["type"], start=ep["start"], end=ep["end"], idx=idx, cover=cover))
ALBUMS.sort(key=lambda a: a["end"], reverse=True)
MOMENT_EPS = sorted([ep for ep in EPISODES if ep["n_photos"] >= 3], key=lambda e: e["end"], reverse=True)
ALBUM_BY_KEY = {a["key"]: a for a in ALBUMS}

# ------------------------------------------------------------------ suggestions
_tag_counts = Counter()
for pid, m in json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8")).items():
    for t in m["tags"]:
        t = t.strip().lower()
        if len(t) >= 3 and re.fullmatch(r"[a-z][a-z ]+", t): _tag_counts[t] += 1
VOCAB = [t for t, c in _tag_counts.most_common() if c >= 3][:600]
TEXT_IDEAS = ["Goa", "Bangalore", "Beach", "Birthday", "Wedding", "Road trip", "Garden", "Concert", "Himalayas", "Street food"]
PICTURE_TOPICS = [("Beach", "beach", "a beach with sea and sand"), ("Birthday", "birthday", "a birthday cake with candles and a party"),
                  ("Food", "food", "a plate of food on a table"), ("Pets", "pets", "a pet dog or cat"), ("Wedding", "wedding", "a bride and groom at a wedding"),
                  ("Road trip", "road trip", "a car on an open road trip"), ("Garden", "garden", "a garden with flowers and plants"),
                  ("Concert", "concert", "a band performing on stage at a concert"), ("City", "city street", "a busy city street with buildings"),
                  ("Kitchen", "kitchen", "a home kitchen"), ("Mountains", "mountains", "mountains and hills landscape"), ("Sunset", "sunset", "a sunset over the horizon")]
PICTURES = []
for label, q, prompt in PICTURE_TOPICS:
    s_ = E @ encode(prompt); j = int(np.argmax(s_))
    if s_[j] >= 0.25: PICTURES.append(dict(label=label, query=q, cover=j))      # never show a tile whose best match is weak
print(f"ready: {N} photos; people={len(SMART['people']['idx'])} documents={len(SMART['documents']['idx'])} "
      f"places={ {k: len(v['idx']) for k, v in PLACES.items()} } albums={len(ALBUMS)} moments={len(MOMENT_EPS)}; "
      f"gemini key {'set' if os.environ.get('GEMINI_API_KEY') else 'NOT set'}", flush=True)


def _label(a):
    s, e = date.fromisoformat(a["start"]), date.fromisoformat(a["end"])
    f = lambda d: d.strftime("%d %b %Y").lstrip("0")
    return f(s) if a["start"] == a["end"] else f"{f(s)} – {f(e)}"


# ------------------------------------------------------------------ routes
@app.get("/")
def index(): return send_from_directory(Path(app.template_folder if os.path.isabs(app.template_folder) else Path(app.root_path) / app.template_folder), "index.html", max_age=0)   # static page: not run through Jinja


@app.get("/api/timeline")
def api_timeline():
    off = max(int(request.args.get("offset", 0) or 0), 0); lim = min(max(int(request.args.get("limit", 240) or 240), 1), 500)
    return jsonify(total=N, photos=[card(i) for i in NEWEST_FIRST[off:off + lim]])


@app.get("/api/collections")
def api_collections():
    return jsonify(collections=[
        dict(key="albums", title="Albums", count=len(ALBUMS), cover=card(ALBUMS[0]["cover"]) if ALBUMS else None),
        dict(key="device", title="On this device", count=N, cover=card(NEWEST_FIRST[0])),
        dict(key="people", title="People", count=len(SMART["people"]["idx"]), cover=card(SMART["people"]["cover"])),
        dict(key="documents", title="Documents", count=len(SMART["documents"]["idx"]), cover=card(SMART["documents"]["cover"])),
        dict(key="places", title="Places", count=len(PLACES), cover=card(PLACES["beaches"]["cover"]), unit="places"),
        dict(key="moments", title="Moments", count=len(MOMENT_EPS), cover=card(NEWEST_FIRST[1]), unit="moments"),
    ])


@app.get("/api/collection/<key>")
def api_collection(key):
    if key in SMART:
        c = SMART[key]; return jsonify(title=c["title"], note=c["note"], photos=[card(i) for i in newest(c["idx"])])
    if key == "device":
        return jsonify(title="On this device", note="Folders on the phone, plus downloaded and imported pictures.",
                       items=[dict(key=k, title=FOLDERS[k][0], sub="", count=len(v), cover=card(v[0]), kind="album") for k, v in FOLDER_IDX.items()])
    if key == "places":
        return jsonify(title="Places", note="Scenes recognised from the pixels. Map locations are not in this MVP.",
                       items=[dict(key=k, title=v["title"], sub="", count=len(v["idx"]), cover=card(v["cover"]), kind="place") for k, v in PLACES.items()])
    if key == "albums":
        return jsonify(title="Albums", note="Automatic albums from trips and events.",
                       items=[dict(key=a["key"], title=("Trip" if a["type"] == "trip" else "Event"), sub=_label(a), count=len(a["idx"]),
                                   cover=card(a["cover"]), kind="album") for a in ALBUMS])
    if key == "moments":
        out = []
        for ep in MOMENT_EPS[:150]:
            idx = [ID2IDX[p] for p in ep["photo_ids"] if p in ID2IDX]
            out.append(moment_json(dict(start=ep["start"], end=ep["end"], count=len(idx), thumbs=diverse_thumbs(idx, E, DATES), members=sorted(idx, key=lambda i: DATES[i]))))
        return jsonify(title="Moments", note="Stretches of time that belong together.", moments=out)
    return jsonify(error="Unknown collection"), 404


@app.get("/api/place/<key>")
def api_place(key):
    if key not in PLACES: return jsonify(error="Unknown place"), 404
    return jsonify(title=PLACES[key]["title"], photos=[card(i) for i in newest(PLACES[key]["idx"])])


@app.get("/api/album/<key>")
def api_album(key):
    if key in FOLDER_IDX: return jsonify(title=FOLDERS[key][0], photos=[card(i) for i in FOLDER_IDX[key]])
    a = ALBUM_BY_KEY.get(key)
    if not a: return jsonify(error="Unknown album"), 404
    return jsonify(title=("Trip" if a["type"] == "trip" else "Event") + " · " + _label(a), photos=[card(i) for i in newest(a["idx"])])


@app.get("/api/suggestions")
def api_suggestions():
    return jsonify(text=TEXT_IDEAS, pictures=[dict(label=p["label"], query=p["query"], thumb=card(p["cover"])["thumb"]) for p in PICTURES])


@app.get("/api/suggest")
def api_suggest():
    q = re.sub(r"\s+", " ", request.args.get("q", "").strip().lower())
    if not q: return jsonify(suggestions=[])
    cached = json.loads(gemini_expand.CACHE_PATH.read_text(encoding="utf-8")) if gemini_expand.CACHE_PATH.exists() else {}
    ai = [k for k in cached if k.startswith(q)]
    tags = [t for t in VOCAB if t.startswith(q) or (" " + q) in t]
    seen, out = set(), []
    for t in ai + tags:
        if t not in seen and t != q: seen.add(t); out.append(t.title())
    return jsonify(suggestions=out[:7])


@app.get("/api/similar/<pid>")
def api_similar(pid):
    if pid not in ID2IDX: return jsonify(error="Unknown photo"), 404
    i = ID2IDX[pid]; s = E @ E[i]; s[i] = -1
    order = np.argsort(-s)[:48]; best = float(s[order[0]])
    keep = [int(j) for j in order if s[j] >= max(0.5, best - 0.10)][:30]     # relative to the closest photo; image-image CLIP scores are flat (0.6-0.85)
    return jsonify(source=card(i), photos=[card(j) for j in keep], note=None if keep else "Nothing else in this library looks like it.")


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q: return jsonify(error="Empty query"), 400
    if len(q) > 120: return jsonify(error="Query too long"), 400
    hits = META.search(q); exact = None
    if hits:
        mm, flat = metadata_moments(hits, E, DATES, 5)
        exact = dict(photo_count=len(hits), moments=[moment_json(m) for m in mm], photos=[card(i) for i in flat] if not mm else [])
    if len(hits) >= EXACT_ENOUGH: return jsonify(route="metadata", query=q, exact=exact)

    exp = gemini_expand.expand(q)
    if exp and exp["categories"]:
        cats = []
        for c in exp["categories"]:
            neg = E @ encode(c["avoid_query"]) if c.get("avoid_query") else None
            moments, flat, best = visual_moments(E @ encode(c["visual_query"]), E, DATES, 3, neg)
            cats.append(dict(label=c["label"], visual_query=c["visual_query"], match=round(best, 3),
                             moments=[moment_json(m) for m in moments], photos=[card(i) for i in flat[:12]] if not moments else []))
        return jsonify(route="expanded", query=q, interpretation=exp["interpretation"], kind=exp["kind"], exact=exact,
                       categories=cats, model=exp["model"], cached=exp["cached"])
    if exp is not None:        # Gemini answered but judged it not a photo subject: say so, show no junk matches
        if exact: return jsonify(route="metadata", query=q, exact=exact)
        return jsonify(route="none", query=q, notice=f"Nothing to suggest for “{q}”. Try a place, event, object or activity.")
    why = gemini_expand.last_error()   # Gemini unavailable: visible fallback to a direct visual search
    moments, flat, best = visual_moments(E @ encode(q), E, DATES, 5)
    return jsonify(route="visual", query=q, exact=exact, notice="AI suggestions are unavailable right now, so this is a direct visual match.",
                   error_detail=why, moments=[moment_json(m) for m in moments], photos=[card(i) for i in flat] if not moments else [])


@app.get("/api/browse")
def api_browse():
    try: idx, lo, hi = browse(request.args.get("start", ""), request.args.get("end", ""), DATES)
    except ValueError: return jsonify(error="Bad date range"), 400
    return jsonify(start=lo, end=hi, photos=[card(i) for i in idx])


ORIGIN = {"phone": "From your phone", "instagram": "Instagram screenshot", "archive.org": "archive.org (source guessed from the file name)",
          "wikimedia": "Wikimedia Commons (source guessed from the file name)", "unknown": "Downloaded image (source not recorded)"}


@app.get("/api/credit/<pid>")
def api_credit(pid):
    if pid not in attr.index: return jsonify(error="Unknown photo"), 404
    r = attr.loc[pid]; i = ID2IDX[pid]
    return jsonify(creator=r["creator"], title=r["title"], licence=r["licence"], licence_url=r["licence_url"], source=r["source_page"],
                   user_supplied=pid.startswith("u_"), origin=ORIGIN.get(SRC[i], ""), date=DATES[i], date_source=DSRC[i])


@app.get("/thumb/<pid>.jpg")
def thumb(pid): return send_from_directory(ROOT / "photos" / "thumbs", f"{pid}.jpg", max_age=86400)


@app.get("/photo/<pid>.jpg")
def photo(pid): return send_from_directory(ROOT / "photos" / "serving", f"{pid}.jpg", max_age=86400)


@app.errorhandler(404)
def not_found(e): return jsonify(error="Not found"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
