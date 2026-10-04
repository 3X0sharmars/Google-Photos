"""Add the user's own photos (folder 'Photo Dwonload') to the library. Originals are never modified.

For each file: serving (800px) + thumbnail (256px) copies (EXIF stripped, orientation applied), a manifest row, an attribution row, and metadata.
Dates (never invented here):
  * EXIF capture date (DateTimeOriginal)        -> date_source 'exif'      (not used for scanner EXIF: that is the scan date, not the capture date)
  * date in the file name (IMG-YYYYMMDD-WA, Screenshot_YYYYMMDD_..., Screenshot_D-M-YYYY_...) -> 'filename'
  * otherwise -> left empty here; scripts/retimeline.py assigns a simulated date (date_source 'synthetic').
Source / licence are NOT known for downloaded images: they are guessed from the file name only to group them, and the licence is recorded as
'not recorded'. Put corrections in data/user_sources.csv (filename,source,title,tags,creator,licence,source_url,date) and re-run.
Run from moments/:  python scripts/ingest_user_photos.py
"""
import csv, re, sys
from datetime import date
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent.parent / "Photo Dwonload"
OVR = ROOT / "data" / "user_sources.csv"
EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
if not SRC.exists(): sys.exit(f"FATAL: {SRC} not found")

man = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False, dtype=str)
att = pd.read_csv(ROOT / "data" / "attribution.csv", keep_default_na=False, dtype=str)
if "date_source" not in man.columns: man["date_source"] = "synthetic"
man = man[~man.photo_id.str.startswith("u_")]; att = att[~att.photo_id.str.startswith("u_")]      # idempotent re-run
overrides = {}
if OVR.exists():
    for r in csv.DictReader(open(OVR, encoding="utf-8-sig")): overrides[r["filename"]] = r


def slug(s): return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:70]


def classify(name):
    """-> (source, kind, title, tags). Guess from the file name only."""
    low = name.lower(); stem = re.sub(r"\.[a-z0-9]+$", "", name, flags=re.I); stem = re.sub(r"\s*\(\d+\)$|\s*\[\d+\]$", "", stem)
    if re.match(r"^\d{8}_\d{6}$", stem): return "phone", "camera", "", ["camera"]
    if re.match(r"^IMG-\d{8}-WA\d+$", stem): return "phone", "whatsapp", "", ["whatsapp"]
    m = re.match(r"^Screenshot_\d{8}_\d{6}_(.+)$", stem)
    if m:
        app = m.group(1).replace("_", " "); return "phone", "screenshot", "", ["screenshot", app.lower()]
    if re.match(r"^Screenshot_\d+-\d+-\d{4}_", stem) and "instagram" in low: return "instagram", "screenshot", "", ["screenshot", "instagram"]
    if stem.startswith("MS-001_"): return "archive.org", "scan", "", ["scan"]
    if re.match(r"^photo-[\w-]+_\d+_o$", stem): return "archive.org", "photo", "", []
    if re.fullmatch(r"[0-9a-f]{20,}", stem) or re.fullmatch(r"\d{6,}", stem): return "unknown", "photo", "", []
    return "wikimedia", "photo", re.sub(r"\s+", " ", stem.replace("_", " ")).strip(), []


def filename_date(name):
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", name)           # 20260802, IMG-20260412-WA..
    if m:
        try: return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except ValueError: pass
    m = re.search(r"Screenshot_(\d{1,2})-(\d{1,2})-(20\d{2})_", name)      # Screenshot_4-10-2026_...
    if m:
        try: return date(int(m[3]), int(m[2]), int(m[1])).isoformat()
        except ValueError: pass
    return ""


def exif_date(im, kind):
    if kind == "scan": return ""                                            # scanner EXIF = scan date, not capture date
    ex = im.getexif(); d = (ex.get_ifd(0x8769) or {}).get(36867) or ex.get(306)
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})", str(d or ""))
    if not m or int(m[1]) < 1990: return ""
    try: return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    except ValueError: return ""


(ROOT / "photos" / "serving").mkdir(parents=True, exist_ok=True); (ROOT / "photos" / "thumbs").mkdir(parents=True, exist_ok=True)
rows, arows, used = [], [], set()
files = sorted(p for p in SRC.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
for f in files:
    o = overrides.get(f.name, {})
    source, kind, title, tags = classify(f.name)
    source = o.get("source") or source; title = o.get("title") or title
    if o.get("tags"): tags = [t.strip() for t in o["tags"].split("|") if t.strip()]
    pid = "u_" + slug(f.stem)
    while pid in used: pid += "_x"
    used.add(pid)
    with Image.open(f) as im0:
        d_exif = exif_date(im0, kind)
        im = ImageOps.exif_transpose(im0).convert("RGB"); w, h = im.size
        a = im.copy(); a.thumbnail((800, 800)); a.save(ROOT / "photos" / "serving" / f"{pid}.jpg", quality=85)
        b = im.copy(); b.thumbnail((256, 256)); b.save(ROOT / "photos" / "thumbs" / f"{pid}.jpg", quality=80)
    d, dsrc = (o.get("date"), "override") if o.get("date") else ((d_exif, "exif") if d_exif else ((filename_date(f.name), "filename") if filename_date(f.name) else ("", "synthetic")))
    rows.append(dict(photo_id=pid, flickr_id="", source=source, original_url="user:" + f.name, local_path=f"Photo Dwonload/{f.name}", width=w, height=h,
                     provided_tags="|".join(tags), episode_id="", assigned_date=d, date_source=dsrc))
    guessed = "" if o.get("source") else " (source guessed from file name)"
    arows.append(dict(photo_id=pid, creator=o.get("creator", ""), title=title or f.stem, licence=o.get("licence") or "not recorded",
                      licence_url="", source_page=o.get("source_url", "") ))
    arows[-1]["title"] = (title or f.stem)

man = pd.concat([man, pd.DataFrame(rows)], ignore_index=True)
att = pd.concat([att, pd.DataFrame(arows)], ignore_index=True)
man.to_csv(ROOT / "data" / "manifest.csv", index=False); att.to_csv(ROOT / "data" / "attribution.csv", index=False)
import collections
print(f"ingested {len(rows)} user photos")
print("by source:", dict(collections.Counter(r["source"] for r in rows)))
print("date source:", dict(collections.Counter(r["date_source"] for r in rows)))
print(f"manifest now {len(man)} rows ({(man.source.isin(['phone','instagram','archive.org','wikimedia','unknown'])).sum()} user photos)")
