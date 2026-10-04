"""Build data/metadata.json: the searchable metadata layer = Flickr title + Flickr tags per photo.
(Keywords/titles are metadata supplied by the photo's owner, comparable to labels/captions in a real photo library.)
Run from moments/:  python scripts/build_metadata.py
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
m = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
a = pd.read_csv(ROOT / "data" / "attribution.csv", keep_default_na=False).set_index("photo_id")
out = {}
for r in m.itertuples(index=False):
    title = a.at[r.photo_id, "title"] if r.photo_id in a.index else ""
    tags = [t.strip() for t in str(r.provided_tags).split("|") if t.strip()]
    out[r.photo_id] = dict(title=str(title), tags=tags)
(ROOT / "data" / "metadata.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"metadata for {len(out)} photos; {sum(1 for v in out.values() if v['tags'])} have tags, {sum(1 for v in out.values() if v['title'])} have titles")
