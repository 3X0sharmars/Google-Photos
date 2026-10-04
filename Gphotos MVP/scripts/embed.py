"""CLIP image embeddings (clip-ViT-B-32) of photos/serving/, row order identical to data/manifest.csv.
Pixels only. Incremental + checkpointed every 256 images; fails loudly on any mismatch.
Run from the moments/ folder:  python scripts/embed.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "embeddings_cache.npz"
m = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
ids = list(m.photo_id)
cache = {}
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=False); cache = dict(zip(z["ids"].tolist(), z["vecs"]))
todo = [p for p in ids if p not in cache]
print(f"manifest={len(ids)} cached={len(cache)} to_embed={len(todo)}", flush=True)


def save():
    k = list(cache); np.savez(CACHE, ids=np.array(k), vecs=np.stack([cache[x] for x in k]))


if todo:
    model = SentenceTransformer("clip-ViT-B-32", device="cpu")
    t0 = time.time(); B = 32
    for s in range(0, len(todo), B):
        chunk = todo[s:s + B]; imgs = []
        for p in chunk:
            f = ROOT / "photos" / "serving" / f"{p}.jpg"
            if not f.exists(): sys.exit(f"FATAL: missing {f}")
            imgs.append(Image.open(f).convert("RGB"))
        v = model.encode(imgs, batch_size=B, convert_to_numpy=True, normalize_embeddings=True)
        for p, x in zip(chunk, v): cache[p] = x.astype(np.float32)
        done = s + len(chunk)
        if done % 256 < B or done == len(todo):
            save(); print(f"  {done}/{len(todo)}  {done/(time.time()-t0):.2f} img/s", flush=True)
    save()
E = np.stack([cache[p] for p in ids]).astype(np.float32)
if E.shape[0] != len(m): sys.exit(f"FATAL: embeddings {E.shape[0]} != manifest {len(m)}")
if not np.isfinite(E).all(): sys.exit("FATAL: non-finite embeddings")
np.save(ROOT / "data" / "embeddings.npy", E)
print(f"OK embeddings.npy {E.shape} == manifest rows {len(m)}")
