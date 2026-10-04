"""Coherent synthetic timeline (replaces the upload-order timeline that mixed unrelated photos into one 'trip').

Real dates are kept: photos whose date_source is exif / filename / override stay on that date, one episode per real day.
All other photos get simulated dates, in episodes that are VISUALLY COHERENT:
  * trips      : seeds drawn from travel/landscape-looking photos, then the nearest photos by CLIP similarity
  * events     : seeds from people/celebration-looking photos
  * documents  : the most document-like remaining photos
  * ordinary   : small visually-similar groups
The Flickr-ID rule is kept at episode level: within each episode type, episodes whose photos come from an older upload era (median Flickr id)
get earlier dates. Every episode window is >= 3 days away from every other one (and from real-dated photos), so the +/-2 day browse view of one
moment can never contain another episode's photos.
Run from moments/:  python scripts/retimeline.py
"""
import json, sys
import datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
START, END = dt.date(2022, 1, 1), dt.date(2025, 12, 31)
GAP = 2                                    # browse pads +/-2 days, so the next episode must start >= end + GAP + 1
rng = np.random.default_rng(SEED)

m = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False, dtype=str)
E = np.load(ROOT / "data" / "embeddings.npy")
if len(m) != len(E): sys.exit("FATAL: manifest/embeddings mismatch")
real = m.date_source.isin(["exif", "filename", "override"]).values
syn = np.where(~real)[0]; N = len(syn)
print(f"{len(m)} photos: {real.sum()} real-dated (kept), {N} to simulate")

# ---------------------------------------------------------------- plan: how many episodes of each type and size
n_ep = int(np.clip(round(N / 17), 120, 180))
counts = dict(trip=round(n_ep * .25), event=round(n_ep * .20), ordinary=round(n_ep * .40)); counts["doc"] = n_ep - sum(counts.values())
bounds = dict(trip=(20, 60), event=(10, 30), ordinary=(1, 8), doc=(1, 4))
sizes = {t: [int(x) for x in rng.integers(lo, hi + 1, counts[t])] for t, (lo, hi) in bounds.items()}
total = lambda: sum(sum(v) for v in sizes.values())
g = 0
while total() != N and g < 500000:
    g += 1; t = str(rng.choice(["trip", "event", "trip", "ordinary", "event", "ordinary"])); i = int(rng.integers(len(sizes[t]))); lo, hi = bounds[t]
    if total() < N and sizes[t][i] < hi: sizes[t][i] += 1
    elif total() > N and sizes[t][i] > lo: sizes[t][i] -= 1
if total() != N: sys.exit(f"FATAL: cannot fit {N} photos into the plan")

# ---------------------------------------------------------------- visually coherent grouping
from sentence_transformers import SentenceTransformer
clip = SentenceTransformer("clip-ViT-B-32", device="cpu")
enc = lambda t: clip.encode(t, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
s_trip = E @ enc("a travel photo of a landscape, beach, mountains or a sightseeing place outdoors")
s_event = E @ enc("people at a party, wedding, concert or birthday celebration")
s_doc = E @ enc("a photo of a paper document, a receipt, a form or a screenshot of text")
remaining = set(int(i) for i in syn)


def knn(seed, size):
    cand = np.array(sorted(remaining)); pick = cand[np.argsort(-(E[cand] @ E[seed]))[:min(len(cand), size * 2)]]
    cen = E[pick].mean(axis=0); cen /= np.linalg.norm(cen)
    return [int(x) for x in cand[np.argsort(-(E[cand] @ cen))[:size]]]


def pick_seed(score):
    cand = np.array(sorted(remaining)); top = cand[np.argsort(-score[cand])[:max(5, len(cand) // 3)]]
    return int(rng.choice(top))


groups = {t: [] for t in sizes}
for sz in sorted(sizes["trip"], reverse=True):
    gr = knn(pick_seed(s_trip), sz); groups["trip"].append(gr); remaining -= set(gr)
for sz in sorted(sizes["event"], reverse=True):
    gr = knn(pick_seed(s_event), sz); groups["event"].append(gr); remaining -= set(gr)
for sz in sizes["doc"]:
    cand = np.array(sorted(remaining)); gr = [int(x) for x in cand[np.argsort(-s_doc[cand])[:sz]]]; groups["doc"].append(gr); remaining -= set(gr)
for sz in sorted(sizes["ordinary"], reverse=True):
    gr = knn(int(rng.choice(sorted(remaining))), sz); groups["ordinary"].append(gr); remaining -= set(gr)
assert not remaining, f"{len(remaining)} photos unassigned"

# ---------------------------------------------------------------- contradiction repair: no snow photo inside a non-snow episode
# (a lone snowy photo in a beach/city episode is what made a "Goa" day show a fjord). Swap it with a non-snow photo from a snowy-mountain
# episode that fits the beach/city episode better, so both episodes stay coherent and every size is unchanged.
SNOW, SNOWY_EP = 0.25, 0.10
snow = E @ enc("snow-covered mountains, ice, glacier, snowy winter landscape")
cent = lambda gr: (lambda c: c / np.linalg.norm(c))(E[gr].mean(axis=0))
frac = lambda gr: float(np.mean(snow[gr] > SNOW))
allg = [(t, i) for t in groups for i in range(len(groups[t]))]
snowy_eps = [(t, i) for t, i in allg if len(groups[t][i]) >= 10 and frac(groups[t][i]) >= SNOWY_EP]
swaps = 0
for t, gi in allg:
    A = groups[t][gi]
    if len(A) < 4 or frac(A) >= SNOWY_EP: continue
    for p_ in [x for x in A if snow[x] > SNOW]:
        cA, best = cent(A), None
        for tb, gb in snowy_eps:
            B = groups[tb][gb]; qs = [q for q in B if snow[q] < 0.2]
            if not qs: continue
            q = max(qs, key=lambda q: float(E[q] @ cA)); gain = float(E[p_] @ cent(B)) + float(E[q] @ cA)
            if best is None or gain > best[0]: best = (gain, tb, gb, q)
        if best and float(E[p_] @ cent(groups[best[1]][best[2]])) > 0.5:
            _, tb, gb, q = best; B = groups[tb][gb]; A[A.index(p_)] = q; B[B.index(q)] = p_; swaps += 1
print(f"snow-consistency repair: {swaps} swaps ({len(snowy_eps)} snowy-mountain episodes)")

# ---------------------------------------------------------------- era (older Flickr uploads -> earlier dates, per episode type)
fid = pd.to_numeric(m.flickr_id, errors="coerce").values
glob_med = float(np.nanmedian(fid))
era = lambda gr: float(np.nanmedian(fid[gr])) if np.isfinite(fid[gr]).any() else glob_med

# ---------------------------------------------------------------- windows: bursty, non-overlapping, GAP days from everything
months = [(y, mo) for y in range(START.year, END.year + 1) for mo in range(1, 13)]
w = rng.lognormal(0, 1.1, len(months)); w /= w.sum()
taken = []
for d in pd.to_datetime(m.assigned_date[real]).dt.date.unique(): taken.append((d, d))          # real dates are fixed points


def free(s, e): return all(e + dt.timedelta(days=GAP) < a or s > b + dt.timedelta(days=GAP) for a, b in taken)


def place(span):
    for _ in range(6000):
        y, mo = months[rng.choice(len(months), p=w)]; s = dt.date(y, mo, int(rng.integers(1, 29))); e = s + dt.timedelta(days=span - 1)
        if e <= END and free(s, e): taken.append((s, e)); return s, e
    d = START                                                    # dense fallback: first free slot anywhere
    while d + dt.timedelta(days=span - 1) <= END:
        e = d + dt.timedelta(days=span - 1)
        if free(d, e): taken.append((d, e)); return d, e
        d += dt.timedelta(days=1)
    sys.exit("FATAL: no room left on the timeline")


episodes = []
for t in ("trip", "event", "doc", "ordinary"):
    spans = [int(rng.integers(3, 8)) if t == "trip" else 1 for _ in groups[t]]
    wins = sorted(place(sp) for sp in spans)                                       # windows by start date
    for gr, (s, e) in zip(sorted(groups[t], key=era), wins):                       # older era -> earlier window
        span = (e - s).days + 1
        days = [0] * len(gr) if span == 1 else sorted(int(x) for x in rng.choice(span, len(gr), p=rng.dirichlet(np.ones(span) * 1.5)))
        episodes.append(dict(type=t, start=s, end=e, idx=gr, days=days))
episodes.sort(key=lambda x: (x["start"], x["end"]))

m["date_source"] = m.date_source.where(real, "synthetic")
out = []
for k, ep in enumerate(episodes, 1):
    eid = f"ep{k:03d}"
    for i, d in zip(ep["idx"], ep["days"]):
        m.at[i, "episode_id"] = eid; m.at[i, "assigned_date"] = (ep["start"] + dt.timedelta(days=d)).isoformat()
    out.append(dict(episode_id=eid, type=ep["type"], start=ep["start"].isoformat(), end=ep["end"].isoformat(), n_photos=len(ep["idx"]), photo_ids=[m.photo_id[i] for i in ep["idx"]]))
for d, gdf in m[real].groupby("assigned_date"):                                    # one episode per real day
    eid = "r_" + d.replace("-", ""); m.loc[gdf.index, "episode_id"] = eid
    out.append(dict(episode_id=eid, type="real", start=d, end=d, n_photos=len(gdf), photo_ids=list(gdf.photo_id)))
out.sort(key=lambda x: (x["start"], x["end"]))
assert (m.assigned_date != "").all() and (m.episode_id != "").all()
m.to_csv(ROOT / "data" / "manifest.csv", index=False)
(ROOT / "data" / "episodes.json").write_text(json.dumps(dict(
    note="Simulated timeline for photos without a real date; episodes are visually coherent; real-dated photos keep their dates.", seed=SEED, episodes=out), indent=1))

# ---------------------------------------------------------------- self-checks
syn_eps = [e for e in out if e["type"] != "real"]
wins = sorted((dt.date.fromisoformat(e["start"]), dt.date.fromisoformat(e["end"]), e["type"]) for e in out)
bad = [(a, b) for a, b in zip(wins, wins[1:]) if (b[0] - a[1]).days <= GAP and not (a[2] == "real" and b[2] == "real")]   # real-vs-real days are not ours to move
print(f"episodes: {len(syn_eps)} simulated + {len(out) - len(syn_eps)} real-day; simulated windows closer than {GAP + 1} days to anything: {len(bad)}")
for t in ("trip", "event", "ordinary", "doc"):
    es = [e for e in out if e["type"] == t]; print(f"  {t:9s} n={len(es):3d} photos={sum(e['n_photos'] for e in es)}")
ids = {p: i for i, p in enumerate(m.photo_id)}
def coh(es): return float(np.mean([np.mean(E[[ids[p] for p in e['photo_ids']]] @ (lambda c: c / np.linalg.norm(c))(E[[ids[p] for p in e['photo_ids']]].mean(0))) for e in es if e["n_photos"] >= 5]))
print(f"mean photo-to-episode-centre similarity: trips {coh([e for e in out if e['type']=='trip']):.3f}, events {coh([e for e in out if e['type']=='event']):.3f}")
old = ROOT / "data" / "backup_before_user_photos"
if (old / "episodes.json").exists():
    oe = json.loads((old / "episodes.json").read_text())["episodes"]; Eo = np.load(old / "embeddings.npy"); oid = {p: i for i, p in enumerate(pd.read_csv(old / "manifest.csv").photo_id)}
    c = lambda es: float(np.mean([np.mean(Eo[[oid[p] for p in e['photo_ids']]] @ (lambda x: x / np.linalg.norm(x))(Eo[[oid[p] for p in e['photo_ids']]].mean(0))) for e in es if e["n_photos"] >= 5]))
    print(f"  (before: trips {c([e for e in oe if e['type']=='trip']):.3f}, events {c([e for e in oe if e['type']=='event']):.3f})")
if bad: sys.exit("FATAL: episode windows too close")
