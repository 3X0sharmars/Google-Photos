"""Moment retrieval over CLIP image embeddings (pixels only) plus a metadata-hit variant.
A "moment" = photos from the same stretch of (assigned) dates; 4 visually distinct thumbnails represent it."""
import datetime as dt

import numpy as np

TOP_K, GAP_DAYS, MIN_CLUSTER, N_THUMBS, FALLBACK_N, BROWSE_PAD = 100, 3, 3, 4, 20, 2
ABS_FLOOR = 0.225             # a photo below this similarity is never a match
REL_MARGIN = 0.06             # ...nor is one more than this far below the best match (stops the weak tail leaking in)


def _day(s): return dt.date.fromisoformat(s)


def temporal_clusters(idx, dates):
    order = sorted(idx, key=lambda i: dates[i])
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if (_day(dates[i]) - _day(dates[cur[-1]])).days <= GAP_DAYS: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    return groups


def _kmeans(X, k, iters=25, seed=0):
    """Tiny k-means (k-means++ start) so the app does not need scikit-learn. Returns (labels, centres)."""
    rng = np.random.default_rng(seed); n = len(X)
    C = [X[int(rng.integers(n))]]
    for _ in range(1, k):
        d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[int(rng.choice(n, p=d / d.sum()))] if d.sum() > 0 else X[int(rng.integers(n))])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(axis=0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C): break
        C = newC
    return lab, C


def diverse_thumbs(members, E, dates):
    """Diversity rule: k-means (k=4) over member embeddings, nearest member to each centroid (never the 4 top scorers)."""
    members = list(members)
    if len(members) <= N_THUMBS: return sorted(members, key=lambda i: dates[i])
    X = E[members]
    labels, centres = _kmeans(X, N_THUMBS)
    picks = []
    for c in range(N_THUMBS):
        local = np.where(labels == c)[0]
        if len(local) == 0: continue
        d = np.linalg.norm(X[local] - centres[c], axis=1)
        picks.append(members[int(local[np.argmin(d)])])
    return sorted(set(picks), key=lambda i: dates[i])


def _moment(members, score, E, dates):
    ds = sorted(dates[i] for i in members)
    return dict(start=ds[0], end=ds[-1], count=len(members), score=round(float(score), 3), thumbs=diverse_thumbs(members, E, dates),
                members=sorted(members, key=lambda i: dates[i]))


def visual_moments(sims, E, dates, n_moments=5, neg=None):
    """CLIP similarity -> keep only photos that genuinely match -> temporal clusters -> score = count x mean sim.
    A photo is dropped if it is below ABS_FLOOR, more than REL_MARGIN under the best match, or (when `neg` is given)
    looks at least as much like the contradiction query as like the target. Returns (moments, flat_fallback_idx, best_sim)."""
    top = np.argsort(-sims)[:TOP_K]
    best = float(sims[top[0]]); floor = max(ABS_FLOOR, best - REL_MARGIN)
    cand = [int(i) for i in top if sims[i] >= floor and (neg is None or neg[i] < sims[i])]
    if len(cand) < MIN_CLUSTER: return [], cand[:FALLBACK_N], best
    cl = [dict(members=g, score=len(g) * float(np.mean(sims[g]))) for g in temporal_clusters(cand, dates)]
    cl.sort(key=lambda c: -c["score"])
    good = [c for c in cl if len(c["members"]) >= MIN_CLUSTER]
    if not good: return [], cand[:FALLBACK_N], best
    return [_moment(c["members"], c["score"], E, dates) for c in good[:n_moments]], [], best


def metadata_moments(hits, E, dates, n_moments=5):
    """hits = [(photo_index, weight)] from the metadata index. Same moment grouping; score = summed match weight."""
    if not hits: return [], []
    w = {i: wt for i, wt in hits}
    cl = [dict(members=g, score=sum(w[i] for i in g)) for g in temporal_clusters(list(w), dates)]
    cl.sort(key=lambda c: -c["score"])
    return [_moment(c["members"], c["score"], E, dates) for c in cl[:n_moments]], [i for i, _ in hits[:FALLBACK_N]]


def browse(start, end, dates):
    lo = _day(start) - dt.timedelta(days=BROWSE_PAD); hi = _day(end) + dt.timedelta(days=BROWSE_PAD)
    idx = [i for i, d in enumerate(dates) if lo <= _day(d) <= hi]
    return sorted(idx, key=lambda i: (dates[i], i)), lo.isoformat(), hi.isoformat()
