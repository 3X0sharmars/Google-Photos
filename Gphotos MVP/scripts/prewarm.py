"""Pre-fill data/suggestion_cache.json for common queries so a demo does not depend on live Gemini quota.
Skips queries already cached. Run from moments/:  python scripts/prewarm.py [query ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import gemini_expand as g  # noqa: E402

DEFAULT = ["mumbai", "delhi", "jaipur", "chennai", "hyderabad", "himalayas", "paris", "london", "new york", "singapore", "dubai", "bali"]
for q in (sys.argv[1:] or DEFAULT):
    r = g.expand(q)
    if r is None: print(f"  {q:14s} FAILED: {g.last_error()}"); continue
    print(f"  {q:14s} {'cached' if r['cached'] else 'new   '} [{r['model']}] " + ", ".join(c["label"] for c in r["categories"]) or "(no categories)")
