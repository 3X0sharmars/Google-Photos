"""Step 1 of search: match the query against photo metadata (Flickr title + tags)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "the", "of", "in", "at", "to", "my", "me", "photo", "photos", "picture", "pictures", "pic", "pics", "from", "with", "and", "for"}


def stem(w):
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz": return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"): return w[:-1]
    return w


def tokens(text):
    return [stem(t) for t in _TOK.findall(str(text).lower()) if t not in _STOP]


class MetadataIndex:
    def __init__(self, ids):
        meta = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
        self.title_tok, self.tag_tok = [], []
        for pid in ids:
            m = meta.get(pid, {"title": "", "tags": []})
            self.title_tok.append(set(tokens(m["title"])))
            ts = set()
            for t in m["tags"]: ts.update(tokens(t))
            self.tag_tok.append(ts)

    def search(self, query):
        """All query words must appear (as words) in the title or tags. Returns [(photo_index, weight)] best first.
        Weight: 2 per word found in tags + 1 per word found in the title."""
        q = [t for t in tokens(query)]
        if not q: return []
        hits = []
        for i, (ti, ta) in enumerate(zip(self.title_tok, self.tag_tok)):
            if all((w in ti or w in ta) for w in q):
                hits.append((i, sum((2 if w in ta else 0) + (1 if w in ti else 0) for w in q)))
        hits.sort(key=lambda x: -x[1])
        return hits
