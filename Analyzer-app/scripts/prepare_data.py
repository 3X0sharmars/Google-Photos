"""Build analyzer-app/data/{dataset.json,stats.json} from ../analyzer_ready/ (run: python scripts/prepare_data.py).
dataset.json = the retained records (author names are not included anywhere). stats.json = headline numbers for section A."""
import csv, json, re, collections
from pathlib import Path
from urllib.parse import urlparse, parse_qs
csv.field_size_limit(2**31 - 1)
HERE = Path(__file__).resolve().parent.parent
SRC = HERE.parent / "analyzer_ready"

def platform_group(p):
    s = (p or "").strip(); l = s.lower()
    if l.startswith("reddit") or l.startswith("r/"): return "Reddit"
    if l == "youtube": return "YouTube"
    if l in ("playstore", "google play"): return "Google Play"
    if l.startswith("apple app store") or l == "appstore": return "Apple App Store"
    if l in ("help_community", "google help community", "google_pixel_community"): return "Google Help Community"
    if l in ("survey", "interview", "user_testing"): return {"survey": "Primary survey", "interview": "Interviews", "user_testing": "User testing"}[l]
    return s or "Unknown"

def source_key(url, platform):
    if not url: return f"nourl:{platform_group(platform)}"
    u = urlparse(url); host = u.netloc.lower().replace("www.", "")
    if "youtube.com" in host:
        v = parse_qs(u.query).get("v", [None])[0]
        if v: return f"yt:{v}"
    m = re.search(r"/comments/([a-z0-9]+)", u.path)
    if "reddit.com" in host and m: return f"reddit:{m.group(1)}"
    return f"{host}{u.path.rstrip('/')}"

kept = [json.loads(l) for l in open(SRC / "analyzer_input.jsonl", encoding="utf-8")]
removed = list(csv.DictReader(open(SRC / "removed_records_audit.csv", encoding="utf-8-sig", newline="")))
dups = list(csv.DictReader(open(SRC / "duplicates_removed_audit.csv", encoding="utf-8-sig", newline="")))
counts = json.load(open(SRC / "_counts.json"))

# platform of removed duplicates is not in the audit file; look it up from the kept record they collapsed into
kept_by_srcid = {r["source_record_id"]: r for r in kept}
all_sources = set(); all_platforms = set()
for r in kept:
    all_sources.add(source_key(r["url"], r["platform"])); all_platforms.add(platform_group(r["platform"]))
for r in removed:
    all_sources.add(source_key(r["url"], r["platform"])); all_platforms.add(platform_group(r["platform"]))
for d in dups:
    k = kept_by_srcid.get(int(d["kept_source_record_id"]))
    all_sources.add(source_key(d["url"], k["platform"] if k else "")); 
    if k: all_platforms.add(platform_group(k["platform"]))
kept_sources = {source_key(r["url"], r["platform"]) for r in kept}
total = counts["candidates"]
assert total == len(kept) + len(removed) + len(dups), "counts do not reconcile"

dataset = [dict(id=r["analysis_id"], text=r["text"], platform=r["platform"], platform_group=platform_group(r["platform"]),
                source_type=r["source_type"], date=r["date"], year=r["year"], url=r["url"], thread_title=r["thread_title"]) for r in kept]
(HERE / "data" / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
why = collections.Counter(("date_out_of_range" if x["reason"].startswith("date_out") else x["reason"]) for x in removed)
stats = dict(
    scraped_total=total, unique_sources=len(all_sources), unique_platforms=len(all_platforms),
    retained=len(kept), retained_pct=round(100 * len(kept) / total, 1),
    retained_unique_sources=len(kept_sources), retained_platforms=len({platform_group(r["platform"]) for r in kept}),
    removed=dict(exact_duplicates=len(dups), content_free=why["content_free_short"], outside_2021_2026=why["date_out_of_range"],
                 empty=why["empty_text"], off_topic_thread=why["off_topic_thread"]),
    undated_retained=sum(1 for r in kept if r["date_status"] == "undated"),
    date_window="2021-2026",
    by_platform=dict(collections.Counter(platform_group(r["platform"]) for r in kept).most_common()),
    by_year=dict(sorted(collections.Counter(str(r["year"]) if r["year"] else "undated" for r in kept).items())),
)
(HERE / "data" / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in stats.items() if k not in ("by_platform", "by_year")}, indent=1))
print("platform groups:", stats["by_platform"])
