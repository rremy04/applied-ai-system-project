"""
Evaluation harness for the RAG music recommender.

    python -m tests.eval_rag              offline: keyword parser, no API, fully reproducible
    python -m tests.eval_rag --online     real LLM parsing + generation (needs a key in .env)
    python -m tests.eval_rag --online --runs 5 --out-dir some/dir

Within one mode, every retrieval variant (original scorer, text-only, structured-only,
hybrid, weight sweep) is scored on the SAME parsed query, so differences come from
retrieval, not from parsing. Online mode additionally runs the full pipeline repeatedly
to measure guardrail outcomes and consistency of real LLM output.

GROUND TRUTH was written before any retrieval results were examined. It lists acceptable
catalog attributes (never song IDs):
  acceptable  a returned song counts if its genre OR mood is listed
  required    optional extra condition the song must ALSO meet (e.g. genre for "sad piano",
              so a merely sad-sounding song from another genre earns no credit)
  supporting  numeric ranges reported as a sanity check on hits; they do not decide hits

With ~14 queries these numbers are indicative, not statistically representative.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

from src.config import ALL_KEY_VARS, get_model
from src.llm_utils import make_client
from src.pipeline import DEFAULT_SONGS_PATH, RecommendationPipeline
from src.query_parser import parse_query
from src.recommender import load_songs, recommend_songs
from src.retriever import Retriever

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "eval_results"
K = 3

VARIANT_LABELS = {
    "original_scorer": "Original scorer (no text, no synonyms/compatibility)",
    "w=0.00": "Structured only (scorer via retrieval adapter)",
    "w=0.25": "Hybrid, text weight 0.25",
    "w=0.50": "Hybrid, text weight 0.50 (default)",
    "w=0.75": "Hybrid, text weight 0.75",
    "w=1.00": "Text only (TF-IDF)",
}

# ---------------------------------------------------------------------------
# Ground truth (frozen before results were examined)
# ---------------------------------------------------------------------------

QUERIES: List[Dict] = [
    dict(id="q01", query="mellow music for studying on a rainy night", tests="scene words",
         acceptable={"genre": ["lofi"], "mood": ["chill", "focused"]},
         supporting={"energy": (0.0, 0.5)}),
    dict(id="q02", query="sad piano", tests="synonym map + instrument proxy",
         acceptable={"genre": ["classical"], "mood": ["melancholic"]},
         required={"genre": ["classical"]},
         supporting={"energy": (0.0, 0.4), "tempo_bpm": (0, 90)}),
    dict(id="q03", query="dark synthwave for a night drive", tests="explicit genre + scene",
         acceptable={"genre": ["synthwave"], "mood": ["moody"]},
         required={"genre": ["synthwave"]}),
    dict(id="q04", query="romantic date night, slow and smooth", tests="mood word + tempo",
         acceptable={"genre": ["r&b"], "mood": ["romantic"]},
         supporting={"tempo_bpm": (0, 100)}),
    dict(id="q05", query="aggressive heavy music", tests="synonym map",
         acceptable={"genre": ["metal"], "mood": ["angry"]},
         supporting={"energy": (0.85, 1.0)}),
    dict(id="q06", query="high energy workout music for the gym", tests="activity + energy",
         acceptable={"mood": ["intense", "euphoric", "angry"]},
         supporting={"energy": (0.85, 1.0)}),
    dict(id="q07", query="something to pump me up before a race", tests="paraphrase: energy inference",
         acceptable={"mood": ["intense", "euphoric"]},
         supporting={"energy": (0.8, 1.0)}),
    dict(id="q08", query="music for when I'm heartbroken", tests="paraphrase: emotional inference",
         acceptable={"mood": ["melancholic"]},
         supporting={"energy": (0.0, 0.5), "valence": (0.0, 0.5)}),
    dict(id="q09",
         query="something unobtrusive to keep me company while I work through a long programming session",
         tests="paraphrase: context inference",
         acceptable={"genre": ["lofi"], "mood": ["focused", "chill"]},
         supporting={"energy": (0.0, 0.5)}),
    dict(id="q10", query="songs that make a long drive through the city feel like a movie",
         tests="paraphrase: cinematic scene inference",
         acceptable={"genre": ["synthwave"], "mood": ["moody"]}),
    dict(id="q11", query="something cheerful for a beach day", tests="mixed: mood + scene",
         acceptable={"genre": ["reggae"], "mood": ["happy", "uplifting"]},
         supporting={"valence": (0.7, 1.0)}),
    dict(id="q12", query="music to fall asleep to", tests="paraphrase: no mood word",
         acceptable={"genre": ["ambient", "folk"], "mood": ["dreamy", "chill", "relaxed"]},
         supporting={"energy": (0.0, 0.4)}),
    dict(id="n01", query="asdfgh qwerty", tests="negative: gibberish", expect="no_match"),
    dict(id="n02", query="zxqv blorf 9182", tests="negative: gibberish", expect="no_match"),
]
for _q in QUERIES:
    _q.setdefault("expect", "hit")


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def is_hit(song: Dict, truth: Dict) -> bool:
    """acceptable (genre OR mood) AND every `required` field."""
    acceptable = truth.get("acceptable", {})
    ok = song["genre"] in acceptable.get("genre", []) or song["mood"] in acceptable.get("mood", [])
    if ok:
        for field, values in truth.get("required", {}).items():
            ok = ok and song[field] in values
    return ok


def supporting_ok(song: Dict, truth: Dict) -> Optional[bool]:
    ranges = truth.get("supporting")
    if not ranges:
        return None
    return all(lo <= song[feature] <= hi for feature, (lo, hi) in ranges.items())


def first_hit_rank(songs: List[Dict], truth: Dict, scores: Optional[List[float]] = None) -> Optional[int]:
    """1-based rank of the first hit. With `scores`, a result scoring <= 0 is title-order
    padding, not a retrieval, so it earns no credit (the pipeline answers no_match there)."""
    for rank, song in enumerate(songs, start=1):
        if scores is not None and scores[rank - 1] <= 0:
            continue
        if is_hit(song, truth):
            return rank
    return None


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def _top_rows(songs: List[Dict], scores: List[float], truth: Dict) -> List[Dict]:
    return [{
        "id": s["id"], "title": s["title"], "genre": s["genre"], "mood": s["mood"],
        "score": round(score, 3), "has_signal": score > 0,
        "hit": score > 0 and is_hit(s, truth), "supporting_ok": supporting_ok(s, truth),
    } for s, score in zip(songs, scores)]


# ---------------------------------------------------------------------------
# Retrieval variants (same parsed query for all of them)
# ---------------------------------------------------------------------------

def original_scorer_prefs(parsed: Dict, genres: List[str], moods: List[str]) -> Dict:
    """What the original scorer could be given: exact catalog genre/mood plus numeric targets."""
    prefs: Dict = {}
    for key, requested, valid in (("genre", parsed["genres"], genres), ("mood", parsed["moods"], moods)):
        for value in requested:
            if value in valid:
                prefs[key] = value
                break
    for feature in ("energy", "valence", "acousticness", "danceability"):
        if feature in parsed["numeric_features"]:
            prefs[feature] = parsed["numeric_features"][feature]
    return prefs


def evaluate_retrieval(retriever: Retriever, songs: List[Dict], parsed: Dict, truth: Dict) -> Dict:
    variants = {}

    ranked = recommend_songs(original_scorer_prefs(parsed, retriever.genres, retriever.moods), songs, k=K)
    top = [song for song, _, _ in ranked]
    scores = [score for _, score, _ in ranked]
    variants["original_scorer"] = {
        "top": _top_rows(top, scores, truth),
        "first_hit_rank": first_hit_rank(top, truth, scores),
        "no_signal": False,  # the original scorer never refuses: it always returns songs
    }

    for weight in WEIGHTS:
        results = retriever.retrieve(parsed, k=K, text_weight=weight)
        top = [r["song"] for r in results]
        scores = [r["combined"] for r in results]
        variants[f"w={weight:.2f}"] = {
            "top": _top_rows(top, scores, truth),
            "first_hit_rank": first_hit_rank(top, truth, scores),
            "no_signal": not results or results[0]["combined"] <= 0,  # same rule the pipeline uses
        }
    return variants


def summarize_variant(records: List[Dict], variant: str) -> Dict:
    positives = [r for r in records if r["expect"] == "hit"]
    negatives = [r for r in records if r["expect"] == "no_match"]
    ranks = [r["variants"][variant]["first_hit_rank"] for r in positives]
    top1_rows = [r["variants"][variant]["top"][0] for r in positives if r["variants"][variant]["top"]]
    supports = [row["supporting_ok"] for row in top1_rows if row["hit"] and row["supporting_ok"] is not None]
    return {
        "top1": sum(1 for rank in ranks if rank == 1) / len(positives),
        "top3": sum(1 for rank in ranks if rank is not None) / len(positives),
        "negative_accuracy": (
            sum(1 for r in negatives if r["variants"][variant]["no_signal"]) / len(negatives)
            if negatives else None
        ),
        "top1_hits_meeting_supporting": (sum(supports) / len(supports)) if supports else None,
    }


# ---------------------------------------------------------------------------
# Online: pipeline runs, guardrails, consistency, usage
# ---------------------------------------------------------------------------

class CountingClient:
    """Wraps a real client to count calls and tokens, without changing behavior."""

    def __init__(self, client):
        self._client = client
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        response = self._client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return response


QUOTA_EVENT = "rate or daily quota limit"
PARSE_FALLBACK_EVENT = "parse fell back to keyword parser"


def _is_quota_run(result: Dict) -> bool:
    return any(QUOTA_EVENT in e for e in result["guardrail_events"])


def run_pipeline_stage(pipeline: RecommendationPipeline, songs_by_id: Dict, runs: int,
                       delay: float = 0.0, sleep: Optional[Callable[[float], None]] = None) -> Dict:
    """
    Run every query `runs` times through the real pipeline.

    Rate-limited runs (provider quota) are NOT guardrail outcomes: they are counted
    separately and left out of every rate, so a free-tier limit cannot masquerade as a
    hallucination or fallback problem. `delay` spaces the calls (seconds) for per-minute limits.
    """
    per_query, calls_made = [], 0
    for truth in QUERIES:
        results = []
        for _ in range(runs):
            if calls_made and delay > 0:
                (sleep or time.sleep)(delay)  # resolved at call time so tests can patch it
            results.append(pipeline.answer_query(truth["query"]))
            calls_made += 1
        per_query.append({"truth": truth, "results": results})

    all_runs = [res for pq in per_query for res in pq["results"]]
    quota_runs = [res for res in all_runs if _is_quota_run(res)]
    clean = [res for res in all_runs if not _is_quota_run(res)]
    answered = [res for res in clean if res["source"] in ("llm", "llm_retry", "fallback")]
    count = lambda source: sum(1 for res in answered if res["source"] == source)  # noqa: E731
    negatives = [pq for pq in per_query if pq["truth"]["expect"] == "no_match"]
    positives = [pq for pq in per_query if pq["truth"]["expect"] == "hit"]

    def clean_results(pq):
        return [res for res in pq["results"] if not _is_quota_run(res)]

    def end_to_end(pq):
        runs_ok = clean_results(pq)
        if not runs_ok:
            return "skipped"
        songs = [songs_by_id[r["song_id"]] for r in runs_ok[0]["recommendations"]]
        return first_hit_rank(songs, pq["truth"])

    ranks = [end_to_end(pq) for pq in positives]
    ranks_scored = [r for r in ranks if r != "skipped"]
    consistency, overlaps = [], []
    for pq in positives:
        ok = clean_results(pq)
        if len(ok) < 2:
            continue  # consistency needs at least two clean runs
        id_lists = [[r["song_id"] for r in res["recommendations"]] for res in ok]
        firsts = [ids[0] if ids else None for ids in id_lists]
        consistency.append(len(set(firsts)) == 1 and firsts[0] is not None)
        pairs = list(combinations(id_lists, 2))
        overlaps.append(statistics.mean(jaccard(a, b) for a, b in pairs))

    return {
        "runs_per_query": runs,
        "total_runs": len(all_runs),
        "rate_limited_runs": len(quota_runs),
        "clean_runs": len(clean),
        "answered_runs": len(answered),
        "parse_fallback_runs": sum(
            1 for res in all_runs if any(PARSE_FALLBACK_EVENT in e for e in res["guardrail_events"])
        ),
        "guardrail_pass_rate": count("llm") / len(answered) if answered else None,
        "retry_rate": count("llm_retry") / len(answered) if answered else None,
        "fallback_rate": count("fallback") / len(answered) if answered else None,
        "dropped_id_events": sum(
            1 for res in clean for event in res["guardrail_events"] if "dropped invalid" in event
        ),
        "negative_accuracy": (
            sum(1 for pq in negatives if all(r["source"] == "no_match" for r in clean_results(pq)))
            / len(negatives) if negatives else None
        ),
        "end_to_end_top1": (sum(1 for rank in ranks_scored if rank == 1) / len(ranks_scored)) if ranks_scored else None,
        "end_to_end_top3": (sum(1 for rank in ranks_scored if rank is not None) / len(ranks_scored)) if ranks_scored else None,
        "top1_consistency": (sum(consistency) / len(consistency)) if consistency else None,
        "mean_recommendation_overlap": statistics.mean(overlaps) if overlaps else None,
        "mean_latency_s": statistics.mean(res["latency_s"] for res in all_runs) if all_runs else None,
        "per_query": [{
            "id": pq["truth"]["id"], "query": pq["truth"]["query"],
            "sources": [res["source"] for res in pq["results"]],
            "recommended_ids": [[r["song_id"] for r in res["recommendations"]] for res in pq["results"]],
            "events": [res["guardrail_events"] for res in pq["results"]],
        } for pq in per_query],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(online: bool = False, client=None, runs: int = 3,
                   songs: Optional[List[Dict]] = None, delay: float = 0.0) -> Dict:
    """Run the whole evaluation and return the report dict. Never writes files."""
    saved_keys: Dict[str, str] = {}
    if not online:  # offline must be API-free even if a key is configured
        saved_keys = {n: os.environ.pop(n) for n in ALL_KEY_VARS if n in os.environ}
    try:
        songs = songs if songs is not None else load_songs(str(DEFAULT_SONGS_PATH))
        retriever = Retriever(songs)
        vocab = {"genres": retriever.genres, "moods": retriever.moods}
        counting = CountingClient(client or make_client()) if online else None

        records = []
        for truth in QUERIES:
            parsed = parse_query(truth["query"], vocab, client=counting)  # one parse, shared by variants
            records.append({
                "id": truth["id"], "query": truth["query"], "tests": truth["tests"],
                "expect": truth["expect"], "parsed": parsed,
                "ground_truth": {k: truth[k] for k in ("acceptable", "required", "supporting") if k in truth},
                "variants": evaluate_retrieval(retriever, songs, parsed, truth),
            })

        pipeline = RecommendationPipeline(songs=songs, client=counting)
        pipeline_stage = run_pipeline_stage(pipeline, {s["id"]: s for s in songs},
                                            runs if online else 1, delay=delay if online else 0.0)

        report = {
            "mode": "online" if online else "offline",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "catalog_size": len(songs),
            "n_queries": len(QUERIES),
            "n_positive": sum(1 for q in QUERIES if q["expect"] == "hit"),
            "n_negative": sum(1 for q in QUERIES if q["expect"] == "no_match"),
            "summary": {name: summarize_variant(records, name) for name in VARIANT_LABELS},
            "pipeline": pipeline_stage,
            "records": records,
        }
        if online:
            report["model"] = get_model()
            report["api_usage"] = {
                "calls": counting.calls,
                "input_tokens": counting.input_tokens,
                "output_tokens": counting.output_tokens,
            }
        return report
    finally:
        os.environ.update(saved_keys)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: Optional[float], fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def render_markdown(report: Dict) -> str:
    n_pos, n_neg = report["n_positive"], report["n_negative"]
    lines = [
        f"# RAG evaluation ({report['mode']} mode)",
        "",
        f"Generated {report['generated_at']} on a {report['catalog_size']}-song catalog: "
        f"{n_pos} relevance queries and {n_neg} gibberish queries.",
        "",
        f"> With only {report['n_queries']} queries these results are indicative, not statistically "
        "representative. Ground truth was written before results were examined.",
        "",
        "## Retrieval: top-1 / top-3 hit rate",
        "",
        "Every row is scored on the same parsed query per query. Hit = acceptable genre/mood "
        "(and any required field). Negative accuracy = gibberish correctly yields no signal; "
        "the original scorer never refuses, so it cannot score here.",
        "",
        "| Variant | Top-1 | Top-3 | Negative accuracy | Top-1 hits meeting supporting ranges |",
        "|---|---|---|---|---|",
    ]
    for name, label in VARIANT_LABELS.items():
        s = report["summary"][name]
        lines.append(f"| {label} | {_pct(s['top1'])} | {_pct(s['top3'])} | "
                     f"{_pct(s['negative_accuracy'])} | {_pct(s['top1_hits_meeting_supporting'])} |")

    pipeline = report["pipeline"]
    if report["mode"] == "online":
        lines += [
            "", "## Full pipeline (real LLM)", "",
            f"{pipeline['runs_per_query']} runs per query, {pipeline['total_runs']} in total. "
            f"Guardrail rates are over the {pipeline['answered_runs']} clean answered runs (not no-match, "
            "not rate-limited).", "",
        ] + ([
            f"> **INCOMPLETE:** {pipeline['rate_limited_runs']} run(s) were refused by the provider's "
            "quota or rate limit. They are excluded from every rate below, so these numbers describe "
            "only the runs that completed. Re-run after the limit resets, or with more quota.", "",
        ] if pipeline["rate_limited_runs"] else []) + [
            "| Metric | Value |", "|---|---|",
            f"| Rate-limited runs (excluded below) | {pipeline['rate_limited_runs']} |",
            f"| Parse fell back to keyword parser | {pipeline['parse_fallback_runs']} |",
            f"| Guardrail pass rate (model answer, no retry) | {_pct(pipeline['guardrail_pass_rate'])} |",
            f"| Retry rate | {_pct(pipeline['retry_rate'])} |",
            f"| Fallback rate | {_pct(pipeline['fallback_rate'])} |",
            f"| Dropped invalid-ID events | {pipeline['dropped_id_events']} |",
            f"| End-to-end top-1 / top-3 (first run) | {_pct(pipeline['end_to_end_top1'])} / "
            f"{_pct(pipeline['end_to_end_top3'])} |",
            f"| Negative accuracy (no_match) | {_pct(pipeline['negative_accuracy'])} |",
            f"| Top-1 consistency across runs | {_pct(pipeline['top1_consistency'])} |",
            f"| Mean recommendation overlap (Jaccard) | {_num(pipeline['mean_recommendation_overlap'], '.2f')} |",
            f"| Mean latency per query | {_num(pipeline['mean_latency_s'], '.2f')}s |",
            f"| API calls / tokens in / tokens out | {report['api_usage']['calls']} / "
            f"{report['api_usage']['input_tokens']} / {report['api_usage']['output_tokens']} |",
        ]
    else:
        lines += [
            "", "## Full pipeline (offline)", "",
            f"Gibberish handled as no_match by the real pipeline: {_pct(pipeline['negative_accuracy'])}. "
            "Guardrail rates and consistency need `--online`.",
        ]

    lines += ["", "## Per query (hybrid, text weight 0.50)", "",
              "| # | Query | Tests | First hit rank | Top 3 (✓ hit, ✗ miss, – no signal) |", "|---|---|---|---|---|"]
    for r in report["records"]:
        v = r["variants"]["w=0.50"]
        if r["expect"] == "no_match":
            outcome = "no_match ✓" if v["no_signal"] else "returned songs ✗"
            lines.append(f"| {r['id']} | {r['query']} | {r['tests']} | n/a | {outcome} |")
            continue
        rank = v["first_hit_rank"] or "none"
        top = "; ".join(
            f"{'✓' if t['hit'] else '✗' if t['has_signal'] else '–'} {t['title']}" for t in v["top"]
        )
        lines.append(f"| {r['id']} | {r['query']} | {r['tests']} | {rank} | {top} |")

    misses = [r for r in report["records"]
              if r["expect"] == "hit" and r["variants"]["w=0.50"]["first_hit_rank"] != 1]
    lines += ["", "## Hybrid misses (not top-1), with parsed query and scores", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        v = r["variants"]["w=0.50"]
        lines += [f"**{r['id']}: {r['query']}**  ", f"parsed: `{json.dumps(r['parsed'])}`  ",
                  f"accepts: `{json.dumps(r['ground_truth'].get('acceptable'))}`"
                  + (f", requires `{json.dumps(r['ground_truth']['required'])}`"
                     if "required" in r["ground_truth"] else "") + "  ",
                  "returned: " + "; ".join(
                      f"{t['title']} ({t['genre']}/{t['mood']}, {t['score']}, "
                      f"{'hit' if t['hit'] else 'miss' if t['has_signal'] else 'no signal'})"
                      for t in v["top"]), ""]
    return "\n".join(lines) + "\n"


def write_reports(report: Dict, out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"{report['mode']}.md", out_dir / f"{report['mode']}.json"
    md.write_text(render_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return [md, js]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.eval_rag", description=__doc__.split("\n")[1])
    parser.add_argument("--online", action="store_true", help="use the real LLM (needs GEMINI_API_KEY or ANTHROPIC_API_KEY)")
    parser.add_argument("--runs", type=int, default=3, help="online: repeated runs per query (default 3)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="online: seconds to wait between calls. Free tiers often allow 5 requests "
                             "per minute, so 13 is a safe value")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    client = None
    if args.online:
        client = make_client()
        if client is None:
            print("Error: --online needs GEMINI_API_KEY or ANTHROPIC_API_KEY (see .env.example).", file=sys.stderr)
            return 1

    started = time.perf_counter()
    report = run_evaluation(online=args.online, client=client, runs=max(1, args.runs),
                            delay=max(0.0, args.delay))
    paths = write_reports(report, Path(args.out_dir))

    print(render_markdown(report))
    print(f"Finished in {time.perf_counter() - started:.1f}s. Wrote: " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
