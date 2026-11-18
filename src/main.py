"""
Command line runner for the Music Recommender.

    python -m src.main                       interactive: type requests, get answers (RAG)
    python -m src.main --query "sad piano"   answer one request and exit
    python -m src.main --demo                the original fixed-profile demo (no API needed)

Add --debug to see the parsed query and retrieval scores, --verbose for INFO logs on the
console (they always go to logs/app.log).
"""

import argparse
import logging
import sys
from typing import Callable, Dict, List, Optional

from .config import get_api_key, get_model
from .logging_setup import setup_logging
from .pipeline import DEFAULT_SONGS_PATH, RecommendationPipeline
from .recommender import load_songs, recommend_songs

QUIT_WORDS = {"quit", "exit", "q"}
SOURCE_LABELS = {
    "llm": "answered by the AI model",
    "llm_retry": "answered by the AI model after one retry",
    "fallback": "scorer fallback: no AI explanation",
    "no_match": "no match",
    "error": "error",
}
WELCOME = (
    "Music recommender. Describe what you'd like to hear, "
    "e.g. 'mellow music for studying on a rainy night'.\n"
    "Press Enter on a blank line, or type 'quit', to exit."
)
CLIENT_FAILED_NOTICE = (
    "Note: an API key is set, but the AI client could not start (see the warning above). "
    "Usually this means you are running a different Python than the project's. Run the app with "
    "'.venv/bin/python -m src.main' (or activate .venv first), then answers will get AI explanations."
)
NO_KEY_NOTICE = (
    "Note: no API key was found (set GEMINI_API_KEY or ANTHROPIC_API_KEY, see .env.example), so "
    "answers come from the scoring engine only, without AI explanations."
)


PROFILES = {
    "High-Energy Pop": {
        "genre": "pop", "mood": "happy",
        "energy": 0.90, "valence": 0.80, "acousticness": 0.10, "danceability": 0.85,
    },
    "Chill Lofi": {
        "genre": "lofi", "mood": "chill",
        "energy": 0.38, "valence": 0.58, "acousticness": 0.80, "danceability": 0.60,
    },
    "Deep Intense Rock": {
        "genre": "rock", "mood": "intense",
        "energy": 0.92, "valence": 0.40, "acousticness": 0.08, "danceability": 0.60,
    },
    # Adversarial / edge-case profiles (System Evaluation)
    "Conflicted: Sad but Energetic": {
        "genre": "classical", "mood": "sad",
        "energy": 0.90, "valence": 0.20, "acousticness": 0.95, "danceability": 0.30,
    },
    "Unknown Genre and Mood": {
        "genre": "k-pop", "mood": "bubbly",
        "energy": 0.70, "valence": 0.80, "acousticness": 0.20, "danceability": 0.80,
    },
    "Dead Center (all 0.5)": {
        "genre": "pop", "mood": "happy",
        "energy": 0.50, "valence": 0.50, "acousticness": 0.50, "danceability": 0.50,
    },
    "Out of Range (energy 2.0)": {
        "genre": "pop", "mood": "happy",
        "energy": 2.0, "valence": 0.80, "acousticness": 0.10, "danceability": 0.85,
    },
}


def show_recommendations(name: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    """Print the top k recommendations for one named profile."""
    recommendations = recommend_songs(user_prefs, songs, k=k)

    print(f"\nProfile: {name}")
    print("=" * 60)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} by {song['artist']}")
        print(f"   Score: {score:.2f}")
        print("   Reasons:")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
        print("-" * 60)

# ---------------------------------------------------------------- RAG mode

def format_result(result: Dict, debug: bool = False) -> str:
    """Render a pipeline result as text."""
    lines = [result["summary"], ""] if result["summary"] else []
    for rank, rec in enumerate(result["recommendations"], start=1):
        lines.append(f"{rank}. {rec['title']} by {rec['artist']}")
        lines.append(f"   {rec['reason']}")
    if result["recommendations"]:
        lines.append("")

    lines.append(f"[{SOURCE_LABELS.get(result['source'], result['source'])}]")
    if result["guardrail_events"]:
        lines.append("Guardrail events:")
        lines.extend(f"  - {event}" for event in result["guardrail_events"])

    if debug:
        lines.append("")
        lines.append(f"parsed query: {result['parsed_query']}")
        lines.append(f"latency: {result['latency_s']:.2f}s")
        if result["retrieved"]:
            lines.append("retrieved (combined = text + structured, x compatibility):")
            for r in result["retrieved"]:
                lines.append(
                    f"  #{r['song']['id']:<3} {r['song']['title']:<22} combined {r['combined']:.2f}"
                    f"  text {r['text_sim']:.2f}  structured {r['struct_norm']:.2f}"
                    f"  x{r['compat_factor']:.1f}"
                )
    return "\n".join(lines)


def run_query(pipeline: RecommendationPipeline, query: str, debug: bool = False,
              out: Callable[[str], None] = print) -> Dict:
    result = pipeline.answer_query(query)
    out(format_result(result, debug=debug))
    return result


def run_interactive(pipeline: RecommendationPipeline, debug: bool = False,
                    input_fn: Callable[[str], str] = input,
                    out: Callable[[str], None] = print) -> None:
    """Prompt for requests until a quit word, a blank line, Ctrl-D or Ctrl-C."""
    out(WELCOME)
    while True:
        try:
            query = input_fn("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            break
        if not query or query.lower() in QUIT_WORDS:
            break
        out("")
        run_query(pipeline, query, debug=debug, out=out)
    out("Goodbye.")


# ---------------------------------------------------------------- demo mode

def run_demo(songs_path: str = str(DEFAULT_SONGS_PATH)) -> None:
    """The original fixed-profile demo. Uses the scorer only, so no API key is needed."""
    songs = load_songs(songs_path)
    print(f"Loaded songs: {len(songs)}")

    for name, user_prefs in PROFILES.items():
        show_recommendations(name, user_prefs, songs)


# ---------------------------------------------------------------- entry point

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.main", description="Music recommender")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-q", "--query", help="answer this request once and exit")
    mode.add_argument("--demo", action="store_true", help="run the original fixed-profile demo")
    parser.add_argument("--debug", action="store_true", help="show the parsed query and retrieval scores")
    parser.add_argument("--verbose", action="store_true", help="show INFO logs on the console")
    return parser


def main(
    argv: Optional[List[str]] = None,
    pipeline: Optional[RecommendationPipeline] = None,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    err: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
    configure_logging: Callable[..., object] = setup_logging,
) -> int:
    """Returns the process exit code. Everything it touches is injectable for tests."""
    args = build_arg_parser().parse_args(argv)

    configure_logging(console_level=logging.INFO if args.verbose else logging.WARNING)

    if args.demo:
        run_demo()
        return 0

    if pipeline is None:
        try:
            pipeline = RecommendationPipeline()
        except Exception as exc:
            err(f"Error: could not start the recommender ({type(exc).__name__}: {exc})")
            return 1

    if pipeline.client is None:
        out(CLIENT_FAILED_NOTICE if get_api_key() else NO_KEY_NOTICE)
    else:
        out(f"AI explanations from {get_model()}.")

    if args.query is not None:
        result = run_query(pipeline, args.query, debug=args.debug, out=out)
        return 1 if result["source"] == "error" else 0

    run_interactive(pipeline, debug=args.debug, input_fn=input_fn, out=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
