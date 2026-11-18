# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders
å
This version keeps the original content-based recommender (genre, mood and numeric features scored with fixed weights) and adds a retrieval-augmented layer on top. You describe what you want in plain English, the system retrieves matching songs from the catalog, and an LLM (Google Gemini by default) explains the picks using only those songs. Guardrails, logging and an evaluation harness are included, and everything still works without an API key by falling back to the scoring engine. See [RAG Extension](#rag-extension-ask-in-plain-english).

---

## Original Project

This project extends my Module 3 project, **Music Recommender Simulation** (the starter that asked for a content-based recommender). Its goal was to show how apps like Spotify predict what a listener will like, by scoring songs against a taste profile with a weighted rule set. It could score an 18-song catalog on genre, mood and numeric features, return the top five songs with reasons for a fixed profile, and surface bias such as filter bubbles.

---

## How The System Works

A content-based recommender. It scores every song in `data/songs.csv` against one user profile with fixed rules, then returns the top `k`.

- **Song features used:** genre, mood, energy, valence, acousticness, danceability.
- **UserProfile stores:** favorite genre, favorite mood, and a target for each numeric feature.

```
User prefs → load 18 songs → score each song → sort (ties by title) → top k + reasons
```

### Algorithm Recipe

| Component | Points | Rule |
|---|---|---|
| Mood match | +2.0 | exact match, else 0 |
| Genre match | +1.0 | exact match, else 0 |
| Energy | 0–2.0 | `2.0 * (1 - abs(song - target))` |
| Valence | 0–1.0 | `1.0 * (1 - abs(song - target))` |
| Acousticness | 0–1.0 | `1.0 * (1 - abs(song - target))` |
| Danceability | 0–0.5 | `0.5 * (1 - abs(song - target))` |

Max is 7.5. Sort high to low, break ties by title, return the top 5, and explain each pick with its biggest contributors.

Mood outweighs genre because 14 of 18 genres appear once, so a big genre bonus would just pick "the one song in your genre."

### Expected biases

- Genre (+1.0) can lift a mediocre song over a slightly better mood/energy match.
- Exact-match labels: "relaxed" earns nothing against "chill."
- One target per user, so mixed tastes are poorly served.
- Mood and energy are over half the points, so they can outvote the other features.
- Always picking the closest songs makes a filter bubble, and 18 songs means repeats.

---

## RAG Extension: Ask in Plain English

The weighted scorer above is still the engine, but you can now describe what you want ("mellow music for studying on a rainy night") instead of filling in a profile. A request goes through a retrieve-then-generate pipeline:

1. **Parse** (`src/query_parser.py`): the LLM turns the request into a structured query (genres, moods, scenes, related words, numeric targets). With no API key, or if the call fails, a keyword parser produces the same shape.
2. **Retrieve** (`src/retriever.py`): TF-IDF text similarity over each song's description, blended 50/50 with the weighted scorer after normalizing synonyms (for example "sad" becomes "melancholic"). A song whose mood clashes with the request has its score halved.
3. **Generate** (`src/generator.py`): the LLM writes the recommendation from the retrieved songs only and must cite them by ID.
4. **Guard** (`src/pipeline.py`): invalid song IDs are dropped and logged. If nothing valid remains, it retries once, then falls back to the scorer's own explanations. A request that matches nothing returns "no match" without calling the model. The pipeline never raises, and every model call has a timeout.

The full flow is in [diagrams/architecture.mmd](diagrams/architecture.mmd) (GitHub renders Mermaid files).

Every answer says where it came from: `llm`, `llm_retry`, `fallback`, `no_match` or `error`. This is real output from a run with no API key, so it shows the scorer fallback:

```
Note: no API key was found (set GEMINI_API_KEY or ANTHROPIC_API_KEY, see .env.example), so answers come from the scoring engine only, without AI explanations.
The AI explanation was unavailable, so these are the top matches from the scoring engine.

1. Moonlit Sonata by Clara Wren
   mood matches: melancholic (+2.0); energy 0.22 vs target 0.25 (+1.94); valence 0.25 vs target 0.20 (+0.95); text match (0.46): melancholic, piano, sad
2. Pine Smoke Lullaby by Wren & Willow
   energy 0.30 vs target 0.25 (+1.90); valence 0.57 vs target 0.20 (+0.63)
3. Spacewalk Thoughts by Orbit Bloom
   energy 0.28 vs target 0.25 (+1.94); valence 0.65 vs target 0.20 (+0.55)

[scorer fallback: no AI explanation]
Guardrail events:
  - no API client: using scorer fallback
```

### Evaluation results (offline)

`python -m tests.eval_rag` runs 14 queries (12 relevance, 2 gibberish) with ground truth written before looking at results. Offline means the keyword parser and no API. Full per-query output is in [eval_results/offline.md](eval_results/offline.md).

| Variant | Top-1 | Top-3 | Gibberish refused |
|---|---|---|---|
| Original scorer alone | 50% | 50% | 0% |
| Structured only (scorer through the retrieval adapter) | 83% | 83% | 100% |
| Hybrid, text weight 0.25 / 0.50 / 0.75 | 83% | 83% | 100% |
| Text only (TF-IDF) | 83% | 83% | 100% |

The gain over the original scorer comes from synonym normalization and compatibility scoring. Text search on top added nothing measurable on this 18-song catalog, and every text weight tied. With 12 relevance queries these numbers are indicative only. The full online evaluation (real LLM: guardrail rates and consistency across repeated runs) has **not been run yet**; so far only a handful of real calls have been checked. See section 9 of the [model card](model_card.md) for limitations and bias.

---

## Architecture Overview

The system is a retrieve-then-generate pipeline with a fallback. The flow in [diagrams/architecture.mmd](diagrams/architecture.mmd) runs left to right:

- **Input:** a request from the CLI (`python -m src.main`).
- **Parse:** the LLM turns the request into structured fields, or a keyword parser does if no key is set.
- **Retrieve:** the catalog is scored with TF-IDF text similarity and the weighted scorer, then blended.
- **Generate and guard:** the LLM writes an answer from the retrieved songs only. Invalid song IDs are dropped, with one retry. If that fails, the answer falls back to the scorer's own reasons.
- **Output:** the answer, its source (`llm`, `llm_retry`, `fallback`, `no_match`), and any guardrail events.

Humans check the system in two places: a person reviews sampled answers and the log, and the evaluation harness (`tests/eval_rag.py`) reports results before anything changes.

---

## Sample Interactions

Three real inputs and outputs, copied from the app. The first two came from the Gemini model; the third is the guardrail for a request that matches nothing.

**1. Input:** `sad piano` (Gemini answer, first run)

```
Here is the best match for your request for sad piano music, selected based on mood, acousticness, and style.

1. Moonlit Sonata by Clara Wren
   This song perfectly fits your request with a melancholic mood, high acousticness, low valence, and explicit text matches for sad piano music.

[answered by the AI model]
```

**2. Input:** `study music that helps me focus` (interactive mode, run from my terminal)

```
Here are the top lofi tracks from the catalog designed for studying and maintaining focus.

1. Focus Flow by LoRoom
   This lofi track features a focused mood and a text match for concentration and study, making it ideal for focus.
2. Midnight Coding by LoRoom
   It offers a chill lofi vibe with a tempo and acousticness profile well-suited for background study music.
3. Library Rain by Paper Lanterns
   With a low energy level, high acousticness, and lofi genre classification, this track provides a calming atmosphere for studying.

[answered by the AI model]
```

Across five repeat runs of this request, Focus Flow was the top pick every time, but the other two picks changed. Two runs needed a retry and one fell back to the scorer, because of the free-tier quota (see Testing Summary).

**3. Input:** `asdfgh qwerty` (gibberish: the guardrail refuses to guess)

```
I couldn't find anything in the catalog that matches that. Try a mood, an activity, or a genre.

[no match]
Guardrail events:
  - no retrieval signal: nothing in the catalog matched
```

---

## Design Decisions and Trade-offs

- **TF-IDF plus the weighted scorer, not embeddings.** TF-IDF needs no extra model, runs offline, and is reproducible. The trade-off is that it only matches words, so I added a synonym table ("sad" to "melancholic") and hand-written notes to each song. Embeddings would be the next step on a larger catalog.
- **Equal blend of text and structured scores (0.5 / 0.5).** I measured text weights from 0 to 1 and they tied on this test set, so I kept the simplest choice. I did not claim it is optimal.
- **A mood clash halves a song's score.** This keeps "sad piano" from being answered with a metal track that happens to match on valence. The cost is a stronger filter bubble, which the model card covers.
- **The model may only recommend retrieved songs.** Invented IDs are dropped and logged. If nothing valid remains, the app retries once, then falls back to the scorer's own reasons. I chose drop-and-continue over failing a whole answer on one bad ID.
- **No retry on a quota error.** Retrying right away cannot succeed and only uses more of the daily allowance.
- **Gemini as the default provider.** I first recommended Anthropic, but Anthropic's API needs paid credits, and the free Gemini tier let me build and test the system without cost. Claude still works through the same code path, although I have not run that path against the real API.
- **Thinking turned off for Gemini.** With the default settings the model used most of its output budget on hidden reasoning and cut off its JSON answer. The adapter also pins the v1beta API version.
- **Every fallback is visible.** A failed parse or generation shows up in the output and in `logs/app.log`, so a fallback never looks like a normal answer.
- **No minimum-score cutoff yet.** Weak but nonzero matches still reach the model. I deferred this until the evaluation shows it is needed.

---

## Testing Summary

**Automated tests:** 129 pass and 3 are skipped by default because they call the real API (run with `RUN_API_TESTS=1`). The suite covers scoring, synonyms, compatibility, the parser, the retriever, the generator, the guardrails and retry, logging, the CLI, the Gemini adapter, and the evaluation harness itself. The suite runs in about two seconds and makes no API calls.

**Offline evaluation (keyword parser, no API):** on 12 relevance queries plus 2 gibberish queries, the hybrid retriever reached 83% top-1 and top-3, compared with 50% for the original scorer. Both gibberish queries were refused. The sample is small, so these numbers are indicative only.

**What worked:** synonym mapping and mood-compatibility scoring fixed the "sad piano" failure. Gibberish is refused. In the real runs I checked by hand, every cited song was a retrieved one and no ID was dropped. The drop-and-retry path is covered by tests, not by a real failure.

**What didn't:**
- Paraphrases with no keyword overlap ("music for when I'm heartbroken", "music to fall asleep to") get no signal from the keyword parser. The real model handles these better, but that is not measured yet.
- The first version of the evaluation scored 92%. A bug let zero-score results count as hits, and I fixed it. The corrected figure is 83%.
- The full real-model evaluation (guardrail rates and consistency across repeated runs) has not been run. The free-tier quota of 20 requests per day for that model blocks it, and the harness now reports quota-limited runs separately rather than counting them as failures.

**Real-model checks so far:** three integration tests pass against the real API, and the manual runs above are real output.

---

## Getting Started

Developed and tested on Python 3.13.7. Run every command from the project folder (the one that contains `src/` and `data/`).

### Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install the pinned dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* Add an API key to enable the AI features (Google Gemini by default; Anthropic also works). Without one, everything still runs, but answers come from the scoring engine with no AI explanations.

   ```bash
   cp .env.example .env           # then edit .env and set GEMINI_API_KEY
   ```

   `.env` is git-ignored. Never put a real key in `.env.example`, which is meant to be committed. You can also set `LLM_PROVIDER` (`gemini` or `anthropic`), `LLM_MODEL` (default `gemini-3.5-flash`) and `LLM_TIMEOUT_SECONDS` (default 30) there.

### Running the app

| Command | What it does |
|---|---|
| `python -m src.main` | Interactive: type a request, get an answer. Press Enter on a blank line, or type `quit`, to exit. |
| `python -m src.main --query "sad piano"` | Answer one request and exit. |
| `python -m src.main --demo` | The original fixed-profile demo (scorer only, no API needed). |

Add `--debug` to see the parsed query and each song's retrieval scores, or `--verbose` to show INFO logs on the console. Logs always go to `logs/app.log`.

### Running tests

```bash
python -m pytest
```

This runs the unit tests, the evaluation-harness tests and a regression check that the offline evaluation stays above the frozen floors in `tests/eval_thresholds.json`. A few tests are skipped by default because they call the real LLM API. To run them:

```bash
RUN_API_TESTS=1 python -m pytest tests/test_llm_integration.py -v     # needs a key in .env
```

### Running the evaluation

```bash
python -m tests.eval_rag                  # offline: no API, reproducible
python -m tests.eval_rag --online --runs 3   # real LLM; makes roughly 100 small API calls
```

Both write a Markdown and a JSON report to `eval_results/`.

### Troubleshooting

- `ModuleNotFoundError: No module named 'src'`: run from the project folder and use `python -m ...` as shown above.
- "no API key was found": expected without a key. Add one as in step 3 to get AI explanations.
- Slow or failing answers: lower or raise `LLM_TIMEOUT_SECONDS`. After one retry the app falls back to the scorer, and `logs/app.log` records why.
- `cannot import name 'genai'` or "an API key is set, but the AI client could not start": you are running a different Python from the project's. If Conda is active (`(base)` in your prompt), use `.venv/bin/python -m ...` for every command, or deactivate Conda first.

---

## Reproducible Execution Evidence

Output from the commands below, run from the project folder with `.venv/bin/python` on macOS, Python 3.13.7. Your output will differ slightly in timestamps and in the live-model answers.

**1. One request (no match, guardrail refusal)**

```bash
.venv/bin/python -m src.main --query "asdfgh qwerty"
```

```text
AI explanations from gemini-3.5-flash.
WARNING LLM parse failed (ServerError: 503 UNAVAILABLE ...); using heuristic parser
WARNING guardrail: no retrieval signal: nothing in the catalog matched
I couldn't find anything in the catalog that matches that. Try a mood, an activity, or a genre.

[no match]
Guardrail events:
  - parse fell back to keyword parser: ServerError: 503 UNAVAILABLE ...
  - no retrieval signal: nothing in the catalog matched
```

The 503 came from a live Gemini call that was overloaded at the time. The app fell back to the keyword parser and still refused. Without an API key, the same request prints `no API key was found` and then the same refusal.

**2. Automated tests**

```bash
.venv/bin/python -m pytest
```

```text
collected 132 items
...
================ 129 passed, 3 skipped in 2.45s =================
```

The 3 skipped tests call the real API and run only with `RUN_API_TESTS=1`.

**3. Offline evaluation**

```bash
.venv/bin/python -m tests.eval_rag
```

```text
| Variant | Top-1 | Top-3 | Negative accuracy |
|---|---|---|---|
| Original scorer | 50% | 50% | 0% |
| Hybrid, text weight 0.50 (default) | 83% | 83% | 100% |
Full pipeline (offline): gibberish handled as no_match: 100%
Finished. Wrote: eval_results/offline.md, eval_results/offline.json
```

The full per-query table is in [eval_results/offline.md](eval_results/offline.md). It includes the two queries that miss (`music for when I'm heartbroken`, `music to fall asleep to`).

---

## Sample Recommendation Output

Sample output for a pop/happy profile, from an earlier single-profile version of the scoring demo (`python -m src.main --demo` now prints all seven profiles shown below):

```
Loaded songs: 18

Profile: genre=pop, mood=happy, energy=0.8
Top recommendations
============================================================
1. Sunrise City by Neon Echo
   Score: 4.96
   Reasons:
     - mood matches: happy (+2.0)
     - genre matches: pop (+1.0)
     - energy 0.82 vs target 0.80 (+1.96)
------------------------------------------------------------
2. Rooftop Lights by Indigo Parade
   Score: 3.92
   Reasons:
     - mood matches: happy (+2.0)
     - energy 0.76 vs target 0.80 (+1.92)
------------------------------------------------------------
3. Gym Hero by Max Pulse
   Score: 2.74
   Reasons:
     - genre matches: pop (+1.0)
     - energy 0.93 vs target 0.80 (+1.74)
------------------------------------------------------------
4. Concrete Crown by Verse Kingdom
   Score: 1.96
   Reasons:
     - energy 0.78 vs target 0.80 (+1.96)
------------------------------------------------------------
5. Night Drive Loop by Neon Echo
   Score: 1.90
   Reasons:
     - energy 0.75 vs target 0.80 (+1.90)
------------------------------------------------------------
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## System Evaluation: Profile Test Output

All runs use `python -m src.main --demo` on the 18-song catalog. The first three profiles are normal users; the last four are adversarial/edge cases designed to trick the scoring.

```
Profile: High-Energy Pop
============================================================
1. Sunrise City by Neon Echo
   Score: 7.19
   Reasons:
     - mood matches: happy (+2.0)
     - genre matches: pop (+1.0)
     - energy 0.82 vs target 0.90 (+1.84)
     - valence 0.84 vs target 0.80 (+0.96)
     - acousticness 0.18 vs target 0.10 (+0.92)
     - danceability 0.79 vs target 0.85 (+0.47)
------------------------------------------------------------
2. Rooftop Lights by Indigo Parade
   Score: 5.95
   Reasons:
     - mood matches: happy (+2.0)
     - energy 0.76 vs target 0.90 (+1.72)
     - valence 0.81 vs target 0.80 (+0.99)
     - acousticness 0.35 vs target 0.10 (+0.75)
     - danceability 0.82 vs target 0.85 (+0.48)
------------------------------------------------------------
3. Gym Hero by Max Pulse
   Score: 5.35
   Reasons:
     - genre matches: pop (+1.0)
     - energy 0.93 vs target 0.90 (+1.94)
     - valence 0.77 vs target 0.80 (+0.97)
     - acousticness 0.05 vs target 0.10 (+0.95)
     - danceability 0.88 vs target 0.85 (+0.48)
------------------------------------------------------------
4. Pulse Horizon by Kilo Flare
   Score: 4.29
   Reasons:
     - energy 0.95 vs target 0.90 (+1.90)
     - valence 0.79 vs target 0.80 (+0.99)
     - acousticness 0.03 vs target 0.10 (+0.93)
     - danceability 0.90 vs target 0.85 (+0.47)
------------------------------------------------------------
5. Storm Runner by Voltline
   Score: 4.07
   Reasons:
     - energy 0.91 vs target 0.90 (+1.98)
     - valence 0.48 vs target 0.80 (+0.68)
     - acousticness 0.10 vs target 0.10 (+1.00)
     - danceability 0.66 vs target 0.85 (+0.41)
------------------------------------------------------------
```

```
Profile: Chill Lofi
============================================================
1. Library Rain by Paper Lanterns
   Score: 7.35
   Reasons:
     - mood matches: chill (+2.0)
     - genre matches: lofi (+1.0)
     - energy 0.35 vs target 0.38 (+1.94)
     - valence 0.60 vs target 0.58 (+0.98)
     - acousticness 0.86 vs target 0.80 (+0.94)
     - danceability 0.58 vs target 0.60 (+0.49)
------------------------------------------------------------
2. Midnight Coding by LoRoom
   Score: 7.30
   Reasons:
     - mood matches: chill (+2.0)
     - genre matches: lofi (+1.0)
     - energy 0.42 vs target 0.38 (+1.92)
     - valence 0.56 vs target 0.58 (+0.98)
     - acousticness 0.71 vs target 0.80 (+0.91)
     - danceability 0.62 vs target 0.60 (+0.49)
------------------------------------------------------------
3. Spacewalk Thoughts by Orbit Bloom
   Score: 6.01
   Reasons:
     - mood matches: chill (+2.0)
     - energy 0.28 vs target 0.38 (+1.80)
     - valence 0.65 vs target 0.58 (+0.93)
     - acousticness 0.92 vs target 0.80 (+0.88)
     - danceability 0.41 vs target 0.60 (+0.41)
------------------------------------------------------------
4. Focus Flow by LoRoom
   Score: 5.43
   Reasons:
     - genre matches: lofi (+1.0)
     - energy 0.40 vs target 0.38 (+1.96)
     - valence 0.59 vs target 0.58 (+0.99)
     - acousticness 0.78 vs target 0.80 (+0.98)
     - danceability 0.60 vs target 0.60 (+0.50)
------------------------------------------------------------
5. Coffee Shop Stories by Slow Stereo
   Score: 4.23
   Reasons:
     - energy 0.37 vs target 0.38 (+1.98)
     - valence 0.71 vs target 0.58 (+0.87)
     - acousticness 0.89 vs target 0.80 (+0.91)
     - danceability 0.54 vs target 0.60 (+0.47)
------------------------------------------------------------
```

```
Profile: Deep Intense Rock
============================================================
1. Storm Runner by Voltline
   Score: 7.35
   Reasons:
     - mood matches: intense (+2.0)
     - genre matches: rock (+1.0)
     - energy 0.91 vs target 0.92 (+1.98)
     - valence 0.48 vs target 0.40 (+0.92)
     - acousticness 0.10 vs target 0.08 (+0.98)
     - danceability 0.66 vs target 0.60 (+0.47)
------------------------------------------------------------
2. Gym Hero by Max Pulse
   Score: 5.94
   Reasons:
     - mood matches: intense (+2.0)
     - energy 0.93 vs target 0.92 (+1.98)
     - valence 0.77 vs target 0.40 (+0.63)
     - acousticness 0.05 vs target 0.08 (+0.97)
     - danceability 0.88 vs target 0.60 (+0.36)
------------------------------------------------------------
3. Iron Requiem by Black Tundra
   Score: 4.08
   Reasons:
     - energy 0.97 vs target 0.92 (+1.90)
     - valence 0.20 vs target 0.40 (+0.80)
     - acousticness 0.04 vs target 0.08 (+0.96)
     - danceability 0.45 vs target 0.60 (+0.43)
------------------------------------------------------------
4. Night Drive Loop by Neon Echo
   Score: 3.86
   Reasons:
     - energy 0.75 vs target 0.92 (+1.66)
     - valence 0.49 vs target 0.40 (+0.91)
     - acousticness 0.22 vs target 0.08 (+0.86)
     - danceability 0.73 vs target 0.60 (+0.43)
------------------------------------------------------------
5. Pulse Horizon by Kilo Flare
   Score: 3.85
   Reasons:
     - energy 0.95 vs target 0.92 (+1.94)
     - valence 0.79 vs target 0.40 (+0.61)
     - acousticness 0.03 vs target 0.08 (+0.95)
     - danceability 0.90 vs target 0.60 (+0.35)
------------------------------------------------------------
```

```
Profile: Conflicted: Sad but Energetic
============================================================
1. Moonlit Sonata by Clara Wren
   Score: 4.07
   Reasons:
     - genre matches: classical (+1.0)
     - energy 0.22 vs target 0.90 (+0.64)
     - valence 0.25 vs target 0.20 (+0.95)
     - acousticness 0.97 vs target 0.95 (+0.98)
     - danceability 0.30 vs target 0.30 (+0.50)
------------------------------------------------------------
2. Iron Requiem by Black Tundra
   Score: 3.38
   Reasons:
     - energy 0.97 vs target 0.90 (+1.86)
     - valence 0.20 vs target 0.20 (+1.00)
     - acousticness 0.04 vs target 0.95 (+0.09)
     - danceability 0.45 vs target 0.30 (+0.42)
------------------------------------------------------------
3. Storm Runner by Voltline
   Score: 3.17
   Reasons:
     - energy 0.91 vs target 0.90 (+1.98)
     - valence 0.48 vs target 0.20 (+0.72)
     - acousticness 0.10 vs target 0.95 (+0.15)
     - danceability 0.66 vs target 0.30 (+0.32)
------------------------------------------------------------
4. Night Drive Loop by Neon Echo
   Score: 2.97
   Reasons:
     - energy 0.75 vs target 0.90 (+1.70)
     - valence 0.49 vs target 0.20 (+0.71)
     - acousticness 0.22 vs target 0.95 (+0.27)
     - danceability 0.73 vs target 0.30 (+0.29)
------------------------------------------------------------
5. Dusty Back Road by Hollis Creek
   Score: 2.86
   Reasons:
     - energy 0.51 vs target 0.90 (+1.22)
     - valence 0.68 vs target 0.20 (+0.52)
     - acousticness 0.74 vs target 0.95 (+0.79)
     - danceability 0.64 vs target 0.30 (+0.33)
------------------------------------------------------------
```

```
Profile: Unknown Genre and Mood
============================================================
1. Rooftop Lights by Indigo Parade
   Score: 4.21
   Reasons:
     - energy 0.76 vs target 0.70 (+1.88)
     - valence 0.81 vs target 0.80 (+0.99)
     - acousticness 0.35 vs target 0.20 (+0.85)
     - danceability 0.82 vs target 0.80 (+0.49)
------------------------------------------------------------
2. Sunrise City by Neon Echo
   Score: 4.20
   Reasons:
     - energy 0.82 vs target 0.70 (+1.76)
     - valence 0.84 vs target 0.80 (+0.96)
     - acousticness 0.18 vs target 0.20 (+0.98)
     - danceability 0.79 vs target 0.80 (+0.49)
------------------------------------------------------------
3. Concrete Crown by Verse Kingdom
   Score: 4.05
   Reasons:
     - energy 0.78 vs target 0.70 (+1.84)
     - valence 0.62 vs target 0.80 (+0.82)
     - acousticness 0.12 vs target 0.20 (+0.92)
     - danceability 0.85 vs target 0.80 (+0.48)
------------------------------------------------------------
4. Night Drive Loop by Neon Echo
   Score: 4.04
   Reasons:
     - energy 0.75 vs target 0.70 (+1.90)
     - valence 0.49 vs target 0.80 (+0.69)
     - acousticness 0.22 vs target 0.20 (+0.98)
     - danceability 0.73 vs target 0.80 (+0.46)
------------------------------------------------------------
5. Island Breeze by Sun Rootz
   Score: 3.93
   Reasons:
     - energy 0.58 vs target 0.70 (+1.76)
     - valence 0.88 vs target 0.80 (+0.92)
     - acousticness 0.45 vs target 0.20 (+0.75)
     - danceability 0.80 vs target 0.80 (+0.50)
------------------------------------------------------------
```

```
Profile: Dead Center (all 0.5)
============================================================
1. Sunrise City by Neon Echo
   Score: 6.05
   Reasons:
     - mood matches: happy (+2.0)
     - genre matches: pop (+1.0)
     - energy 0.82 vs target 0.50 (+1.36)
     - valence 0.84 vs target 0.50 (+0.66)
     - acousticness 0.18 vs target 0.50 (+0.68)
     - danceability 0.79 vs target 0.50 (+0.35)
------------------------------------------------------------
2. Rooftop Lights by Indigo Parade
   Score: 5.36
   Reasons:
     - mood matches: happy (+2.0)
     - energy 0.76 vs target 0.50 (+1.48)
     - valence 0.81 vs target 0.50 (+0.69)
     - acousticness 0.35 vs target 0.50 (+0.85)
     - danceability 0.82 vs target 0.50 (+0.34)
------------------------------------------------------------
3. Velvet Hours by Amara Sky
   Score: 4.07
   Reasons:
     - energy 0.48 vs target 0.50 (+1.96)
     - valence 0.66 vs target 0.50 (+0.84)
     - acousticness 0.38 vs target 0.50 (+0.88)
     - danceability 0.72 vs target 0.50 (+0.39)
------------------------------------------------------------
4. Midnight Coding by LoRoom
   Score: 4.01
   Reasons:
     - energy 0.42 vs target 0.50 (+1.84)
     - valence 0.56 vs target 0.50 (+0.94)
     - acousticness 0.71 vs target 0.50 (+0.79)
     - danceability 0.62 vs target 0.50 (+0.44)
------------------------------------------------------------
5. Dusty Back Road by Hollis Creek
   Score: 3.99
   Reasons:
     - energy 0.51 vs target 0.50 (+1.98)
     - valence 0.68 vs target 0.50 (+0.82)
     - acousticness 0.74 vs target 0.50 (+0.76)
     - danceability 0.64 vs target 0.50 (+0.43)
------------------------------------------------------------
```

```
Profile: Out of Range (energy 2.0)
============================================================
1. Sunrise City by Neon Echo
   Score: 4.99
   Reasons:
     - mood matches: happy (+2.0)
     - genre matches: pop (+1.0)
     - energy 0.82 vs target 2.00 (+-0.36)
     - valence 0.84 vs target 0.80 (+0.96)
     - acousticness 0.18 vs target 0.10 (+0.92)
     - danceability 0.79 vs target 0.85 (+0.47)
------------------------------------------------------------
2. Rooftop Lights by Indigo Parade
   Score: 3.74
   Reasons:
     - mood matches: happy (+2.0)
     - energy 0.76 vs target 2.00 (+-0.48)
     - valence 0.81 vs target 0.80 (+0.99)
     - acousticness 0.35 vs target 0.10 (+0.75)
     - danceability 0.82 vs target 0.85 (+0.48)
------------------------------------------------------------
3. Gym Hero by Max Pulse
   Score: 3.27
   Reasons:
     - genre matches: pop (+1.0)
     - energy 0.93 vs target 2.00 (+-0.14)
     - valence 0.77 vs target 0.80 (+0.97)
     - acousticness 0.05 vs target 0.10 (+0.95)
     - danceability 0.88 vs target 0.85 (+0.48)
------------------------------------------------------------
4. Pulse Horizon by Kilo Flare
   Score: 2.29
   Reasons:
     - energy 0.95 vs target 2.00 (+-0.10)
     - valence 0.79 vs target 0.80 (+0.99)
     - acousticness 0.03 vs target 0.10 (+0.93)
     - danceability 0.90 vs target 0.85 (+0.47)
------------------------------------------------------------
5. Storm Runner by Voltline
   Score: 1.91
   Reasons:
     - energy 0.91 vs target 2.00 (+-0.18)
     - valence 0.48 vs target 0.80 (+0.68)
     - acousticness 0.10 vs target 0.10 (+1.00)
     - danceability 0.66 vs target 0.85 (+0.41)
------------------------------------------------------------
```

---

## Experiments You Tried

**Weight shift:** energy weight 2.0 -> 4.0 and genre bonus 1.0 -> 0.5 (max score 7.5 -> 9.0; every term still non-negative for targets in 0-1, so the math stays valid). Reverted afterward.

| Profile | Baseline top 5 | After weight shift |
|---|---|---|
| High-Energy Pop | Sunrise City, Rooftop Lights, Gym Hero, Pulse Horizon, Storm Runner | same order |
| Chill Lofi | Library Rain, Midnight Coding, Spacewalk, Focus Flow, Coffee Shop | same order |
| Deep Intense Rock | Storm Runner, Gym Hero, Iron Requiem, Night Drive Loop, Pulse Horizon | Storm Runner, Gym Hero, Iron Requiem, Pulse Horizon, Concrete Crown |
| Sad but Energetic | **Moonlit Sonata**, Iron Requiem, Storm Runner, Night Drive, Dusty Back Road | **Iron Requiem**, Storm Runner, Night Drive, Gym Hero, Sunrise City |
| Unknown Genre and Mood | Rooftop, Sunrise, Concrete Crown, Night Drive, Island Breeze | Rooftop, Sunrise, Night Drive, Concrete Crown, Island Breeze |
| Dead Center | Sunrise, Rooftop, Velvet Hours, Midnight Coding, Dusty Back Road | Sunrise, Rooftop, Velvet Hours, Dusty Back Road, Midnight Coding |
| Out of Range | Sunrise, Rooftop, Gym Hero, Pulse Horizon, Storm Runner | same order |

Result: different, not clearly more accurate. Normal profiles barely moved (their winners already matched on energy). The only big change was the conflicted profile: Moonlit Sonata (a very low-energy song) fell out of first place, so the energy number "won" over the genre/acoustic taste the user also stated. Whether that is better depends on which signal the user meant.


---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

The main lesson was that a recommender's behavior comes from its rules, and a small rule change can alter results a lot. Adding retrieval and a guarded LLM step made the system more useful, but also made failures harder to see, so I added logging, tests and an evaluation harness to catch them.

The graded responsible-AI reflection (limitations, misuse, surprises during testing, and how I worked with AI) is in [model_card.md](model_card.md), sections 9 to 11.



