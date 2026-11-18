# RAG evaluation (offline mode)

Generated 2026-11-18T01:52:23+00:00 on a 18-song catalog: 12 relevance queries and 2 gibberish queries.

> With only 14 queries these results are indicative, not statistically representative. Ground truth was written before results were examined.

## Retrieval: top-1 / top-3 hit rate

Every row is scored on the same parsed query per query. Hit = acceptable genre/mood (and any required field). Negative accuracy = gibberish correctly yields no signal; the original scorer never refuses, so it cannot score here.

| Variant | Top-1 | Top-3 | Negative accuracy | Top-1 hits meeting supporting ranges |
|---|---|---|---|---|
| Original scorer (no text, no synonyms/compatibility) | 50% | 50% | 0% | 100% |
| Structured only (scorer via retrieval adapter) | 83% | 83% | 100% | 100% |
| Hybrid, text weight 0.25 | 83% | 83% | 100% | 100% |
| Hybrid, text weight 0.50 (default) | 83% | 83% | 100% | 100% |
| Hybrid, text weight 0.75 | 83% | 83% | 100% | 100% |
| Text only (TF-IDF) | 83% | 83% | 100% | 100% |

## Full pipeline (offline)

Gibberish handled as no_match by the real pipeline: 100%. Guardrail rates and consistency need `--online`.

## Per query (hybrid, text weight 0.50)

| # | Query | Tests | First hit rank | Top 3 (✓ hit, ✗ miss, – no signal) |
|---|---|---|---|---|
| q01 | mellow music for studying on a rainy night | scene words | 1 | ✓ Library Rain; ✓ Midnight Coding; ✓ Spacewalk Thoughts |
| q02 | sad piano | synonym map + instrument proxy | 1 | ✓ Moonlit Sonata; ✗ Pine Smoke Lullaby; ✗ Spacewalk Thoughts |
| q03 | dark synthwave for a night drive | explicit genre + scene | 1 | ✓ Night Drive Loop; ✗ Iron Requiem; ✗ Moonlit Sonata |
| q04 | romantic date night, slow and smooth | mood word + tempo | 1 | ✓ Velvet Hours; ✗ Moonlit Sonata; ✗ Midnight Coding |
| q05 | aggressive heavy music | synonym map | 1 | ✓ Iron Requiem; ✗ Storm Runner; ✗ Pulse Horizon |
| q06 | high energy workout music for the gym | activity + energy | 1 | ✓ Gym Hero; ✓ Storm Runner; ✗ Night Drive Loop |
| q07 | something to pump me up before a race | paraphrase: energy inference | 1 | ✓ Gym Hero; – Coffee Shop Stories; – Concrete Crown |
| q08 | music for when I'm heartbroken | paraphrase: emotional inference | none | – Coffee Shop Stories; – Concrete Crown; – Dusty Back Road |
| q09 | something unobtrusive to keep me company while I work through a long programming session | paraphrase: context inference | 1 | ✓ Focus Flow; – Coffee Shop Stories; – Concrete Crown |
| q10 | songs that make a long drive through the city feel like a movie | paraphrase: cinematic scene inference | 1 | ✓ Night Drive Loop; ✗ Concrete Crown; – Coffee Shop Stories |
| q11 | something cheerful for a beach day | mixed: mood + scene | 1 | ✓ Sunrise City; ✓ Rooftop Lights; ✓ Island Breeze |
| q12 | music to fall asleep to | paraphrase: no mood word | none | – Coffee Shop Stories; – Concrete Crown; – Dusty Back Road |
| n01 | asdfgh qwerty | negative: gibberish | n/a | no_match ✓ |
| n02 | zxqv blorf 9182 | negative: gibberish | n/a | no_match ✓ |

## Hybrid misses (not top-1), with parsed query and scores

**q08: music for when I'm heartbroken**  
parsed: `{"genres": [], "moods": [], "scenes": [], "synonyms": ["when", "m", "heartbroken"], "numeric_features": {}}`  
accepts: `{"mood": ["melancholic"]}`  
returned: Coffee Shop Stories (jazz/relaxed, 0.0, no signal); Concrete Crown (hip hop/confident, 0.0, no signal); Dusty Back Road (country/nostalgic, 0.0, no signal)

**q12: music to fall asleep to**  
parsed: `{"genres": [], "moods": [], "scenes": [], "synonyms": ["fall", "asleep"], "numeric_features": {}}`  
accepts: `{"genre": ["ambient", "folk"], "mood": ["dreamy", "chill", "relaxed"]}`  
returned: Coffee Shop Stories (jazz/relaxed, 0.0, no signal); Concrete Crown (hip hop/confident, 0.0, no signal); Dusty Back Road (country/nostalgic, 0.0, no signal)

