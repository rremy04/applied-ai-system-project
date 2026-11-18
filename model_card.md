# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**VibeFinder 1.0**

---

## 2. Intended Use  

VibeFinder suggests five songs from a small catalog based on one listener profile: a favorite genre, a favorite mood, and target levels for energy, brightness, acoustic sound and danceability.

It assumes a listener has one steady taste. It is for classroom exploration only.

**Not for:** real listeners, real music apps, or any decision about artists. It has 18 songs and no listening history, so it cannot be fair to everyone.

---

## 3. How the Model Works  

The recommender compares every song in the catalog to one description of what the listener wants, gives each song points, and shows the highest scorers.

A song earns points in six ways:

- **Mood:** 2 points if it matches the listener's favorite mood, otherwise none.
- **Genre:** 1 point if it matches the favorite genre, otherwise none.
- **Energy:** up to 2 points, more the closer the song's energy is to what the listener wants.
- **Valence (how bright or dark it sounds):** up to 1 point, using the same closeness idea.
- **Acousticness (organic vs. electronic):** up to 1 point, using the same closeness idea.
- **Danceability:** up to half a point, since it mostly repeats energy.

Closeness works like a ruler: a perfect match gets full points, and points shrink as the gap grows. A song that is too low and a song that is too high are treated the same.

Mood counts for more than genre because mood is how a song feels, and people will cross genres to get that feeling. Most genres appear only once in the catalog, so a big genre bonus would just pick that one song.

After scoring every song, the system sorts from highest to lowest, breaks ties alphabetically by title, and returns the top five. Each pick comes with the reasons that earned its points, such as "mood matches: happy (+2.0)".

Changes from the starter: the starter returned the first songs in the file with a placeholder explanation. This version loads the CSV, scores every song, ranks them, and explains each result.

---

## 4. Data  

The catalog is `data/songs.csv`. It has 18 songs, each with genre, mood, energy, tempo, valence, danceability and acousticness.

There are 15 genres and 14 moods. Most genres and moods show up only once. Lofi (3 songs) and chill (3 songs) are the biggest groups, and pop has 2.

I did not add or remove songs. Missing from the data: lyrics, language, artist popularity, year, and anything about what a person has actually listened to.

---

## 5. Strengths  

It works well when the listener's taste is clear and the labels line up. Chill Lofi and Deep Intense Rock both got lists I would agree with.

It separates very different tastes. The Chill Lofi and Deep Intense Rock lists share no songs.

Every pick comes with reasons, so it is easy to see why a song ranked where it did.

---

## 6. Limitations and Bias 

The biggest weakness I found is that the system matches labels as exact words, so near-synonyms count as total misses. My "sad but energetic" test user asked for mood "sad," but the catalog's closest song is labeled "melancholic," so no song earned the 2-point mood bonus. The top pick, Moonlit Sonata, won only on genre plus acousticness and valence, even though its energy (0.22) is nowhere near the 0.90 the user wanted. A second weakness is that the catalog is tiny (18 songs, 14 genres appearing once), so a genre or mood the user names often matches just one song, and high-energy fans keep getting the same few songs (Gym Hero, Storm Runner, Pulse Horizon). Finally, a target outside 0 to 1 (like energy 2.0) is not rejected; it quietly produces negative points instead of an error.

---

## 7. Evaluation  

**Profiles tested.** Three normal users (High-Energy Pop, Chill Lofi, Deep Intense Rock) and four adversarial ones: Sad but Energetic (conflicting wishes), Unknown Genre and Mood (k-pop / "bubbly," which are not in the catalog), Dead Center (every target 0.5), and Out of Range (energy 2.0). Full top-5 terminal output is in the README under "System Evaluation."

**Do the results feel right?** For the normal profiles, yes. Chill Lofi returns the three lofi songs plus ambient and jazz, all quiet and acoustic, which is what I would expect. Deep Intense Rock puts Storm Runner first by a wide margin, and metal (Iron Requiem) is third, which makes sense musically.

**Why Gym Hero keeps showing up for Happy Pop fans.** Gym Hero is pop but its mood is "intense," not "happy," so it gets no mood points. It still ranks third for the High-Energy Pop profile (5.35) because it is pop (+1.0) and its energy, brightness and electronic sound are almost exactly what that listener asked for (about 4.3 points from closeness). In plain terms: the system sees "pop, loud, upbeat, danceable" and does not know the song is a workout track rather than a happy one. Mood is a single yes/no, so a wrong mood only costs 2 points and strong numbers can cover it.

**Surprises.**
- Sad but Energetic: no song is labeled "sad," so the classical song with the right genre and sound won even though it is the opposite of high energy. Iron Requiem and Storm Runner (energy match, wrong everything else) were close behind.
- Unknown Genre and Mood: with no label matches possible, the top five scores are within 0.3 points of each other (4.21 to 3.93), so the ranking is basically a coin flip on small numeric gaps.
- Out of Range: energy 2.0 gave negative energy points (e.g. -0.36 for Sunrise City) but the order stayed sensible, so the bad input was not caught.

**Experiment: weight shift.** I doubled the energy weight (2.0 to 4.0) and halved the genre bonus (1.0 to 0.5); the maximum score goes from 7.5 to 9.0 and nothing can go negative for valid targets. The normal profiles barely changed, and the conflicted profile changed the most (Moonlit Sonata dropped from 1st to out of the top 5 while Iron Requiem took 1st). The change made the output different, not clearly more accurate: it fixed the "sad but energetic" case only by trusting the energy number over the user's other stated tastes. I reverted it afterward.

**Profile-by-profile comparisons.**
- *High-Energy Pop vs. Chill Lofi:* nearly opposite lists. Pop gets bright, loud, electronic songs (Sunrise City, Pulse Horizon); lofi gets quiet, acoustic ones (Library Rain, Spacewalk Thoughts). That is exactly what the energy and acousticness targets are testing.
- *High-Energy Pop vs. Deep Intense Rock:* both want high energy, so Gym Hero and Pulse Horizon appear in both. The difference is the top: mood "happy" brings Sunrise City and Rooftop Lights, mood "intense" brings Storm Runner, and a lower valence target pulls in darker songs like Iron Requiem.
- *Chill Lofi vs. Deep Intense Rock:* no overlap at all, which is a good sign the scoring separates very different tastes.
- *Sad but Energetic vs. Chill Lofi:* both like acoustic, low-valence songs, but the conflicted user's high energy pulls in Iron Requiem and Storm Runner, which a calm listener would never see.
- *Unknown Genre and Mood vs. High-Energy Pop:* both still lead with Rooftop Lights and Sunrise City, because their numbers (bright, danceable, fairly energetic) fit. The difference is below them: with no mood or genre bonus available, Concrete Crown, Night Drive Loop and Island Breeze fill the list on small numeric gaps instead of Gym Hero.
- *Dead Center vs. Out of Range:* both still list Sunrise City and Rooftop Lights first because the happy/pop labels are the same; Dead Center's lower targets add mid-energy songs (Velvet Hours, Dusty Back Road) while Out of Range keeps the loud ones.

---

## 8. Future Work  

- Treat close moods as partly matching, so "sad" gets some credit for "melancholic."
- Reject or clamp targets outside 0 to 1.
- Add a diversity rule so the top five are not all near-copies of one song.
- Let a user have two tastes, like gym music and study music, instead of one target.

---

## 9. RAG Extension: Evaluation and Limitations

**What was added.** Free-text requests ("sad piano") are parsed into a structured query, retrieved with a blend of TF-IDF text similarity and the weighted scorer above, and then explained by an LLM (Google Gemini by default) using only the retrieved songs. A guardrail drops song IDs that were not retrieved, retries once, and falls back to the scorer's own reasons. A query that retrieves nothing is answered with "no match" instead of a guess.

**How it was evaluated.** `python -m tests.eval_rag` runs 14 queries (12 relevance, 2 gibberish). Ground truth lists acceptable genres and moods, never song IDs, and was written before any results were looked at. A hit means the returned song has an acceptable genre or mood (and any required field). Every retrieval variant is scored on the same parsed query, so differences come from retrieval and not from parsing. Full per-query output is in `eval_results/offline.md` and `.json`.

**Offline results (keyword parser, no API, 12 relevance queries).**

| Variant | Top-1 | Top-3 | Gibberish refused |
|---|---|---|---|
| Original scorer alone | 50% | 50% | 0% |
| Structured only (scorer through the retrieval adapter) | 83% | 83% | 100% |
| Hybrid, text weight 0.25 / 0.50 / 0.75 | 83% | 83% | 100% |
| Text only (TF-IDF) | 83% | 83% | 100% |

**What the numbers do and do not show.**
- The jump from 50% to 83% comes from synonym normalization and compatibility scoring in the retrieval adapter. Adding text search on top changed nothing measurable: every text weight from 0 to 1 tied. This test set cannot say which weight is best, and 0.5 should not be described as optimal.
- Top-3 can never exceed top-1 here, because the catalog is so small. It is a weak metric until the catalog grows.
- With 12 relevance queries these figures are indicative only. A frozen floor in `tests/eval_thresholds.json` guards against regressions; it is not evidence of general recommendation quality.

**Offline limitations (stated plainly).**
- The offline parser has no sense of meaning. "Music for when I'm heartbroken" and "music to fall asleep to" produced no signal at all, so both are misses (the real pipeline answers "no match" for them). I deliberately did not add "heartbroken" to the synonym table, because that would be fitting the system to the test.
- The paraphrase queries are weaker generalization tests than intended. Three of the hits share a word with the hand-written notes column (pump, work, drive), so they partly measure vocabulary I wrote myself.
- The offline parser splits "I'm" into "I" and "m" and lets "m" through as a search term. It is harmless here but is a real tokenization bug.
- Synonym normalization and compatibility scoring are bundled into the adapter, so the 50% to 83% gain cannot be split between them. The mood-clash and implied-feature tables are my own judgment calls (for example, "moody" clashes with "happy").
- Moods are still single labels, a clash halves a song's score by a fixed factor, and nothing checks whether an explanation is true. A valid song ID with a wrong claim ("calm" for a metal track) passes the guardrail.
- No minimum retrieval score is applied beyond "top score is zero". Weak but nonzero matches still reach the model.
- A measurement bug was caught during review: the first run reported 92% because zero-score padding results (sorted alphabetically) were graded as hits. One of them matched by luck. The harness now gives no credit to results with no signal, and a regression test covers it. The honest baseline is 83%.

**Bias and risk in the RAG layer.**
- *Tiny corpus.* The catalog has 18 songs, 15 genres and 14 moods, and most labels appear once. A request usually has one obvious answer, so retrieval quality is hard to judge and a good score may just mean "the one song with that label." Across the 12 relevance queries the top 3 drew on 16 of the 18 songs, and two songs (Coffee Shop Stories and Dusty Back Road) were never retrieved by any of them.
- *Filter-bubble effect.* The system returns the songs closest to what was asked for, and the compatibility rule goes further: a song whose mood clashes with the request has its score halved. A sad request will therefore almost never surface an upbeat song. That is intended, but it narrows exposure by design, and there is no diversity rule or "something different" option. In the offline run Moonlit Sonata and Night Drive Loop were returned most often (3 of 25 top-3 slots each). Those queries were deliberately varied, so this understates how repetitive a single listener's session would feel, and repetition over a session was not measured.
- *LLM hallucination risk.* The guardrail verifies that every recommended song ID was actually retrieved, which stops invented songs. It does not verify that the explanation is true. The model could still attribute qualities a song does not have, or add details beyond the attributes it was given. The prompt restricts it to the listed attributes, but how often it strays has not been measured: only a few real calls have been checked by hand, and the full online evaluation has not been run. The fallback explanations are built directly from the scorer's reasons, so they cannot invent anything. The user's request is also untrusted text; the prompt tells the model to ignore instructions inside it that conflict with the rules, but that was not tested against deliberate prompt-injection attempts.

**Provider notes.** The default is Google Gemini (`gemini-3.5-flash`); Anthropic's Claude is supported in the same code path, but that path has never been run against the real API, only against fakes. Three findings from testing Gemini that shaped the adapter: with default settings the model spent 480 of a 500-token budget on hidden "thinking", which cut the JSON answer off after 37 characters, so thinking is switched off; the default API version rejected system instructions on some models, so the adapter pins `v1beta`; and model names change (one lighter model was already closed to new accounts), so the model is a setting, not a constant.

**Not yet measured: the online run.** Guardrail pass, retry and fallback rates, repeated-run consistency, and how real LLM parsing handles the paraphrase queries have not been measured. Run `python -m tests.eval_rag --online` with a key in `.env`. Until that has run, none of those claims should be made. Only a few real calls have been checked so far (a manual query and three integration tests, all passing on `gemini-3.5-flash`).

---

## 10. Personal Reflection  

The biggest thing I learned while working on this project is that the weights are opinions. I picked mood = 2 points and genre = 1 point because it felt right, and then a "sad" test user got no mood points at all because the dataset says "melancholic." Nothing was broken. The rules just did what I told them.

AI tools were good for getting the structure up fast and for thinking of strange test profiles. I still had to read the actual scores myself. The tests passed the whole time, but they would never have told me that an energy of 2.0 gives negative points. I only saw that by reading the output.

What surprised me is how much it feels like a real recommender. Chill Lofi gave me a playlist I would actually use, and it is just addition and subtraction. Then Gym Hero showed up for the happy pop listener and I could see how a feed can feel almost right and still be off.

Next I would try the mood fix first, then let one user have two profiles. I want to see if the Gym Hero problem goes away.

Overall, this was truly a fun project to work on as someone who loves Spotify so getting to work on something like this was exciting

The biggest lesson about working with AI was that I had to fact-check everything it generated. Not everything it proposed was what I was trying to achieve. When it offered choices on how to implement something, instead of giving me only one way, I could make my own decisions about which option fit the project. That made me the one accountable for the result.

---

## 11. Responsible AI Reflection

### Limitations and biases

The main limits are listed in sections 6 and 9. The short version: an 18-song catalog where most labels appear once, a keyword parser that cannot understand paraphrases, an LLM explanation step that is not checked for truth (only for valid song IDs), and an evaluation too small to show general quality. Bias is mostly structural. The mood-clash penalty keeps the system from answering a sad request with an upbeat song. That protects against bad matches, but it also narrows what a listener sees.

### Could the system be misused?

- **Emotional targeting.** A request like "music for when I'm heartbroken" gets melancholy songs back, and nothing in the system checks whether a listener is in a fragile state. Repeatedly returning sad songs could deepen a low mood. It should not be presented as wellbeing support. Mitigation: the system stores no listener history, and this card says it is not a clinical or wellbeing tool.
- **Private text in logs.** Every query is written in plain text to `logs/app.log`. A user could type something personal into it. Mitigation: the log is git-ignored and local, but a deployed version would need redaction and a retention limit.
- **Prompt injection through the request.** The request text goes into the prompt, so a crafted request could try to override the rules. The prompt marks the request as untrusted and tells the model to follow the rules, but I have not tested this against deliberate attacks. The guardrail limits the damage: the model can only name retrieved songs, so an attack cannot make it recommend something outside the catalog.
- **Cost abuse.** Anyone with the API key can spend it. Mitigation: the key lives only in the git-ignored `.env`, each call has a timeout, a request makes at most three model calls (parse, generate, one retry), and the free tier's daily quota caps the damage in this setup.

### What surprised me while testing reliability

- **My first accuracy number was wrong.** The harness reported 92%, but zero-score results were graded as hits, and one of them only matched by alphabetical luck. The honest figure is 83%. I added a regression test so the bug can't come back.
- **Gemini spent its budget on hidden reasoning.** With default settings, the model used 480 of 500 output tokens on reasoning, and the JSON answer was cut off after 37 characters. The fix was to turn thinking off, but the failure was silent until I looked at the token counts.
- **A quota error looked like a model failure.** On the free tier, a daily limit made answers fall back and retry, which looked like the recommender breaking. The retry could never succeed, so the system now skips it, and the evaluation reports quota-limited runs separately.
- **The fallback was silent.** When the parser lost its connection, the app quietly switched to keyword matching. Only the debug view showed it. It is now a visible guardrail event.
- **Top-1 was stable but the rest changed.** Across five runs of one request, the top pick never changed, but the second and third picks moved. The measured consistency is still thin, because the full repeated-run evaluation has not been run.

### How I worked with AI

I used an AI assistant (Claude) throughout the project for design, code, tests and documentation. Two concrete examples, one accepted and one changed:

- **Accepted, with changes: the guardrail policy.** The assistant proposed dropping invalid song IDs, retrying once when nothing valid remains, and then falling back to the scorer. I accepted this. It is the right trade-off because one bad ID should not discard a good answer. I did add the no-retry rule for quota errors, which the proposal did not cover.
- **Not accepted: the Anthropic-first plan.** The assistant recommended an Anthropic key. I chose Gemini instead, because Anthropic's API needs paid credits and I did not want to spend money on the project. That meant building a provider adapter, and testing against the real API exposed two problems the original plan did not foresee: thinking tokens truncating the answer, and the default API version rejecting system instructions.

Other suggestions were checked against the code and the evaluation rather than taken on trust. The assistant's first grading of the evaluation counted zero-score results as hits, and that was caught by reading the per-query rows, not by the tests. Every number in this card was produced by the code in this repository, and the real-model evaluation remains unrun.

