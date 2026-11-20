---
name: related-words
description: "Give it any sentence or passage and it returns English words that relate to it in meaning, each with a similarity score, part of speech, Zipf frequency (1.5-5.0, rarer = lower) and its dictionary meaning. Gemma-256 retrieval over a 180k-sense index, runs on CPU."
---

# related-words

Text in -> related words out (with score + meaning). One command:

```bash
python <skill-dir>/run.py "The committee's ambiguous remarks sparked a protracted debate."
python .../run.py abc xyz ...          # all positional args are joined into one text
echo "some text" | python .../run.py   # or stdin
```

Options: `-n N` words in the final list (default 20), `-k N` words per chunk (default 20), `--chunks` per-chunk detail, `--full` untruncated meanings, `--json`, `--no-daemon`, `--stop`.

Output is a short markdown list: `1. **word** (pos, zipf, score) - meaning`, best match first. Use `--json` for parsing.

## Warm server (automatic)
The first call spawns a background server that keeps Gemma + the index in memory (~9 s once). Further calls reuse it (~0.1 s). It exits by itself after 60 s with no requests (`IDLE=secs` to change), freeing the memory. `--stop` kills it now; `--no-daemon` loads in-process. Socket: `/tmp/related-words-<uid>.sock`.

## How it works
1. Chunks the text (sentences, then splits >22-word sentences at , ; : and/but/because..., merges <5-word bits).
2. Embeds each chunk with `google/embeddinggemma-300m` (query prompt), truncates to 256-d (MRL) and L2-normalises.
3. Cosine against the index in `resources/` (144,908 senses x 256-d int8, 67,160 lemmas, zipf 1.5-5.0), keeps the best sense per word.
4. Prints per-chunk top-K, then a passage-level list (best score per word across chunks).

## Resources (inside this skill)
- `resources/emb256_i8.npy` + `emb256_scale.npy` (int8 vectors + per-row scale, 37 MB), `resources/meta_small.parquet` (lemma, pos, zipf, meaning; 4 MB). Total 41 MB.
- Pruned from the 104 MB original (180k senses): dropped 15,396 near-duplicate senses (cos>0.92 to an earlier sense of the same word), 17,490 senses past the 8th per word, 2,103 place/taxon/surname senses; vectors int8; meanings cut at 220 chars. On 10 test sentences the returned words match the full index 96% of the time (top-1 same 9/10).
- The Gemma model (1.2 GB, gated on HF) is NOT stored here. Default path `~/word-sense-retrieval/models/embeddinggemma-300m`; override with `GEMMA=/path`. Otherwise it falls back to the HF id (needs a token + network; offline mode is on by default, unset `HF_HUB_OFFLINE`).

## Speed (i5-1135G7, 4 cores / 8 threads, measured)
- Cold start ~9 s (imports + model load + index); warm calls ~0.1 s. Per chunk ~38 ms encode; search 10-47 ms total.
- Threads default = half the logical CPUs (4 here): fastest, ~20% faster than 8 threads. bf16 is slower on this CPU; int8 is ~2x faster but changes 10-25% of top-5 words, so it is not used.

## Quality (project evals, Gemma-256): definition->word MRR 0.41, R@10 0.67; situation->word MRR 0.15. Results are semantic matches; use zipf to prefer rarer words.
