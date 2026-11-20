# packaging.md - shrinking the index, speed, and the skill

## What ships
`skills/related-words/` = `run.py` + `SKILL.md` + `resources/` (41 MB). Gemma itself (1.2 GB, gated) is not included; point `GEMMA=/path/to/embeddinggemma-300m` at a local copy or let it fall back to the Hugging Face id (needs a token).

## Pruning the index: 104 MB -> 41 MB (`scripts/index/prune_index.py`)
Starting point: 179,897 senses x 256 fp16 (92 MB) + metadata (11.7 MB).

Analysis of the original (share of senses):
| Candidate "useless" group | Senses | % |
|---|---|---|
| near-duplicate sense of the same word (cos > 0.92 to an earlier one) | 15,396 | 8.6 |
| senses past the 6th for a word (3,047 words have >8 senses) | 28,604 | 16 |
| zipf < 1.8 (very obscure) | 17,092 | 9.5 |
| zipf >= 4.5 (already known) | 17,637 | 9.8 |
| meaning under 20 characters | 16,862 | 9.4 |
| hyphenated / multi-word entries | 16,697 | 9.3 |
| place / species / surname senses | 2,132 | 1.2 |
| meaning refers to "such/this/the same" | 8,192 | 4.6 |
| spelling-variant glosses | 92 | 0.1 |
Frequency spread is even (zipf buckets 2.0-5.0 each hold 26-40k senses). Plain adverbs are 2.5%.

Dropped (budget was 45 MB): near-duplicates 15,396; place/taxon/surname 2,103; sense number 9+ per word 17,490. Kept obscure and very common words (your call; dropping them would shrink further). Result: **144,908 senses, 67,160 of 67,913 words (753 lost)**.

Storage: int8 per-row symmetric quantisation (`scale = max|row| / 127`, scale stored fp16), metadata slimmed to lemma/pos/zipf(fp16)/meaning (cut at 220 chars), zstd-19 parquet.

| File | Size |
|---|---|
| `emb256_i8.npy` | 37.1 MB |
| `emb256_scale.npy` | 0.3 MB |
| `meta_small.parquet` | 3.8 MB |
| **total** | **41.2 MB** |
At load `run.py` multiplies back to fp32 (about 150 MB of RAM in the warm server).

## Quality checks after pruning
- int8 alone: top-10 overlap 0.993 against fp16 on 300 perturbed index queries.
- Whole pruned index vs full on 10 hand-written sentences (`scripts/index/compare_pruned.py`): returned words match **96%**, top-1 identical 9/10. This is a small, informal test - not a re-run of eval A/B/C.

## Speed (laptop i5-1135G7, 4 cores / 8 threads; `scripts/speed/`)
| Option | Result |
|---|---|
| fp32, 4 threads (**chosen**) | ~38 ms per chunk; lossless |
| fp32, 8 threads (torch default) | ~20% slower |
| bf16 | slower (no native bf16 on this CPU) |
| int8 dynamic quantised model | up to ~2x faster, but top-5 overlap 0.75-0.90 -> rejected |
| batch size, max_len | no difference |
| ONNX / onnxruntime | export failed (needs `onnxscript`), not pursued |
| search over 180k senses | ~25 ms (torch fp32); less on the pruned index |
A 123-word passage takes about 0.45 s end to end once the model is loaded.

Warm server (in `run.py`): first call spawns a background process holding the model + index (cold start about 9 s), later calls take about 0.1 s, it exits after 60 s idle (`IDLE=secs`), socket `/tmp/related-words-<uid>.sock` mode 600. `--stop` kills it, `--no-daemon` runs in-process.

## Output format
Markdown list for agents: `1. **word** (pos, zipf, score) - meaning`; `--chunks` per-chunk detail; `--json` for parsing; `--full` untruncated meanings; `-n` list length, `-k` words per chunk.

## Using it as an AI-agent skill
Copy `skills/related-words/` into your agent's skills folder (for Claude Code: `~/.claude/skills/` or a project `.claude/skills/`). `SKILL.md` has the name/description frontmatter the agent reads; the agent then runs `python <skill-dir>/run.py "text"`.
