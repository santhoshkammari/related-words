# related-words

A fun side project: give it a sentence or a passage and get back English words that relate to it in meaning, each with its meaning, part of speech, rarity (zipf) and match score.

```
$ python skills/related-words/run.py "He was known for his meticulous attention to detail."
1. **meticulous** (adj, zipf 3.2, score 0.56) - Characterized by very precise, conscientious attention to details.
2. **painstaking** (adj, zipf 2.9, score 0.55) - Carefully attentive to details; studious; diligent in performing a process or procedure.
3. **neat-handed** (adj, zipf 3.8, score 0.53) - Dextrous; having the skill and discipline to do precision work neatly.
```
It is search over dictionary meanings: [EmbeddingGemma-300m](https://huggingface.co/google/embeddinggemma-300m) cut to 256 dims, compared with a 144,908-sense Wiktionary index (67,160 words). CPU only. The first call loads the model (~9 s) and leaves a background server for 60 s, so later calls take ~0.1 s.

## Quick start
```bash
pip install torch sentence-transformers pandas numpy pyarrow
# Gemma is gated: accept the terms at the model page, then either
export GEMMA=/path/to/embeddinggemma-300m        # local copy, or
huggingface-cli login && unset HF_HUB_OFFLINE    # download on first use
python skills/related-words/run.py abc xyz ...   # text = all arguments (stdin works too)
```
Flags: `-n` list length, `-k` words per chunk, `--chunks`, `--json`, `--full`, `--stop`, `--no-daemon`.

## Use it as an AI skill
`skills/related-words/` is a ready agent skill (`SKILL.md` + `run.py` + 41 MB `resources/`). Copy it into your agent's skills directory (Claude Code: `~/.claude/skills/`).

## Docs
- [dataset.md](dataset.md) - sources, how the sense table, index and eval sets were built, sizes, licences
- [approach.md](approach.md) - architecture, why Gemma-256, everything tried and abandoned, scores
- [packaging.md](packaging.md) - pruning 104 MB -> 41 MB, speed benchmarks, the skill
- [docs/experiment-log.md](docs/experiment-log.md), [docs/knowledge-notes.md](docs/knowledge-notes.md) - raw working notes
- `scripts/` - `data/` (sense table), `index/` (index build, pruning, comparison, CLI), `eval/` (eval sets, harness, baselines), `student/` (abandoned static-student training), `speed/` (benchmarks)

## Results (Gemma-256)
Definition -> word (WordNet, n=1920): R@1 0.28, R@10 0.67, MRR 0.411. Hand-written situations (n=64): MRR 0.149. Results are meaning matches; use zipf to prefer rarer words.

## Licence and attribution
Code: MIT. `skills/related-words/resources/` is derived from English Wiktionary via [kaikki.org](https://kaikki.org) (CC BY-SA 4.0) and embeddings from Gemma (subject to the [Gemma terms](https://ai.google.dev/gemma/terms)). The model weights are not included. Word frequencies: `wordfreq`.
