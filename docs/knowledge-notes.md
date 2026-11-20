# Word-sense retrieval: what we know, what we built, what's next

Goal: give a sentence or passage, get back the English words that relate to it in meaning
("She stayed calm though the meeting turned hostile" -> even-tempered, unflappable, imperturbable).

Where things live
- gpu4 (main repo, git, training): `/disk2/<user>/word-sense-retrieval` (run `source activate.sh`)
- this laptop (demo copy, not a git repo): `~/word-sense-retrieval` (model in `models/`, index in `data/index/`)
- Run the demo: `K=8 WSR_ROOT=. HF_HUB_OFFLINE=1 python3 -I src/suggest.py "your text"`

## 1. The idea in plain words
Every dictionary sense becomes a short card: `lemma (pos): definition`.
A model turns each card into a list of numbers (an "embedding") where similar meanings land close together.
Your sentence is turned into numbers the same way; we return the cards that are closest.
A word with several senses is scored by its best sense.
No sentence-to-word classifier is trained; it is search over meanings.

## 2. The two models
| | embeddinggemma-300m ("Gemma") | potion-base-8M ("potion") |
|---|---|---|
| What it is | a real neural network, 300M weights | one big table, a row per word piece |
| How it reads text | many layers think about the whole sentence | look up each piece's row, average them |
| Needs | torch | only NumPy |
| Speed, 1 query on CPU | 25 ms (server, 8 threads), ~123 ms (laptop) | 0.06 ms |
| Speed, 1000 texts | 3.5 s | 3.5 ms (about 1000x faster) |
| Understands situations | well | weakly (see results) |
| License | gated, needs accepted terms on Hugging Face | open |
Plan: Gemma = quality / reranker. A potion-style student distilled from Gemma = fast first pass.
Other small encoders (MiniLM, bge-small, gte-small, e5-small, arctic-xs, granite-small) were only baselines.
Sage-A1.5 (a 35B chat model on vLLM) is used only to write synthetic training/eval situations, not in the final system.

## 3. The data
- Source: kaikki.org English Wiktionary dump -> `data/senses.parquet`: 399,633 lemmas, 567,305 senses.
- Kept: single lowercase words; noun/verb/adj/adv; dropped inflected forms, offensive/obscure tags, glosses under 8 chars.
- Examples: up to 5 dictionary example sentences per sense (literary quotes kept separately).
- Splits are by lemma (train 359,543 / val 20,242 / test 19,848 senses) so no word leaks between train and test.
- **zipf**: a "how common is this word" score from the `wordfreq` library, ~0 (never seen) to ~8 ("the").
  4-5 = everyday (tired, crowded); 2-3 = less common (procrastinate, unflappable); under 1.5 = rare/technical junk (nardilysin).
- Word pool = zipf 1.5 to 5.0: **67,913 words, 179,897 senses** (about 330k of the 400k lemmas are under 1.5 and excluded).
- Known flaws: near-duplicate forms (tire / tired / overtired), a few explicit senses, some weak glosses. Needs a cleanup pass.

## 4. The index (what the demo searches)
- `data/index/emb.npy`: 179,897 x 768, fp16, NumPy file (276 MB). `meta.parquet` holds lemma/pos/gloss/zipf.
- Built by `src/build_index.py` in bf16 with a NaN check. Query side: `src/suggest.py`.
- Shrink path to ~100 MB: Gemma allows cutting to 256 numbers (similarity correlation 0.935 vs full), drop near-duplicates and
  explicit senses, store int8 (256-d int8 for 180k senses is ~46 MB). Measure quality loss at each step before accepting.

## 5. Number formats (fp32 / fp16 / bf16)
- Gemma's internal activations overflow in fp16. Use bf16 or fp32. (My first index was built in fp16 and came out all NaN.)
- Storing the finished embeddings is different: they are all in [-1, 1], so fp16 storage is safe and slightly more precise than bf16.
- Search speed for one query over the index (laptop i5-1135G7, no hardware bf16):
  NumPy fp32 26 ms | NumPy bf16 210 ms | NumPy fp16 614 ms | torch fp32 25 ms | torch bf16 13 ms | torch fp16 13.6 ms.
  Top-20 results identical in every case. NumPy has no real half-precision math, so use fp32 there; torch is fine with bf16/fp16.
- Gemma model on this CPU: fp32 123 ms, bf16 142 ms (no speedup without bf16 hardware; a CPU with AMX/avx512_bf16 is untested), same embedding.
- Potion is plain NumPy fp32; none of this applies to it.

## 6. How a potion-style student is made from Gemma (Tokenlearn)
1. Collect many texts (definitions, examples, web paragraphs, later Sage situations).
2. Gemma embeds each text = the answer key (cut to 256-d).
3. Start a table with one row per word, nudge rows until the average of a text's rows matches Gemma's embedding.
4. Drop Gemma, keep the table (NumPy only).
- Tokenlearn 2.0 (checked in the official post): PCA is still used *before* training (shrinks the answer key and the starting table;
  "if not done, distillation does not work at all"). The old *after-training* PCA + SIF reweighting step was dropped. Our repo copy
  (third_party/tokenlearn on gpu4) follows 2.0: flags `--pca-dims`, `--target-pca-dims`, saves one model. Its README still mentions two models; unresolved.
- The stock featurizer uses mean token embeddings as the target. For Gemma that has cosine ~0 with the real embedding
  (measured: -0.04..-0.01), so we use our own featurizer (`src/featurize_teacher.py`) with real `encode()` outputs.
- The student lives in its own 256-d space, NOT Gemma's. It needs its own index, or acts as a first filter with Gemma re-ranking the top ~100.
- Pipeline validated end to end with bge-small as teacher (367k texts in 37 s). Not yet run with Gemma.

## 7. Evaluation (`src/eval_retrieval.py`)
- A: 64 hand-written situations (small; differences are only suggestive). B: situations built from masked dictionary examples (near floor for untrained models).
  C: WordNet definitions as queries (a definition-matching test). Metrics: recall@k and MRR over lemmas.
- Gemma sanity (gpu4): pipeline Transformer+Pooling+Dense+Dense+Normalize; query prompt `task: search result | query: `,
  document prompt `title: none | text: `; vocab 262,145; context 2048; 1,643 sent/s on GPU.

## 8. Results so far
| Model | A R@5 | A MRR | C MRR |
|---|---|---|---|
| Gemma 768-d, zero-shot (query/document prompts) | 0.30 | 0.189 | 0.437 |
| best earlier small encoder (bge-small-class) | 0.19 | 0.088 | 0.252 |
| bge-small fine-tuned on dictionary examples | 0.08 | - | 0.355 |
| static student distilled from bge-small | - | 0.041 | beat/matched teacher |
- Fine-tuning on dictionary examples helps definition matching but hurts situations; Sage-written situations are the planned fix.
- Gemma at 256-d: the run did not finish, so there is no number. potion-8M/32M on this exact benchmark: not yet run.
- Those A/C values for non-Gemma rows come from earlier logs (`plan/RESULTS_all.md`); recheck there before quoting.

## 9. Sage synthetic situations
- `src/gen_situations.py`: for each sense, Sage writes k realistic situations in varied registers, never using the word.
- Fixes found: ~30% of replies were JSON objects not lists, and some nested a list under a key; the parser now flattens any shape (52/60 rows full).
  `--max_rank 2` skips obscure senses. Leaks (word or inflection in the text) are filtered (~10%).
- Status: stopped at 1,649 of 16,000 rows (14,000 train senses x5, 2,000 test senses x2). Resumable. ~2.4 req/s when the GPU is not shared.
- Test-split situations become a much bigger Eval D (~4,000 items instead of 64).

## 10. Next steps (in order)
1. Baseline table on the same benchmark: Gemma 768 vs 256, potion-8M, potion-32M.
2. Resume Sage generation; build Eval D.
3. Distill the Gemma -> potion student (with Sage situations in the corpus); compare to stock potion and to Gemma.
4. Fine-tune Gemma with train-split situations; check A/D do not regress.
5. Clean the index (dedupe, content filter, 256-d int8) and re-measure.
6. Commit the new files on gpu4 (build_index, suggest, eval changes); nothing is pushed to any remote without your go-ahead.

## 11. Gotchas
- Never run ksage/kvllm/sagekill (they kill every vLLM engine). Sage holds ~90 GB of the GPU; only ~5 GB is free, so use small batches.
- Never write the HF token to files. Use it as an env var only (and rotate the one pasted in chat).
- `pkill -f` patterns can match other jobs (it killed 3 unrelated PIDs once). Kill by exact PID.
- Datasets streaming can crash at exit (end with `os._exit(0)`); nltk needs `python -P` from the project dir.

## 12. How any model becomes a potion model (read from source, 2026-11-21)
**Correction to section 6:** `distill()` still applies PCA and Zipf/SIF weighting to the *starting* table. Tokenlearn 2.0 only dropped the extra PCA+SIF step *after* training.
Code read: model2vec 253d598 (`distill/distillation.py`, `distill/inference.py`, `train/similarity.py`, `train/base.py`, `model.py`), tokenlearn e0c3d28 (`train.py`, `utils.py`).

**Step 1 - distill (no training, minutes):**
1. Take the teacher's tokenizer vocab (plus optional extra words, e.g. top 56k words from your texts).
2. Run every token alone through the teacher as a 1-token sequence, mean-pool the output -> one vector per token.
3. PCA down to 256 dims. 4. Multiply each vector by a SIF weight `s/(s+p)`, where p comes from an assumed Zipf curve over token *rank* (not real counts), s=1e-4.
5. Store as float16/32 table. Config says normalize=True.
**Encoding a sentence:** tokenize -> look up rows -> (weights) -> mean -> L2 normalize. No attention, so very fast.

**Step 2 - Tokenlearn (training, makes it much better):**
- Targets: teacher's real sentence embeddings for many texts (C4 for stock potion). Optionally PCA the targets (`--target-pca-dims`).
- Student = `StaticModelForSimilarity` (n_layers=0, freeze_weights=True): an EmbeddingBag over the table; if out_dim != table dim a single Linear is added.
- Loss in current code is **cosine distance** (`CosineLoss` in `train/similarity.py`), not MSE as the blog says. Early stopping on a 10% validation split (<=10k rows), lr 1e-3, Adam-style loop.
- After training, `to_static_model()` exports table (+weights) back to a plain numpy StaticModel.
- Because training only matches *direction*, norms don't matter (output is normalized).
- Open: README says two models; code saves one.

**For us (Gemma):** the student is its own 256-d space, not Gemma's. Targets must be real Gemma embeddings from our featurizer (stock featurizer averages raw token vectors, wrong for Gemma). Use the same query/document prompts as at retrieval.
