# approach.md - architecture and what we tried

## Final architecture (default, "Approach 1")
**Search over meanings with one model: `google/embeddinggemma-300m` ("Gemma"), cut to 256-d.**
1. Every dictionary sense is a card `lemma (pos): gloss`, embedded once with the document prompt (index, built offline).
2. The user text is chunked (`chunk()`: sentences, long ones split at `, ; :` and and/but/because/although/while/which/so/yet/then, max 22 words, pieces under 5 words merged).
3. Each chunk is embedded with the query prompt `task: search result | query: `, cut to 256, L2-normalised.
4. `Q @ E.T`, top 300 senses, de-duplicate by lemma, top K=5 words per chunk; the passage list keeps each word's best score over chunks.
No classifier is trained; nothing is fine-tuned. Two-stage "cheap recall then Gemma rerank" was ruled out by the user.

Final configuration: fp32, `torch.set_num_threads(4)` (physical cores), 256-d, K=5, maxw=22, minw=5, int8 pruned index, warm background server (60 s idle).

## Why Gemma-256
| Model | A MRR | B MRR | C MRR |
|---|---|---|---|
| Gemma 768 | 0.189 | 0.0209 | 0.437 |
| **Gemma 256 (chosen)** | 0.149 | 0.018 | 0.411 |
| potion-base-8M (static) | 0.045 | 0.0077 | 0.269 |
| best small encoders (bge-small class) | ~0.09 | - | ~0.25 |
256-d keeps 94% of the C score at a third of the index size.

## What we tried and why it ended
1. **Zero-shot small encoders** (MiniLM, bge-small, gte-small, e5-small, arctic-xs, granite-small, potion 8M/32M): all weak on situations (best A R@5 0.20).
2. **Fine-tune bge-small on dictionary examples**: C MRR 0.355, but situations got worse (A R@5 0.08). Dropped.
3. **Static "potion" student distilled from Gemma** (Tokenlearn-style, goal: 98% of Gemma-256 with no neural net at query time). Six recipes, all plateau at 60-69% of Gemma-256 on C:
   | tag | recipe | C MRR | % of G256 |
   |---|---|---|---|
   | a | distill, pruned to 30k rows, Tokenlearn | 0.247 | 60% |
   | b | all 111k seen tokens | 0.276 | 67% |
   | c | + 3x data | 0.263 | 64% |
   | d | + similarity-KL loss | 0.279 | 68% |
   | e | + 200k hashed bigram rows | 0.281 | 68% |
   | f | + InfoNCE on sense pairs | 0.283 | 69% |
   | g | + 2 transformer layers (4 epochs) | 0.274 | 67% |
   More data, different losses, bigrams and contrastive signal did not move it: the limit is bag-of-tokens, which ignores context. Abandoned; details in [docs/experiment-log.md](docs/experiment-log.md).
4. **Sage-written synthetic situations** (35B chat model): generation stopped at 1,649/16,000 rows; not used.
5. **Speed variants** (see packaging.md): bf16 slower on this CPU; int8 dynamic query model changes 10-25% of results (rejected); ONNX export blocked (`onnxscript` missing, tracing exporter error `unordered_map::at`).

## Gotchas learned
- Gemma activations overflow in fp16 -> bf16/fp32 for the model, fp16 only for storing finished embeddings.
- Stock Tokenlearn featuriser averages raw token vectors, which has cosine ~0 with Gemma's real embedding; use real `encode()` outputs (`featurize_gemma.py`).
- The model is gated and is **not** in this repo.
