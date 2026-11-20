# Student (potion-style static model) experiment log
Goal: static student reaching >=98% of Gemma-256 retrieval accuracy (no Sage data, no 2-stage rerank).
Metric: Eval A (situations, n=64, noisy), B (n=1668, very hard), C (definitions, n=1920). Gemma-256 targets: A MRR 0.149, C MRR 0.411 -> 98% = A 0.146, C 0.403.
Results live in gpu4 `runs/eval/<tag>.json`; logs in `runs/logs/`. Code: `src/featurize_gemma.py`, `src/train_student.py`, `src/train_rank.py`.

## Baselines
| model | A MRR | B MRR | C R@1 | C R@10 | C MRR | %Gemma256 (C) |
|---|---|---|---|---|---|---|
| Gemma 768 | 0.189 | 0.0209 | 0.302 | 0.702 | 0.437 | 106% |
| Gemma 256 (target) | 0.149 | 0.018 | 0.28 | 0.67 | 0.411 | 100% |
| potion-base-8M | 0.045 | 0.0077 | 0.186 | 0.439 | 0.269 | 65% |
| potion-base-32M | ~0.121(768-era table) | 0.0092 | 0.224 | 0.492 | n/a | n/a |

## Experiments
Data v1 (`data/student/train_gemma256`): 547k texts = 249k dict entries/examples/masked examples (train split only) + 150k C4 + 150k Wikipedia sentences; Gemma-256 targets (doc prompt for dict entries, query prompt otherwise).
Data v2 (`data/student/train_big`): 1.65M texts (same recipe, 700k C4 + 700k wiki). Encoding took 32 min.

| tag | what | val cos-loss | A MRR | C MRR | %G256 |
|---|---|---|---|---|---|
| student_a | distill Gemma (255k tok, PCA256) pruned to 30k rows (7.7M params), Tokenlearn fit on v1, 30 ep | 0.189 | 0.035 | 0.247 | 60% |
| student_b | same but all 111k seen tokens kept (28M params), 37 ep | 0.164 | 0.053 | 0.276 | 67% |
| student_c | warm start from b, v2 data (3x), lr 5e-4, 10 ep | 0.1625 | 0.059 | 0.263 | 64% |
| student_d | `train_rank.py` from student_b on v2: cosine + in-batch similarity KL (tau .05, lam 1), bs 2048, lr 3e-4, 6 ep | 0.1757 (different val set) | 0.053 | 0.279 | 68% |

## Findings
- Pruning to 30k rows costs ~0.03 C MRR (a vs b). Not the main limit.
- 3x more data barely moves val loss (0.164 -> 0.1625) and C MRR did not improve. Data volume is not the bottleneck.
- Plateau at cosine ~0.84 to Gemma. Likely cause: bag-of-tokens ignores word order/context; Gemma is contextual.
- Cosine-to-Gemma is not the retrieval metric.

## Ideas / queue
1. (DONE: student_d, +0.003 C MRR over b, i.e. no real gain) `train_rank.py`: cosine + in-batch similarity-structure KL (tau 0.05) so ranking among texts matches Gemma's. Fine-tune from student_b -> student_d.
2. Contrastive pairs matched to eval: (masked/example sentence -> its dict entry) positives with in-batch negatives, hard negatives from same-lemma/other-sense.
3. Bigram/word-pair word rows added to the table (partial order info).
4. Per-token learnable weights / a tiny extra layer (still lookup-ish) e.g. n_layers>0 in StaticModelForSimilarity.
5. Check ceiling: how well does a bag-of-Gemma-token-vectors from Gemma itself (mean of per-token teacher outputs) do? (already ~ student init level).
6. If 1-4 stall <80%: 98% needs a contextual model (small transformer); outside potion-style scope.

## Update after student_d
- Similarity-structure loss gave 0.279 vs 0.276: within noise. Three different training recipes/data sizes all land at C MRR 0.26-0.28 (~65-68% of Gemma-256). Strong evidence the limit is the model form (token-wise lookup + mean), not data or loss.
- Remaining untested: ideas 2-4 above (hard-negative contrastive, bigram rows, small extra layers).

## Experiment e: hashed-bigram rows (2026-11-21)
- Model: student_b unigram table (trainable) + 200,003 hashed bigram buckets x256, summed with the unigram mean, then L2-normalised. Loss = cosine + similarity-KL. 6 epochs on train_big (1.65M texts). Code: src/bgmodel.py, src/train_bigram.py, eval kind `bg:`.
- Val cosine loss 0.1669 (unigram students: ~0.1625). Retrieval: A MRR 0.057, B MRR 0.010, C R@1 0.19, R@10 0.46, C MRR 0.281 (68% of Gemma-256).
- Verdict: no gain over student_d (0.279). Adding partial word-order information to a static model does not close the gap.
- Cumulative: 5 recipes (distill+tokenlearn, 3.3x data, cosine+KL loss, bigrams, larger vocab) all land at 60-68% of Gemma-256 on C. The 98% goal is not met.
- Remaining untried: hard-negative contrastive training matched to the eval, per-token weights, small trainable layers (n_layers>0). Beyond that, a contextual model is needed.

## Experiment f: distill + InfoNCE on sense pairs (2026-11-21)
- student_b init, unigram static, loss = cosine + sim-KL + InfoNCE (tau 0.05; queries = masked examples / bare gloss; docs = "lemma (pos): gloss"; in-batch negatives incl. up to 3 senses/lemma). 6 epochs. Code: src/train_contrast.py.
- Val cosine to Gemma worsens (0.1625 -> 0.1864): the two objectives compete.
- Retrieval: A MRR 0.046 (R@50 0.41, med rank 82; was 0.27 / 164), B MRR 0.012 (med 3174; was 4478), C MRR 0.283 (R@10 0.47), 69% of Gemma-256.
- Verdict: helps the broad/hard evals (A recall@50, B) a bit; C top-rank quality unchanged. Still ~69%. Goal (98%) not met.
- Pattern after 6 recipes: static bag-of-tokens form saturates at ~68-69% of Gemma-256 on C regardless of data, loss, vocab, bigrams or contrastive signal.

## Experiment g/h: small transformer student (2026-11-21) - STOPPED by user
- g: static table (student_b) + positions + 2 transformer layers, mean-pool + skip, 4 epochs, distill only: C MRR 0.274, A 0.029, B 0.009 (undertrained). h (4 layers, 24 epochs) was started and killed before finishing; no result.
- Decision (user): the 98% goal is not achievable with a potion-style static model. Best static student: student_f, C MRR 0.283 (69% of Gemma-256), ~0.0 ms/query.
- Open if revisited: a contextual student trained longer (g/h code: src/tfmodel.py, src/train_tf.py, eval kind `tf:`).
