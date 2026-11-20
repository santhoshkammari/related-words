# dataset.md - what data we built, from where, and what it achieved

## 1. Source
| Item | Value |
|---|---|
| Dictionary | English Wiktionary, via the kaikki.org "English" JSONL extract (https://kaikki.org/dictionary/English/) |
| Raw file | `data/raw/kaikki-English.jsonl`, 3,431,211,190 bytes (3.43 GB), 1,499,161 entries |
| Frequency | `wordfreq` library, `zipf_frequency(word, "en")` (0 = never seen ... ~8 = "the") |
| WordNet (eval only) | NLTK WordNet, stored under `data/nltk` |
| Web text (student training only) | `allenai/c4` (config `en`) and `sentence-transformers/wikipedia-en-sentences` on Hugging Face |
| Teacher model | `google/embeddinggemma-300m` (gated on Hugging Face; accept the terms, never commit your token) |

Working locations (project, not shipped in this repo): training box `gpu4:/disk2/<user>/word-sense-retrieval` (git repo, `source activate.sh`) and a laptop copy `~/word-sense-retrieval` (model in `models/`, index in `data/index/`). Large files (raw dump, `senses.parquet`, 1.2 GB model, 768-d index) stay out of this repo; every one can be rebuilt with the scripts below.

## 2. Step 1 - sense table (`scripts/data/build_senses.py`)
`python -I build_senses.py data/raw/kaikki-English.jsonl data/senses.parquet`
- 1,499,161 entries read -> **567,305 senses / 399,633 lemmas** in `data/senses.parquet`.
- Kept: single lowercase tokens (`^[a-z][a-z\-]*$`), nouns/verbs/adjectives/adverbs.
- Dropped: tags obsolete, archaic, dated, vulgar, offensive, derogatory, slur, dialectal, misspelling, nonstandard, form-of, alt-of, alternative, abbreviation, initialism, acronym, pejorative; "X form of Y" glosses; glosses under 8 characters.
- Up to 5 example sentences per sense kept (only 20-29% of senses in the mid-frequency band have one).
- Train/val/test split by **lemma hash** (md5 % 100: <5 test, <10 val, else train) so no word leaks between splits: 359,543 / 20,242 / 19,848 senses.
- Columns used later: `lemma, pos, gloss_leaf, zipf, split`.

## 3. Step 2 - word pool and index (`scripts/index/build_index.py`, `suggest256.py --build`)
- Pool = **zipf 1.5 to 5.0**: **67,913 words, 179,897 senses** (about 330k lemmas are below 1.5 and are mostly technical junk such as "nardilysin"; above 5.0 are function words).
- Card text per sense: `lemma (pos): gloss_leaf`, embedded with Gemma's *document* prompt `title: none | text: `.
- Built in bf16 on GPU (fp16 activations overflow to NaN in Gemma - the first attempt was all NaN). Stored fp16: `emb.npy` 179,897 x 768 (276 MB), `meta.parquet` (sense_id, lemma, pos, gloss_leaf, zipf).
- 256-d version: cut each vector to its first 256 numbers (Matryoshka training allows this) and re-normalise -> `emb256.npy` 92 MB. Similarity correlation with 768-d: 0.935.
- Shipped version (pruned, int8) is described in [packaging.md](packaging.md): 144,908 senses / 67,160 words, 41 MB.

## 4. Evaluation sets (`scripts/eval/`)
| Set | What | Size | Built by |
|---|---|---|---|
| A | hand-written situations -> target words (acceptable synonyms that appear in the text were removed) | 64 | `eval/situations.jsonl`, `make_situations.py` |
| B | masked held-out dictionary example sentences -> the word | 1,668 | `eval_retrieval.py` |
| C | WordNet synset definitions -> the synset's lemmas that are in our pool and test split (leaky ones removed) | 1,920 | `make_wn_eval.py` |
Metrics: recall@k and MRR over lemmas (a word's score = best of its senses).

## 5. Training data for the student experiments (abandoned, see approach.md)
- v1 `data/student/train_gemma256`: 547k texts (249k dictionary entries/examples/masked examples from the train split + 150k C4 paragraphs + 150k Wikipedia sentences), targets = Gemma-256 embeddings.
- v2 `train_big`: 1.65M texts (700k C4 + 700k Wikipedia), 32 min to encode.
- Sage-A1.5 synthetic situations (`gen_situations.py`): only 1,649 of 16,000 rows were produced; **not used** in the final system.

## 6. What the data achieved
Gemma-256 over this index: situations A MRR 0.149; definitions C R@1 0.28, R@10 0.67, MRR 0.411 (Gemma-768: A 0.189, C 0.437). B is near floor for every model.

## 7. Known flaws
Near-duplicate word forms (tire/tired/overtired), some weak or context-dependent glosses ("The ball used in this game."), a few explicit senses. Results are meaning matches; use zipf to prefer rarer words.

## 8. Licence of the data
Wiktionary text is CC BY-SA 4.0 / GFDL, so the shipped index metadata (glosses) inherits CC BY-SA. Embeddings are derived with Gemma and fall under the Gemma terms of use.
