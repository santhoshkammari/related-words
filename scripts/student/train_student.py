"""Distill Gemma -> ~8M-param potion student (Model2Vec distill + prune vocab to N rows + Tokenlearn training).
usage: python -I train_student.py DATA OUT [--rows 30000]"""
import argparse, numpy as np, torch, logging
from collections import Counter
from datasets import load_from_disk
from model2vec.distill import distill
from model2vec import StaticModel
from model2vec.train import StaticModelForSimilarity
logging.basicConfig(level=logging.INFO)
ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--rows", type=int, default=30000); ap.add_argument("--teacher", default="google/embeddinggemma-300m")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=256)
a = ap.parse_args()
ds = load_from_disk(a.data); texts = ds["text"]; y = np.array(ds["embedding"], dtype=np.float32)
# 1) distill: every Gemma token through Gemma alone, PCA->256, SIF weights
m = distill(model_name=a.teacher, pca_dims=256, quantize_to="float32", device="cuda", trust_remote_code=False)
n_tok = len(m.tokens) if hasattr(m, "tokens") else len(m.embedding)
print("distilled table", m.embedding.shape, flush=True)
# 2) prune: keep the a.rows-1 most used tokens, everything else shares one extra row
cnt = Counter()
for i in range(0, len(texts), 20000):
    for ids in m.tokenize(texts[i:i + 20000]): cnt.update(ids)
keep = [t for t, _ in cnt.most_common(a.rows - 1)]
mapping = np.full(len(m.embedding), a.rows - 1, dtype=np.int64)
mapping[keep] = np.arange(len(keep))
rest = np.ones(len(m.embedding), bool); rest[keep] = False
vec = np.concatenate([m.embedding[keep], m.embedding[rest].mean(0, keepdims=True)]).astype(np.float32)
print(f"kept {len(keep):,} used tokens of {len(cnt):,} seen / {len(m.embedding):,}; table {vec.shape} = {vec.size/1e6:.1f}M params", flush=True)
small = StaticModel(vectors=vec, tokenizer=m.tokenizer, config=m.config, token_mapping=mapping, normalize=True)
# 3) Tokenlearn training: match Gemma-256 sentence vectors (cosine loss), held-out 10% for early stopping
tr = StaticModelForSimilarity.from_static_model(model=small, out_dim=256, n_layers=0, freeze_weights=True)
tr.fit(X=texts, y=torch.from_numpy(y), batch_size=a.bs, device="cuda", learning_rate=a.lr)
tr.to_static_model().save_pretrained(a.out); print("saved", a.out, flush=True)
