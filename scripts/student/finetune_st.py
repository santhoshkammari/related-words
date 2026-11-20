"""Contrastive fine-tune: masked example sentence -> 'lemma (pos): gloss' (train-split lemmas only).
usage: python -I finetune_st.py BASE OUT [--epochs 3 --bs 128 --lr 5e-5 --qprefix '...' --quotes]"""
import re, argparse, numpy as np, pandas as pd, torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("out"); ap.add_argument("--root", default=".")
ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--qprefix", default="")
ap.add_argument("--quotes", action="store_true"); ap.add_argument("--max_pairs", type=int, default=0)
a = ap.parse_args()

df = pd.read_parquet(f"{a.root}/data/senses.parquet")
df = df[(df.split == "train") & (df.zipf >= 1.5) & (df.zipf <= 5.0)]
rows = []
for r in df.itertuples():
    exs = list(r.examples) + (list(r.quotes) if a.quotes else [])
    doc = f"{r.lemma} ({r.pos}): {r.gloss_leaf}"
    stem = re.escape(r.lemma[:max(3, len(r.lemma) - 2)])
    for e in exs:
        q = re.sub(r"\b" + stem + r"\w*", "___", e, flags=re.I)
        if "___" in q: rows.append((a.qprefix + q, doc))
rng = np.random.default_rng(0); rng.shuffle(rows)
if a.max_pairs: rows = rows[:a.max_pairs]
print("training pairs:", len(rows), "| distinct docs:", len({d for _, d in rows}), "| lemmas:", df.lemma.nunique())
ds = Dataset.from_dict({"anchor": [x for x, _ in rows], "positive": [y for _, y in rows]})

model = SentenceTransformer(a.base, device="cuda"); model.max_seq_length = 96
args = SentenceTransformerTrainingArguments(output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
    learning_rate=a.lr, warmup_ratio=0.1, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=50,
    save_strategy="no", report_to="none", seed=0, dataloader_num_workers=2)
tr = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model))
tr.train()
model.save_pretrained(a.out); print("saved", a.out)
