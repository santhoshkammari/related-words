"""Shrink the 256-d Gemma sense index: drop near-duplicate / excess / place-name senses, int8-quantise, slim the metadata.
usage: python -I prune_index.py IN_DIR OUT_DIR      (IN_DIR has emb256.npy fp16 + meta.parquet; writes emb256_i8.npy, emb256_scale.npy, meta_small.parquet)"""
import sys, os, numpy as np, pandas as pd, torch
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
DUP_COS, MAX_SENSES, GLOSS_MAX = 0.92, 8, 220
m = pd.read_parquet(f"{src}/meta.parquet"); g = m.gloss_leaf.astype(str)
E = np.load(f"{src}/emb256.npy").astype(np.float32); Et = torch.from_numpy(E)
dup = np.zeros(len(m), bool)                                  # sense too similar to an earlier sense of the same word
for _, idx in m.groupby("lemma").indices.items():
    for a in range(1, len(idx)):
        if (Et[idx[:a]] @ Et[idx[a]] > DUP_COS).any(): dup[idx[a]] = True
drop = {"near-dup": dup,
        "place/taxon/surname": g.str.contains(r"\b(?:species of|genus|surname|given name|a town|a village|a city|a river|a county|a district|a parish)\b", case=False).values,
        f"sense #{MAX_SENSES + 1}+": (m.groupby("lemma").cumcount() >= MAX_SENSES).values}
keep = np.ones(len(m), bool)
for k, v in drop.items(): n = int((keep & v).sum()); keep &= ~v; print(f"drop {n:6d}  {k}")
k = m[keep].reset_index(drop=True); Ek = E[keep]
sc = (np.abs(Ek).max(1) / 127).astype(np.float32)             # per-row symmetric int8
np.save(f"{dst}/emb256_i8.npy", np.round(Ek / sc[:, None]).astype(np.int8)); np.save(f"{dst}/emb256_scale.npy", sc.astype(np.float16))
o = k[["lemma", "pos", "zipf", "gloss_leaf"]].copy(); o["zipf"] = o.zipf.round(2).astype("float16"); o["pos"] = o.pos.astype("category")
o["gloss_leaf"] = o.gloss_leaf.astype(str).str.slice(0, GLOSS_MAX)
o.to_parquet(f"{dst}/meta_small.parquet", compression="zstd", compression_level=19, index=False)
print("kept", len(k), "senses,", k.lemma.nunique(), "lemmas;", {f: round(os.path.getsize(f"{dst}/{f}") / 1e6, 1) for f in os.listdir(dst)}, "MB")
