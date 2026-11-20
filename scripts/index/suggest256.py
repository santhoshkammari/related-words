"""Approach 1: Gemma-256 (MRL) passage -> chunks -> word words.
Usage: python3 -I src/suggest256.py --build            # make emb256.npy from emb.npy
       python3 -I src/suggest256.py "passage..."       # or pipe text on stdin
Env: K (words per chunk, default 5), WSR_ROOT, GEMMA, DTYPE (fp32|bf16), MAXW (max words/chunk, 22), MINW (min, 5)
"""
import os, re, sys
import numpy as np, pandas as pd, torch

ROOT = os.environ.get("WSR_ROOT", ".")
IDX = f"{ROOT}/data/index"
DIM = 256

def build():
    e = np.load(f"{IDX}/emb.npy").astype(np.float32)[:, :DIM]
    e /= np.linalg.norm(e, axis=1, keepdims=True)          # MRL: truncate, then renormalize
    np.save(f"{IDX}/emb256.npy", e.astype(np.float16))
    print("saved", e.shape)

def chunk(text, maxw=22, minw=5):
    """Sentences first; split long ones at , ; : and 'and/but/because...'; merge tiny pieces."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    out = []
    for s in sents:
        if len(s.split()) <= maxw:
            out.append(s); continue
        parts = re.split(r"(?<=[,;:])\s+|\s+(?=(?:and|but|because|although|while|which|so|yet|then)\b)", s)
        cur = ""
        for p in parts:
            if not p: continue
            cand = (cur + " " + p).strip()
            if cur and len(cand.split()) > maxw:
                out.append(cur); cur = p
            else:
                cur = cand
        if cur: out.append(cur)
    merged = []
    for c in out:                                           # merge fragments that are too short
        if merged and len(c.split()) < minw: merged[-1] += " " + c
        elif merged and len(merged[-1].split()) < minw: merged[-1] += " " + c
        else: merged.append(c)
    return merged

def main():
    if "--build" in sys.argv: return build()
    text = " ".join(a for a in sys.argv[1:]) or sys.stdin.read()
    K = int(os.environ.get("K", 5))
    torch.set_num_threads(int(os.environ.get("THREADS", max(1, (os.cpu_count() or 8) // 2))))   # 4 threads was fastest on 4-core CPU
    from sentence_transformers import SentenceTransformer
    gp = os.environ.get("GEMMA", f"{ROOT}/models/embeddinggemma-300m")
    m = SentenceTransformer(gp, device="cpu")
    if os.environ.get("DTYPE") == "bf16": m.to(torch.bfloat16)
    E = torch.from_numpy(np.load(f"{IDX}/emb256.npy").astype(np.float32))
    meta = pd.read_parquet(f"{IDX}/meta.parquet")
    chunks = chunk(text, int(os.environ.get("MAXW", 22)), int(os.environ.get("MINW", 5)))
    Q = m.encode(chunks, prompt_name="query", convert_to_tensor=True).float()[:, :DIM]
    Q = torch.nn.functional.normalize(Q, dim=1)
    S = Q @ E.T
    best = {}
    for i, c in enumerate(chunks):
        print(f"\n[{i+1}] {c}")
        top = S[i].topk(300).indices.tolist(); seen = set(); n = 0
        for j in top:
            r = meta.iloc[j]
            if r.lemma in seen: continue
            seen.add(r.lemma); n += 1
            sc = float(S[i, j])
            print(f"   {sc:.3f}  {r.lemma} ({r.pos}, z{r.zipf:.1f})  {str(r.gloss_leaf)[:80]}")
            best[r.lemma] = max(best.get(r.lemma, 0), sc)
            if n >= K: break
    print("\n== passage words (best score across chunks) ==")
    print(", ".join(f"{w}" for w, _ in sorted(best.items(), key=lambda x: -x[1])))

main()
