import os, sys, time, re, numpy as np, pandas as pd, torch
sys.argv = ["x", "--none"]; ROOT = "."; os.environ["WSR_ROOT"] = ROOT
src = open("src/suggest256.py").read().replace("\nmain()\n", "\n")
exec(src.split("def main")[0])                      # reuse chunk(), constants
from sentence_transformers import SentenceTransformer
t = time.time(); m = SentenceTransformer(f"{ROOT}/models/embeddinggemma-300m", device="cpu"); t_load = time.time() - t
t = time.time(); E = torch.from_numpy(np.load(f"{IDX}/emb256.npy").astype(np.float32)); meta = pd.read_parquet(f"{IDX}/meta.parquet"); t_idx = time.time() - t
lem = meta.lemma.values
base = ("The committee's ambiguous remarks sparked a protracted debate, and critics argued that the proposal was both "
        "unrealistic and detrimental to the fragile economy. Meanwhile, residents lamented the meticulous bureaucracy that "
        "stifled innovation. Nevertheless, a few pragmatic voices urged restraint and compromise.")
def run(text, K=5):
    ts = {}; t0 = time.perf_counter()
    ch = chunk(text); ts["chunk"] = time.perf_counter() - t0; t0 = time.perf_counter()
    Q = m.encode(ch, prompt_name="query", convert_to_tensor=True).float()[:, :DIM]; Q = torch.nn.functional.normalize(Q, dim=1); ts["encode"] = time.perf_counter() - t0
    t0 = time.perf_counter(); S = Q @ E.T; top = S.topk(300, dim=1).indices.tolist(); ts["search"] = time.perf_counter() - t0; t0 = time.perf_counter()
    for i in range(len(ch)):
        seen = []; 
        for j in top[i]:
            if lem[j] not in seen: seen.append(lem[j])
            if len(seen) >= K: break
    ts["dedupe"] = time.perf_counter() - t0; return len(ch), ts
print(f"cpu threads {torch.get_num_threads()} | model load {t_load:.1f}s | index+meta load {t_idx:.1f}s | docs {len(E):,}")
for name, rep in [("1 sentence", 0), ("~45 words", 1), ("~140 words", 3), ("~420 words", 9)]:
    text = base if rep == 0 else " ".join([base] * rep)
    if rep == 0: text = base.split(".")[0] + "."
    run(text)                                       # warm-up
    r = [run(text) for _ in range(5)]; nch = r[0][0]
    tot = [sum(x[1].values()) for x in r]
    avg = {k: np.mean([x[1][k] for x in r]) * 1000 for k in r[0][1]}
    print(f"{name:12s} words {len(text.split()):4d} chunks {nch:3d} | total {np.mean(tot)*1000:7.0f} ms (min {min(tot)*1000:.0f}) | " + " ".join(f"{k} {v:.1f}ms" for k, v in avg.items()) + f" | {np.mean(tot)*1000/nch:.0f} ms/chunk")
