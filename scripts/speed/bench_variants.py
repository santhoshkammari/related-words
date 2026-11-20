import os, sys, time, copy, numpy as np, pandas as pd, torch
sys.argv = ["x"]; os.environ["WSR_ROOT"] = "."
exec(open("src/suggest256.py").read().split("def main")[0])
from sentence_transformers import SentenceTransformer
E = torch.from_numpy(np.load(f"{IDX}/emb256.npy").astype(np.float32)); lem = pd.read_parquet(f"{IDX}/meta.parquet").lemma.values
base = ("The committee's ambiguous remarks sparked a protracted debate, and critics argued that the proposal was both "
        "unrealistic and detrimental to the fragile economy. Meanwhile, residents lamented the meticulous bureaucracy that "
        "stifled innovation. Nevertheless, a few pragmatic voices urged restraint and compromise.")
chunks = chunk(" ".join([base] * 3)); print(len(chunks), "chunks")
def enc(m, bs=32):
    Q = m.encode(chunks, prompt_name="query", convert_to_tensor=True, batch_size=bs).float()[:, :DIM]
    return torch.nn.functional.normalize(Q, dim=1)
def words(Q):
    top = (Q @ E.T).topk(300, dim=1).indices.tolist(); out = []
    for t in top:
        s = []
        for j in t:
            if lem[j] not in s: s.append(lem[j])
            if len(s) >= 5: break
        out.append(set(s))
    return out
def timeit(f, n=5):
    f(); ts = []
    for _ in range(n): t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts) * 1000, np.mean(ts) * 1000
gp = "models/embeddinggemma-300m"; ref = None
def report(name, m, bs=32, thr=None):
    global ref
    if thr: torch.set_num_threads(thr)
    mn, avg = timeit(lambda: enc(m, bs)); Q = enc(m, bs); W = words(Q)
    if ref is None: ref = (Q, W)
    cos = (Q * ref[0]).sum(1).mean().item(); ov = np.mean([len(a & b) / 5 for a, b in zip(W, ref[1])])
    print(f"{name:34s} enc {mn:6.0f} ms (avg {avg:.0f}) | cos vs fp32 {cos:.4f} | top5 overlap {ov:.2f}", flush=True)
m = SentenceTransformer(gp, device="cpu"); m.eval()
for thr in (8, 4, 6, 2):
    report(f"fp32 threads={thr}", m, thr=thr)
ref = None; torch.set_num_threads(8); report("fp32 t8 (ref)", m)
for bs in (8, 16, 64): report(f"fp32 t8 bs={bs}", m, bs=bs)
m.max_seq_length = 64; report("fp32 t8 maxlen=64", m); m.max_seq_length = 2048
mb = copy.deepcopy(m).to(torch.bfloat16); report("bf16 t8", mb)
mq = torch.quantization.quantize_dynamic(copy.deepcopy(m), {torch.nn.Linear}, dtype=torch.qint8); report("int8 dynamic t8", mq)
mbq = torch.quantization.quantize_dynamic(copy.deepcopy(m).float(), {torch.nn.Linear}, dtype=torch.qint8); mbq.max_seq_length = 64; report("int8 dynamic t8 maxlen=64", mbq)
