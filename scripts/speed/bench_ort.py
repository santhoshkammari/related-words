import os, sys, time, copy, numpy as np, pandas as pd, torch
sys.argv = ["x"]; os.environ["WSR_ROOT"] = "."
exec(open("scripts/bench_variants.py").read().split("m = SentenceTransformer")[0])
torch.set_num_threads(4)
m = SentenceTransformer(gp, device="cpu"); m.eval(); report("fp32 t4 (ref)", m, thr=4)
# ONNX Runtime fp32
try:
    import onnxruntime as ort
    class W(torch.nn.Module):
        def __init__(s, st): super().__init__(); s.st = st
        def forward(s, input_ids, attention_mask): return s.st({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]
    tok = m.tokenizer; pre = m.prompts["query"]
    b = tok([pre + c for c in chunks[:4]], padding=True, return_tensors="pt")
    torch.onnx.export(W(m), (b["input_ids"], b["attention_mask"]), "/tmp/g.onnx", input_names=["input_ids", "attention_mask"], output_names=["e"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "e": {0: "b"}}, opset_version=18, dynamo=True)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession("/tmp/g.onnx", so, providers=["CPUExecutionProvider"])
    def oenc(bs=32):
        outs = []; order = np.argsort([len(c) for c in chunks]); res = {}
        for i in range(0, len(chunks), bs):
            idx = order[i:i + bs]; t = tok([pre + chunks[j] for j in idx], padding=True, return_tensors="np")
            e = sess.run(None, {"input_ids": t["input_ids"].astype(np.int64), "attention_mask": t["attention_mask"].astype(np.int64)})[0]
            for j, v in zip(idx, e): res[j] = v
        Q = torch.from_numpy(np.stack([res[j] for j in range(len(chunks))]))[:, :DIM]; return torch.nn.functional.normalize(Q, dim=1)
    mn, avg = timeit(oenc); Q = oenc(); W_ = words(Q)
    print(f"{'onnxruntime fp32 t4':34s} enc {mn:6.0f} ms (avg {avg:.0f}) | cos vs fp32 {(Q*ref[0]).sum(1).mean().item():.4f} | top5 overlap {np.mean([len(a&b)/5 for a,b in zip(W_,ref[1])]):.2f}", flush=True)
except Exception as e: print("ORT failed:", repr(e)[:300])
