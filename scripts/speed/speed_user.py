import sys, time, torch
name = sys.argv[1]; threads = int(sys.argv[2]); torch.set_num_threads(threads)
sentence = "I love playing football"
if name.startswith("minishlab") or name.startswith("m2v:"):
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(name.replace("m2v:", "")); enc = lambda t: m.encode(t)
else:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cpu"); enc = lambda t: m.encode(t, batch_size=32, prompt_name="query")
print(name, "| cpu threads", threads, flush=True)
for n in [1, 8, 32, 128, 1000]:
    texts = [sentence] * n; enc(texts); ts = []
    for _ in range(10 if n < 1000 or "minishlab" in name else 3):
        st = time.perf_counter(); enc(texts); ts.append((time.perf_counter() - st) * 1000)
    avg = sum(ts) / len(ts); print(f"{n:5d} sentences | {avg:9.3f} ms | {avg/n:.4f} ms/sentence", flush=True)
