import time, torch, sys
from sentence_transformers import SentenceTransformer
torch.set_num_threads(8)
base = "She stayed calm and kept her composure even though the meeting turned hostile and everyone else was shouting at each other."
texts = [base + f" Case {i}." for i in range(256)]
models = ["sentence-transformers/all-MiniLM-L6-v2","Snowflake/snowflake-arctic-embed-xs","BAAI/bge-small-en-v1.5","thenlper/gte-small","intfloat/e5-small-v2","ibm-granite/granite-embedding-small-english-r2","minishlab/potion-base-8M","google/embeddinggemma-300m"]
for mid in models:
    try:
        m = SentenceTransformer(mid, device="cpu", trust_remote_code=False)
        m.encode(texts[:16], batch_size=16)
        t = time.time(); e = m.encode(texts, batch_size=64); dt = time.time()-t
        t = time.time()
        for x in texts[:30]: m.encode([x])
        lat = (time.time()-t)/30*1000
        print(f"{mid:50} dim={e.shape[1]:4} {len(texts)/dt:7.0f} sent/s  single={lat:6.1f} ms", flush=True)
    except Exception as ex:
        print(mid, "ERR", str(ex)[:150], flush=True)
