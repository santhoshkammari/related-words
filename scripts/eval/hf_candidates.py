import json, urllib.request, urllib.parse
def get(u):
    return json.load(urllib.request.urlopen(u, timeout=30))
seen = {}
queries = ["bge-small","bge-m3","MiniLM","e5-small","gte-small","gte-modernbert","modernbert-embed","embeddinggemma","arctic-embed","snowflake-arctic-embed-s","nomic-embed","static-retrieval","potion","model2vec","jina-embeddings","minishlab","granite-embedding","bert-tiny","ettin","qwen3-embedding","harrier","embed-small"]
for q in queries:
    u = "https://huggingface.co/api/models?" + urllib.parse.urlencode({"search": q, "sort": "downloads", "direction": -1, "limit": 8, "full": "false"})
    try:
        for m in get(u):
            seen[m["id"]] = m
    except Exception as e:
        print("ERR", q, e)
rows = []
for mid, m in seen.items():
    if m.get("pipeline_tag") not in ("sentence-similarity","feature-extraction"): continue
    rows.append((m.get("downloads",0), mid, m.get("likes",0), m.get("lastModified","")[:10]))
rows.sort(reverse=True)
for d, mid, l, lm in rows[:45]:
    print(f"{d:>10} {l:>5} {lm} {mid}")
