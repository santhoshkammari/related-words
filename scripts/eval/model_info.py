import json, urllib.request
ids = ["sentence-transformers/all-MiniLM-L6-v2","BAAI/bge-small-en-v1.5","intfloat/e5-small-v2","thenlper/gte-small","Snowflake/snowflake-arctic-embed-xs","ibm-granite/granite-embedding-small-english-r2","google/embeddinggemma-2","jinaai/jina-embeddings-v5-omni-small","nomic-ai/nomic-embed-text-v1.5","minishlab/potion-base-8M"]
for i in ids:
    try:
        m = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{i}", timeout=30))
        p = (m.get("safetensors") or {}).get("total")
        c = m.get("config",{})
        lic = next((t[8:] for t in m.get("tags",[]) if t.startswith("license:")), "?")
        print(f"{i:55} params={p and round(p/1e6)}M arch={c.get('model_type')} lic={lic}")
    except Exception as e:
        print(i, "ERR", e)
