import urllib.request, json, re, os
ids = ["sentence-transformers/all-MiniLM-L6-v2","Snowflake/snowflake-arctic-embed-xs","BAAI/bge-small-en-v1.5","thenlper/gte-small","intfloat/e5-small-v2","ibm-granite/granite-embedding-small-english-r2","minishlab/potion-base-8M","google/embeddinggemma-300m"]
for i in ids:
    try:
        t = urllib.request.urlopen(f"https://huggingface.co/{i}/raw/main/README.md", timeout=30).read().decode()
    except Exception as e:
        t = ""; print(i, "ERR", e)
    open("data/cards/"+i.replace("/","__")+".md","w").write(t)
    # MTEB averages from model-index metadata
    sc = {}
    for m in re.finditer(r"- task:\n\s+type: (\w+)(?:.|\n)*?metrics:\n((?:\s+- type: \w+\n\s+value: [\d.]+\n?(?:\s+verified: \w+\n?)?)+)", t):
        pass
    print(f"=== {i}  ({len(t)} chars)")
    for kw in ["train", "dataset", "pairs", "billion", "million", "distill", "fine-tun", "contrastive", "synthetic", "MTEB"]:
        pass
