import json, urllib.request, urllib.parse
def get(u): return json.load(urllib.request.urlopen(u, timeout=30))
seen = {}
for q in ["embeddinggemma"]:
    for m in get("https://huggingface.co/api/models?"+urllib.parse.urlencode({"search":q,"sort":"downloads","direction":-1,"limit":40})):
        seen[m["id"]] = m
for mid in list(seen)[:25]:
    try:
        m = get(f"https://huggingface.co/api/models/{mid}")
        p = (m.get("safetensors") or {}).get("total")
        print(f"{m.get('downloads',0):>9} {mid:60} params={p and round(p/1e6)}M")
    except Exception as e:
        print(mid, "ERR")
