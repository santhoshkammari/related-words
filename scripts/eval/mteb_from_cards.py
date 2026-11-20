import yaml, glob, collections, re
for f in sorted(glob.glob("data/cards/*.md")):
    t = open(f).read()
    if not t.startswith("---"): continue
    fm = t.split("---",2)[1]
    try: meta = yaml.safe_load(fm)
    except Exception as e: print(f, "yaml err"); continue
    by = collections.defaultdict(list)
    for mi in meta.get("model-index",[]):
        for r in mi.get("results",[]):
            ty = r["task"]["type"]
            main = {"Retrieval":"ndcg_at_10","STS":"cos_sim_spearman","Classification":"accuracy","Clustering":"v_measure","Reranking":"map","PairClassification":"cos_sim_ap","Summarization":"cos_sim_spearman"}.get(ty)
            for m in r["metrics"]:
                if m["type"]==main: by[ty].append(m["value"]); break
    if by:
        allm = [sum(v)/len(v) for v in by.values()]
        print(f"{f.split('/')[-1][:48]:50}", " ".join(f"{k[:5]}={sum(v)/len(v):.1f}({len(v)})" for k,v in by.items()), f"| macro={sum(allm)/len(allm):.1f}")
