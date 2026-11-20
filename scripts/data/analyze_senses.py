import json, sys, pandas as pd, numpy as np
d = pd.read_parquet("data/senses.parquet")
d["n_ex"] = d.examples.str.len()
bands = [(0, 1), (1, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 8)]
rows = []
for lo, hi in bands:
    b = d[(d.zipf >= lo) & (d.zipf < hi)]
    rows.append(dict(band=f"{lo}-{hi}", lemmas=b.lemma.nunique(), senses=len(b), sense_per_lemma=round(len(b) / max(b.lemma.nunique(), 1), 2),
                     pct_with_example=round((b.n_ex > 0).mean() * 100, 1), pct_with_synonym=round((b.synonyms.str.len() > 0).mean() * 100, 1)))
print(pd.DataFrame(rows).to_string(index=False))
band = d[(d.zipf >= 2.5) & (d.zipf <= 4.5)]
print("\nmid-frequency band 2.5-4.5:", band.lemma.nunique(), "lemmas,", len(band), "senses | pos:", band.pos.value_counts().to_dict())
print("split (band, by lemma):", band.groupby("split").lemma.nunique().to_dict())
print("gloss_leaf len median/95%:", int(band.gloss_leaf.str.len().median()), int(band.gloss_leaf.str.len().quantile(.95)))
ev = [json.loads(l) for l in open("eval/situations.jsonl")]
lem = d.groupby("lemma").agg(zipf=("zipf", "first"), senses=("sense_id", "count"), ex=("n_ex", "sum"))
miss, out_band, ok = [], [], 0
for e in ev:
    t = e["targets"][0]
    if t not in lem.index: miss.append(t)
    elif not (2.5 <= lem.loc[t, "zipf"] <= 4.5): out_band.append((t, lem.loc[t, "zipf"]))
    else: ok += 1
print(f"\neval primary targets: {ok}/{len(ev)} in table & band | missing: {miss} | in table but outside band: {out_band}")
print("eval any-target coverage in table:", sum(any(t in lem.index for t in e["targets"]) for e in ev), "/", len(ev))
print("\nmost polysemous lemmas in band:", band.groupby("lemma").size().sort_values(ascending=False).head(8).to_dict())
