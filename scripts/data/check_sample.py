import sys, pandas as pd
d = pd.read_parquet(sys.argv[1])
print(d.split.value_counts().to_dict(), "| with examples:", round((d.examples.str.len() > 0).mean(), 2), "| with synonyms:", round((d.synonyms.str.len() > 0).mean(), 2))
print("zipf:", d.zipf.describe().round(2).to_dict())
print(d.sample(6, random_state=1)[["lemma", "pos", "gloss_leaf", "zipf"]].to_string(max_colwidth=70))
