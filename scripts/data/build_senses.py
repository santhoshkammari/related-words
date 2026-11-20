"""kaikki English JSONL -> senses.parquet (one row per sense) with a lemma-level split.
usage: python -I build_senses.py IN.jsonl OUT.parquet"""
import sys, json, re, hashlib
import pandas as pd
from wordfreq import zipf_frequency

KEEP_POS = {"noun": "noun", "verb": "verb", "adj": "adj", "adv": "adv"}
BAD_TAGS = {"obsolete", "archaic", "dated", "vulgar", "offensive", "derogatory", "slur", "dialectal", "misspelling",
            "nonstandard", "form-of", "alt-of", "alternative", "abbreviation", "initialism", "acronym", "ethnic-slur", "pejorative"}
FORMGLOSS = re.compile(r"^(alternative|obsolete|archaic|dated|nonstandard|misspelling|plural|past|present|third-person|simple past|"
                       r"singular|comparative|superlative|initialism|abbreviation|acronym|clipping|contraction|eye dialect|"
                       r"pronunciation spelling|synonym|rare|informal|gerund|participle|inflection)\b.{0,60}\b(of|form of)\b", re.I)
WORD = re.compile(r"^[a-z][a-z\-]*$")


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


def split_of(lemma):
    h = int(hashlib.md5(lemma.encode()).hexdigest(), 16) % 100
    return "test" if h < 5 else ("val" if h < 10 else "train")


def main(src, dst):
    rows, n, zcache = [], 0, {}
    for line in open(src, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        n += 1
        if d.get("lang_code") != "en" or d.get("pos") not in KEEP_POS:
            continue
        w = d["word"]
        if not WORD.match(w):  # single lowercase tokens only (hyphen ok): drops names, phrases, capitalised
            continue
        if w not in zcache:
            zcache[w] = zipf_frequency(w, "en")
        z = zcache[w]
        for i, s in enumerate(d.get("senses", [])):
            tags = set(s.get("tags", []))
            if tags & BAD_TAGS or s.get("form_of") or s.get("alt_of"):
                continue
            g = s.get("glosses") or []
            if not g:
                continue
            gloss, leaf = clean(" > ".join(g)), clean(g[-1])
            if len(leaf) < 8 or FORMGLOSS.match(leaf):
                continue
            exs = [e for e in s.get("examples", []) if e.get("text")]
            ex = [clean(e["text"]) for e in exs if e.get("type") != "quotation"]
            qt = [clean(e["text"]) for e in exs if e.get("type") == "quotation"]
            ex = [e for e in ex if 15 <= len(e) <= 300][:5]
            qt = [e for e in qt if 15 <= len(e) <= 300][:2]
            syn = [x["word"] for x in s.get("synonyms", []) if isinstance(x, dict) and x.get("word")][:8]
            rows.append(dict(sense_id=s.get("id") or f"{w}-{d['pos']}-{i}", lemma=w, pos=KEEP_POS[d["pos"]], gloss=gloss,
                             gloss_leaf=leaf, examples=ex, quotes=qt, tags=sorted(tags), synonyms=syn, zipf=round(z, 2),
                             split=split_of(w)))
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    print(f"entries read {n:,} | senses kept {len(df):,} | lemmas {df.lemma.nunique():,}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
