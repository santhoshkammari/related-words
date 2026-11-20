"""Sage-A1.5 (vLLM, :8082) -> synthetic situations per sense. Resumable, leak-filtered.
usage: python -I gen_situations.py OUT.jsonl --n_train 160 --n_test 40 --k_train 5 --k_test 2 [--conc 16]"""
import re, json, asyncio, argparse, random, time, os, pandas as pd
from openai import AsyncOpenAI

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--root", default=".")
ap.add_argument("--n_train", type=int, default=160); ap.add_argument("--n_test", type=int, default=40)
ap.add_argument("--k_train", type=int, default=5); ap.add_argument("--k_test", type=int, default=2)
ap.add_argument("--conc", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--url", default="http://localhost:8082/v1"); ap.add_argument("--max_rank", type=int, default=2)
a = ap.parse_args()

REG = ["casual conversation between friends", "workplace or professional setting", "news report", "short story / fiction narration",
       "formal or academic writing", "social media post or text message", "family or household scene", "travel or everyday errand"]
df = pd.read_parquet(f"{a.root}/data/senses.parquet")
df = df.assign(rank=df.groupby(["lemma", "pos"]).cumcount())
df = df[(df.gloss_leaf.str.len() >= 15) & (df["rank"] <= a.max_rank)]
tr = df[(df.split == "train") & (df.zipf >= 2.0) & (df.zipf <= 4.7)]
te = df[(df.split == "test") & (df.zipf >= 2.5) & (df.zipf <= 4.5)]
sel = pd.concat([tr.sample(min(a.n_train, len(tr)), random_state=a.seed), te.sample(min(a.n_test, len(te)), random_state=a.seed)])
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["sense_id"] for l in open(a.out)}
todo = [r for r in sel.itertuples() if r.sense_id not in done]
print(f"selected {len(sel)} senses ({sum(sel.split=='train')} train / {sum(sel.split=='test')} test), already done {len(done)}, todo {len(todo)}", flush=True)

def leaks(s, w): return bool(re.search(r"\b" + re.escape(w[:max(4, len(w) - 2)]), s, re.I))

def prompt(r, k):
    rng = random.Random(r.sense_id); regs = rng.sample(REG, k)
    lines = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(regs))
    ex = f"\nDictionary example (for the meaning only, do not copy): {r.examples[0]}" if len(r.examples) else ""
    return (f"Word: {r.lemma} ({r.pos})\nMeaning: {r.gloss}{ex}\n\n"
            f"Write {k} different, specific, natural situations (each 12-35 words) where this exact word and meaning would be the perfect word "
            f"to describe what is happening, one for each setting:\n{lines}\n\n"
            f"Rules: NEVER use the word \"{r.lemma}\" or any inflection or derived form of it. Do not define the word. Do not quote the dictionary. "
            f"Describe the situation concretely (people, actions, details) so a reader could infer the word. Vary names, topics and sentence structure.\n"
            f"Return ONLY a JSON list of {k} strings.")

stats = dict(req=0, fail=0, kept=0, leak=0, short=0, toks=0, nojson=0)
fraw = open(a.out + ".raw", "a")
cli = AsyncOpenAI(base_url=a.url, api_key="x", timeout=180)
sem = asyncio.Semaphore(a.conc); fout = open(a.out, "a"); t0 = time.time()

async def one(r):
    k = a.k_train if r.split == "train" else a.k_test
    async with sem:
        try:
            resp = await cli.chat.completions.create(model="sage-a1.5", messages=[{"role": "user", "content": prompt(r, k)}],
                       max_tokens=60 * k + 120, temperature=0.9, top_p=0.95, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            txt = resp.choices[0].message.content or ""; stats["toks"] += resp.usage.completion_tokens
            items = []
            i0 = min([i for i in (txt.find("["), txt.find("{")) if i >= 0], default=-1)
            if i0 >= 0:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(txt[i0:])
                    def flat(o):
                        if isinstance(o, str): return [o]
                        if isinstance(o, dict): o = list(o.values())
                        return [y for x in o for y in flat(x)] if isinstance(o, list) else []
                    items = flat(obj)
                except Exception:
                    items = []
            if not items:
                stats["nojson"] += 1; fraw.write(json.dumps(dict(lemma=r.lemma, finish=resp.choices[0].finish_reason, ntok=resp.usage.completion_tokens, content=txt[:600])) + "\n"); fraw.flush()
        except Exception as e:
            stats["fail"] += 1; items = None
    stats["req"] += 1
    if items is None: return
    good, seen = [], set()
    for s in items:
        if not isinstance(s, str): continue
        s = re.sub(r"\s+", " ", s).strip(); n = len(s.split())
        if leaks(s, r.lemma): stats["leak"] += 1; continue
        if not (8 <= n <= 55) or s.lower() in seen: stats["short"] += 1; continue
        seen.add(s.lower()); good.append(s)
    stats["kept"] += len(good)
    fout.write(json.dumps(dict(sense_id=r.sense_id, lemma=r.lemma, pos=r.pos, split=r.split, gloss_leaf=r.gloss_leaf, situations=good)) + "\n"); fout.flush()
    if stats["req"] % 100 == 0:
        el = time.time() - t0
        print(f"[{el:5.0f}s] req {stats['req']} ({stats['req'] / el:.2f}/s) kept {stats['kept']} leak {stats['leak']} short {stats['short']} fail {stats['fail']} | {stats['toks'] / el:.0f} tok/s", flush=True)

async def main():
    await asyncio.gather(*[one(r) for r in todo])
    el = time.time() - t0
    print(f"DONE {stats} in {el:.0f}s | {stats['req'] / max(el, 1):.2f} req/s | {stats['toks'] / max(el, 1):.0f} tok/s", flush=True)
asyncio.run(main())
