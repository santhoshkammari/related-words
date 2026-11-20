#!/usr/bin/env python3
"""related-words: text -> English words related in meaning, with scores and meanings.
Usage: python run.py "any text ..." [more words ...]    (positional args are joined as the text; stdin also works)
Options: -k N words per chunk (default 5) | -n N max words in final list (default 10) | --chunks per-chunk detail
         --json | --full (untruncated meanings) | --no-daemon (load model in-process, no warm server) | --stop (kill warm server)
Warm server: first call spawns a background server that keeps the model loaded and exits after IDLE seconds
(default 60) without requests. Later calls take ~0.1s instead of ~9s. Automatic, nothing to manage.
Env: GEMMA=/path/to/embeddinggemma-300m | THREADS=N (default half the CPUs) | IDLE=seconds"""
import argparse, json, os, re, socket, subprocess, sys, time, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "resources"); DIM = 256

def chunk(text, maxw=22, minw=5):
    """Sentences first; split long ones at , ; : and 'and/but/because...'; merge tiny pieces."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]; out = []
    for s in sents:
        if len(s.split()) <= maxw: out.append(s); continue
        cur = ""
        for p in re.split(r"(?<=[,;:])\s+|\s+(?=(?:and|but|because|although|while|which|so|yet|then)\b)", s):
            if not p: continue
            cand = (cur + " " + p).strip()
            if cur and len(cand.split()) > maxw: out.append(cur); cur = p
            else: cur = cand
        if cur: out.append(cur)
    merged = []
    for c in out:
        if merged and len(c.split()) < minw: merged[-1] += " " + c
        elif merged and len(merged[-1].split()) < minw: merged[-1] += " " + c
        else: merged.append(c)
    return merged


SOCK = f"/tmp/related-words-{os.getuid()}.sock"

def load():
    import numpy as np, pandas as pd, torch
    torch.set_num_threads(int(os.environ.get("THREADS", max(1, (os.cpu_count() or 8) // 2))))
    from sentence_transformers import SentenceTransformer
    gp = os.environ.get("GEMMA") or next((p for p in [os.path.expanduser("~/word-sense-retrieval/models/embeddinggemma-300m")] if os.path.isdir(p)), "google/embeddinggemma-300m")
    m = SentenceTransformer(gp, device="cpu")
    E = torch.from_numpy(np.load(f"{RES}/emb256_i8.npy").astype(np.float32) * np.load(f"{RES}/emb256_scale.npy").astype(np.float32)[:, None])  # int8 + per-row scale
    meta = pd.read_parquet(f"{RES}/meta_small.parquet")
    return dict(m=m, E=E, lem=meta.lemma.values, pos=meta.pos.values, zipf=meta.zipf.values, gloss=meta.gloss_leaf.values)

def suggest(st, text, k=5):
    import torch
    chunks = chunk(text)
    Q = st["m"].encode(chunks, prompt_name="query", convert_to_tensor=True, show_progress_bar=False).float()[:, :DIM]
    Q = torch.nn.functional.normalize(Q, dim=1); top = (Q @ st["E"].T).topk(300, dim=1); sc, ix = top.values.tolist(), top.indices.tolist()
    res, best = [], {}
    for i, c in enumerate(chunks):
        seen, words = set(), []
        for s, j in zip(sc[i], ix[i]):
            l = str(st["lem"][j])
            if l in seen: continue
            seen.add(l); w = dict(word=l, pos=str(st["pos"][j]), zipf=round(float(st["zipf"][j]), 1), score=round(float(s), 3), meaning=str(st["gloss"][j]))
            words.append(w)
            if l not in best or w["score"] > best[l]["score"]: best[l] = w
            if len(words) >= k: break
        res.append(dict(chunk=c, words=words))
    return dict(chunks=res, words=sorted(best.values(), key=lambda w: -w["score"]))

def serve():
    idle = float(os.environ.get("IDLE", 60))
    srv = socket.socket(socket.AF_UNIX)
    try: srv.bind(SOCK)
    except OSError:
        try: socket.socket(socket.AF_UNIX).connect(SOCK); return          # another server alive
        except OSError: os.unlink(SOCK); srv.bind(SOCK)                    # stale file
    os.chmod(SOCK, 0o600); srv.listen(4)
    st = load(); open(SOCK + ".ready", "w").close()
    srv.settimeout(idle)
    try:
        while True:
            try: c, _ = srv.accept()
            except socket.timeout: break
            with c:
                try:
                    req = json.loads(c.makefile("r").readline())
                    if req.get("stop"): c.sendall(b"{}\n"); break
                    out = suggest(st, req["text"], req.get("k", 5))
                except Exception as e: out = dict(error=str(e))
                c.sendall((json.dumps(out) + "\n").encode())
    finally:
        for p in (SOCK, SOCK + ".ready"):
            try: os.unlink(p)
            except OSError: pass

def ask(req):
    def conn():
        c = socket.socket(socket.AF_UNIX); c.connect(SOCK); return c
    try: c = conn()
    except OSError:
        if os.path.exists(SOCK + ".ready"): os.unlink(SOCK + ".ready")
        if req.get("stop"): return {}
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(600):                                               # wait up to 60s for model load
            if os.path.exists(SOCK + ".ready"):
                try: c = conn(); break
                except OSError: pass
            time.sleep(0.1)
        else: raise RuntimeError("server did not start")
    with c:
        c.sendall((json.dumps(req) + "\n").encode()); return json.loads(c.makefile("r").readline())

def fmt(w, full):
    m = " ".join(w["meaning"].split()); m = m if full or len(m) <= 140 else m[:137] + "..."
    return f"**{w['word']}** ({w['pos']}, zipf {w['zipf']}, score {w['score']:.2f}) - {m}"

def main():
    if "--serve" in sys.argv: return serve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*"); ap.add_argument("-k", type=int, default=20); ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--full", action="store_true"); ap.add_argument("--json", action="store_true"); ap.add_argument("--chunks", action="store_true")
    ap.add_argument("--no-daemon", action="store_true"); ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()
    if a.stop: ask(dict(stop=True)); print("stopped"); return
    text = " ".join(a.text) if a.text else (sys.stdin.read() if not sys.stdin.isatty() else "")
    if not text.strip(): ap.error("give some text (args or stdin)")
    t0 = time.time()
    r = suggest(load(), text, a.k) if a.no_daemon else ask(dict(text=text, k=a.k))
    if "error" in r: sys.exit("error: " + r["error"])
    r["words"] = r["words"][:a.n]
    if a.json: print(json.dumps(r, ensure_ascii=False, indent=1)); return
    print(f"Related words ({len(r['chunks'])} chunk(s), {time.time()-t0:.1f}s). Ranked by semantic match; lower zipf = rarer word.\n")
    for i, w in enumerate(r["words"], 1): print(f"{i}. {fmt(w, a.full)}")
    if a.chunks:
        for i, c in enumerate(r["chunks"], 1):
            print(f"\nChunk {i}: \"{c['chunk']}\"")
            for w in c["words"]: print(f"- {fmt(w, a.full)}")

main()
