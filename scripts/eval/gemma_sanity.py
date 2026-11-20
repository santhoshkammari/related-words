"""Step 2: inspect EmbeddingGemma and test whether tokenlearn's stock target (mean of token embeddings[1:-1])
matches the real sentence embedding. Needs HF access to google/embeddinggemma-300m.
usage: python -I gemma_sanity.py [device]"""
import sys, time, json, numpy as np, torch
from sentence_transformers import SentenceTransformer

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
m = SentenceTransformer("google/embeddinggemma-300m", device=dev)
print("MODULES:", [type(x).__name__ for x in m])
print("max_seq_length:", m.max_seq_length, "| dim:", m.get_sentence_embedding_dimension())
print("PROMPTS:", json.dumps(m.prompts)[:600])
tok = m.tokenizer
print("vocab_size:", len(tok), "| pad/bos/eos/unk ids:", tok.pad_token_id, tok.bos_token_id, tok.eos_token_id, tok.unk_token_id,
      "| padding_side:", tok.padding_side)
texts = ["She stayed calm even though the meeting turned hostile.", "To delay doing something that should be done.",
         "play", "He refused to give up despite repeated failures."]
ids = tok(texts[0])["input_ids"]
print("token ids ex:", ids, "->", tok.convert_ids_to_tokens(ids))

def cos(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)

full = m.encode(texts, convert_to_numpy=True)
tokemb = m.encode(texts, output_value="token_embeddings", convert_to_numpy=False)
stock = np.stack([t[1:-1].float().mean(0).cpu().numpy() if hasattr(t, "float") else np.asarray(t)[1:-1].mean(0) for t in tokemb])
print("token_embeddings dim:", stock.shape[1], "| full embedding dim:", full.shape[1])
if stock.shape[1] == full.shape[1]:
    print("cos(stock target, real embedding) per text:", np.round(cos(stock, full), 3))
else:
    print("DIMS DIFFER -> stock featurizer target is in a different space than encode() output")
print("norm(full) ~", np.round(np.linalg.norm(full, axis=1), 3))

# MRL: how much does truncating to 256 + renorm preserve the similarity structure?
rng = np.random.default_rng(0)
pool = [t for t in open(sys.argv[2]).read().splitlines() if t][:512] if len(sys.argv) > 2 else texts * 8
E = m.encode(pool, convert_to_numpy=True, batch_size=64)
E256 = E[:, :256] / np.linalg.norm(E[:, :256], axis=1, keepdims=True)
S, S2 = E @ E.T, E256 @ E256.T
iu = np.triu_indices(len(E), 1)
print("MRL256 pairwise-sim Pearson r vs 768:", round(float(np.corrcoef(S[iu], S2[iu])[0, 1]), 4))

# throughput
batch = [texts[0] + f" case {i}" for i in range(512)]
m.encode(batch[:32]); torch.cuda.synchronize() if dev == "cuda" else None
t = time.time(); m.encode(batch, batch_size=128); torch.cuda.synchronize() if dev == "cuda" else None
print(f"throughput on {dev}: {len(batch) / (time.time() - t):.0f} sent/s")
