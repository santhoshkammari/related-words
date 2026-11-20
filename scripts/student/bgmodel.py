import numpy as np, torch, torch.nn.functional as F
P1, P2 = 1000003, 998244353
def bucket(a, b, B): return ((a.long() * P1 + b.long() * P2) % B)
def encode_ids(E, Bg, ids, B):               # ids: (n, L) mapped rows, -1 pad
    m1 = ids >= 0; u = E[ids.clamp(min=0)] * m1[..., None]
    out = u.sum(1) / m1.sum(1, keepdim=True).clamp(min=1)
    m2 = m1[:, 1:] & m1[:, :-1]
    if m2.shape[1]:
        h = bucket(ids[:, :-1].clamp(min=0), ids[:, 1:].clamp(min=0), B)
        g = Bg[h] * m2[..., None]; out = out + g.sum(1) / m2.sum(1, keepdim=True).clamp(min=1)
    return F.normalize(out, dim=-1)
def tok_ids(sm, texts, L=64):
    mp = sm.token_mapping if sm.token_mapping is not None else np.arange(len(sm.embedding))
    ids = np.full((len(texts), L), -1, np.int64)
    for i, l in enumerate(sm.tokenize(texts)):
        l = mp[l[:L]] if len(l) else []; ids[i, :len(l)] = l
    return torch.from_numpy(ids)
