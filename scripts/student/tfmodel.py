import torch, torch.nn as nn, torch.nn.functional as F
class TF(nn.Module):
    def __init__(s, E, L=64, layers=2, heads=4, ff=1024):
        super().__init__(); d = E.shape[1]
        s.emb = nn.Embedding.from_pretrained(E.clone(), freeze=False); s.pos = nn.Embedding(L, d); nn.init.normal_(s.pos.weight, std=0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, 0.0, batch_first=True, norm_first=True), layers)
        s.ln = nn.LayerNorm(d); s.skip = nn.Parameter(torch.tensor(1.0))
    def forward(s, ids):
        m = ids >= 0; x0 = s.emb(ids.clamp(min=0)); x = x0 + s.pos.weight[: ids.shape[1]]
        h = s.ln(s.enc(x, src_key_padding_mask=~m)); w = m[..., None].float()
        pool = lambda z: (z * w).sum(1) / w.sum(1).clamp(min=1)
        return F.normalize(pool(h) + s.skip * pool(x0), dim=-1)
