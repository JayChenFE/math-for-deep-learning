"""Validate Chapter 30: Transformer Training (PyTorch required)."""
try:
    import torch; import torch.nn as nn
    class MiniGPT(nn.Module):
        def __init__(self,v=1000,d=128,h=2,b=2,L=32):
            super().__init__(); self.emb=nn.Embedding(v,d)
            self.pe=nn.Parameter(torch.randn(1,L,d)); self.enc=nn.TransformerEncoderLayer(d,h,256,batch_first=True,norm_first=True)
            self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,v)
        def forward(self,x):
            B,T=x.shape; h=self.emb(x)+self.pe[:,:T,:]
            m=nn.Transformer.generate_square_subsequent_mask(T)
            h=self.enc(h,src_mask=m,is_causal=True); return self.head(self.ln(h))
    m=MiniGPT(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    x=torch.randint(0,1000,(2,16)); labels=torch.randint(0,1000,(2,16))
    logits=m(x); loss=nn.functional.cross_entropy(logits.view(-1,1000),labels.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    assert logits.shape==(2,16,1000)
    print("Ch30 PyTorch OK")
except ImportError: print("Ch30 SKIP (no torch)")
