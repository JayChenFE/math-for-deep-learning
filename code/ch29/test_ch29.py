"""Validate Chapter 29: Transformer Architecture (PyTorch required)."""
try:
    import torch; import torch.nn as nn; import math
    class MHA(nn.Module):
        def __init__(self,d=512,h=8):
            super().__init__(); self.d,self.h,self.dk=d,h,d//h
            self.WQ=nn.Linear(d,d); self.WK=nn.Linear(d,d)
            self.WV=nn.Linear(d,d); self.WO=nn.Linear(d,d)
        def forward(self,x):
            B,T,D=x.shape; Q=self.WQ(x).view(B,T,self.h,self.dk).transpose(1,2)
            K=self.WK(x).view(B,T,self.h,self.dk).transpose(1,2)
            V=self.WV(x).view(B,T,self.h,self.dk).transpose(1,2)
            s=Q@K.transpose(-2,-1)/math.sqrt(self.dk); a=torch.softmax(s,dim=-1)
            o=a@V; return self.WO(o.transpose(1,2).contiguous().view(B,T,D))
    torch.manual_seed(42); mha=MHA(); x=torch.randn(2,10,512)
    out=mha(x); assert out.shape==(2,10,512)
    mask=torch.tril(torch.ones(10,10)).unsqueeze(0).unsqueeze(0)
    print("Ch29 PyTorch OK")
except ImportError: print("Ch29 SKIP (no torch)")
