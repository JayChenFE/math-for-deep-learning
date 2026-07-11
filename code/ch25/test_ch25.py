"""Validate Chapter 25: Weight Initialization."""
import numpy as np
x=np.random.randn(1000,256)
for name,std in [("Xavier",np.sqrt(2/(256+256))),("Kaiming",np.sqrt(2/256)),("Bad",0.01)]:
    W=np.random.randn(256,256)*std; h=x@W.T
    assert abs(h.mean())<0.1
    if name=="Bad": assert h.std()<0.3
print("Ch25 ALL PASSED")
