import json
with open('state/concepts-introduced.json','r',encoding='utf-8') as f: c=json.load(f)
c['8.1-8.2']=[{'concept':'Eigenvalue & Eigenvector','definition':'A*v=lambda*v; v direction unchanged, only scaled','used_in':['Ch9 SVD','Ch17 PCA']}]
c['8.3']=[{'concept':'Eigenvalue Decay','definition':'Large lambda=rich info direction; small lambda=discardable','used_in':['Ch9','Ch17']}]
with open('state/concepts-introduced.json','w',encoding='utf-8') as f: json.dump(c,f,ensure_ascii=False,indent=2)
with open('state/cross-refs.json','r',encoding='utf-8') as f: x=json.load(f)
x.append({'ref_id':'REF-020','source':'Ch8','section':'8.3','quote':'PCA covariance eigendecomposition (Ch17)','target':'Ch17','status':'pending'})
x.append({'ref_id':'REF-021','source':'Ch8','section':'8.3','quote':'Eigen -> SVD (Ch9) -> LoRA','target':'Ch9','status':'pending'})
with open('state/cross-refs.json','w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
print('Ch8 state OK')
