import json
with open('state/concepts-introduced.json','r',encoding='utf-8') as f: c=json.load(f)
c['6.1-6.2']=[{'concept':'Matrix Multiplication','definition':'C[i,j]=row_i(A)*col_j(B), core of neural nets','used_in':['Ch7','Ch26-30']}]
c['6.3']=[{'concept':'Linear Transform Geometry','definition':'Matrix warps space grid','used_in':['Ch8','Ch28']}]
c['6.4']=[{'concept':'Linear Layer','definition':'output = X@W.T + b, nn.Linear math','used_in':['Ch26-30']}]
c['6.5']=[{'concept':'Broadcasting','definition':'Auto-expand dims from last axis, dim=1 or match','used_in':['Ch29','Ch30']}]
with open('state/concepts-introduced.json','w',encoding='utf-8') as f: json.dump(c,f,ensure_ascii=False,indent=2)
with open('state/cross-refs.json','r',encoding='utf-8') as f: x=json.load(f)
x.append({'ref_id':'REF-017','source':'Ch6','section':'6.3','quote':'Multi-layer = repeated space warping (Ch28)','target':'Ch28','status':'pending'})
x.append({'ref_id':'REF-018','source':'Ch6','section':'6.5','quote':'Transformer mask broadcasting (Ch29)','target':'Ch29','status':'pending'})
with open('state/cross-refs.json','w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
print('Ch6 state OK')
