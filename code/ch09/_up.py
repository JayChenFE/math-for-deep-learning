import json
with open('state/concepts-introduced.json','r',encoding='utf-8') as f: c=json.load(f)
c['9.1-9.2']=[{'concept':'SVD','definition':'A=U*Sigma*V^T, any matrix decomposes into 3 parts','used_in':['Ch9.3-9.5','LoRA']}]
c['9.3']=[{'concept':'Low-Rank Approximation','definition':'Top-k singular values reconstruct optimal k-rank approx','used_in':['Ch9.5 LoRA','Recommender Systems']}]
c['9.5']=[{'concept':'LoRA','definition':'DeltaW=B@A (low-rank), train only 2dr params for LLM fine-tuning','used_in':['LLM practice']}]
with open('state/concepts-introduced.json','w',encoding='utf-8') as f: json.dump(c,f,ensure_ascii=False,indent=2)
with open('state/cross-refs.json','r',encoding='utf-8') as f: x=json.load(f)
x.append({'ref_id':'REF-022','source':'Ch9','section':'9.5','quote':'LoRA fine-tuning large models','target':'LLM practice','status':'reference'})
with open('state/cross-refs.json','w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
print('Ch9 state OK')
