import json
with open('state/concepts-introduced.json','r',encoding='utf-8') as f: c=json.load(f)
c['5.1-5.2']=[{'concept':'Dot Product & Cosine Similarity','definition':'a*b=sum(a_i*b_i)=|a||b|cos(theta)','used_in':['Ch6','Ch29']}]
c['5.3-5.4']=[{'concept':'Batch Q@K^T','definition':'All query-key similarity scores at once','used_in':['Ch29']}]
with open('state/concepts-introduced.json','w',encoding='utf-8') as f: json.dump(c,f,ensure_ascii=False,indent=2)
with open('state/cross-refs.json','r',encoding='utf-8') as f: x=json.load(f)
x.append({'ref_id':'REF-015','source':'Ch5','section':'5.4','quote':'Q@K.T = attention (Ch29)','target':'Ch29','status':'pending'})
with open('state/cross-refs.json','w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
print('Ch5 state OK')
