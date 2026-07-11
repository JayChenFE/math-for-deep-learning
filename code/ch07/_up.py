import json
with open('state/concepts-introduced.json','r',encoding='utf-8') as f: c=json.load(f)
c['7.1-7.3']=[{'concept':'Inverse/Singular/Pseudoinverse','definition':'A^-1 undoes A; singular=det=0; pinv=least-squares fallback','used_in':['Ch26']}]
c['7.4']=[{'concept':'Normal Equation','definition':'w=(X^T X)^-1 X^T y, one-step optimal weights','used_in':['Ch26']}]
with open('state/concepts-introduced.json','w',encoding='utf-8') as f: json.dump(c,f,ensure_ascii=False,indent=2)
with open('state/cross-refs.json','r',encoding='utf-8') as f: x=json.load(f)
x.append({'ref_id':'REF-019','source':'Ch7','section':'7.4','quote':'Closed-form -> Ch26 normal equation','target':'Ch26','status':'pending'})
with open('state/cross-refs.json','w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
print('Ch7 state OK')
