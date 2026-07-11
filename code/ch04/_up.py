import json
with open('state/concepts-introduced.json','r',encoding='utf-8') as f: c=json.load(f)
c['4.1-4.3']=[{'concept':'Vector add/mul/sub','definition':'Parallelogram rule, scalar scaling, difference','used_in':['Ch5','Ch6']}]
c['4.4']=[{'concept':'Word Embedding Arithmetic','definition':'king-man+woman~queen','used_in':['Ch29']}]
with open('state/concepts-introduced.json','w',encoding='utf-8') as f: json.dump(c,f,ensure_ascii=False,indent=2)
with open('state/cross-refs.json','r',encoding='utf-8') as f: x=json.load(f)
x.append({'ref_id':'REF-014','source':'Ch4','section':'4.4','quote':'Self-Attention (Ch29)','target':'Ch29','status':'pending'})
with open('state/cross-refs.json','w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
print('state OK')
