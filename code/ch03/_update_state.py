import json, sys
sys.path.insert(0, 'state')

# Update concepts
with open('state/concepts-introduced.json', 'r', encoding='utf-8') as f:
    c = json.load(f)
c['3.1'] = [{'concept': 'Scalar', 'definition': '0D, shape=(), e.g. loss value', 'used_in': ['全書']}]
c['3.2'] = [{'concept': 'Vector', 'definition': '1D array, shape=(n,), e.g. user features', 'used_in': ['Ch4', 'Ch5', 'Ch29']}]
c['3.3'] = [{'concept': 'Matrix', 'definition': '2D table, shape=(m,n), e.g. batch input', 'used_in': ['Ch6', 'Ch7', 'Ch29']}]
c['3.4'] = [{'concept': 'Tensor', 'definition': '3D+ array, e.g. Transformer (batch,seq_len,d_model)', 'used_in': ['Ch29']}]
c['3.5'] = [{'concept': 'Axis', 'definition': 'Dimension index; sum along axis removes that dim', 'used_in': ['Ch6', 'Ch22', 'Ch29']}]
with open('state/concepts-introduced.json', 'w', encoding='utf-8') as f:
    json.dump(c, f, ensure_ascii=False, indent=2)
print('concepts OK')

# Update cross-refs
with open('state/cross-refs.json', 'r', encoding='utf-8') as f:
    x = json.load(f)
x.append({'ref_id': 'REF-012', 'source': 'Ch3', 'section': '3.4', 'quote': '(batch,seq_len,d_model) appears in Ch29', 'target': 'Ch29', 'status': 'pending'})
x.append({'ref_id': 'REF-013', 'source': 'Ch3', 'section': '3.5', 'quote': 'softmax(dim=-1) in Ch22/Ch29', 'target': 'Ch22', 'status': 'pending'})
with open('state/cross-refs.json', 'w', encoding='utf-8') as f:
    json.dump(x, f, ensure_ascii=False, indent=2)
print('cross-refs OK')
