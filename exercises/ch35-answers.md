# 第35章  习题答案

---

## 1. ensure_json 装饰器 + 指数退避

```python
import json
import time
import numpy as np

class JSONRetryError(Exception):
    pass

def ensure_json(max_retries=3, base_delay=1.0):
    """装饰器：自动检测输出是否为合法JSON，非法则重试"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            T = kwargs.get('temperature', 1.0)
            for attempt in range(max_retries):
                result = func(*args, **kwargs)
                try:
                    parsed = json.loads(result)
                    return parsed, attempt + 1
                except (json.JSONDecodeError, ValueError):
                    if attempt < max_retries - 1:
                        wait = base_delay * (2 ** attempt)
                        print(f"  Retry {attempt+1}: JSON parse failed, "
                              f"cooling T {T:.2f}->{max(0.05, T*0.5):.2f}, "
                              f"waiting {wait:.0f}s")
                        T = max(0.05, T * 0.5)
                        kwargs['temperature'] = T
                        time.sleep(wait)
            raise JSONRetryError(f"Failed after {max_retries} retries")
        return wrapper
    return decorator

# 模拟测试
np.random.seed(42)
@ensure_json(max_retries=3, base_delay=0.01)
def mock_llm(prompt, temperature=1.0):
    # 模拟：高温度时容易输出非法JSON
    prob_valid = max(0.1, 1.0 - temperature * 0.8)
    if np.random.random() < prob_valid:
        return '{"action": "search", "query": "weather"}'
    else:
        return 'Search: weather (invalid JSON)'

for T in [0.1, 0.5, 1.0]:
    try:
        result, attempts = mock_llm("test", temperature=T)
        print(f"T={T:.1f}: success after {attempts} attempt(s)")
    except JSONRetryError:
        print(f"T={T:.1f}: failed after all retries")
```

---

## 2. Prompt vs Prompt + Logit Bias 对比实验

```python
import numpy as np

np.random.seed(42)
V, n_trials = 30, 100
json_tokens = {0, 1, 2, 3, 29}

def softmax(x, T=1.0):
    x = np.float64(x)/T; x = x-x.max(); e = np.exp(x); return e/e.sum()

def test_strategy(logits, T, use_logit_bias=False):
    successes = 0
    for i in range(n_trials):
        probs = softmax(logits[i], T)
        if use_logit_bias:
            bias = np.full(V, -np.inf)
            for tid in json_tokens: bias[tid] = 0.0
            probs = softmax(logits[i] + bias, T)
        token = np.random.choice(V, p=probs)
        if token in json_tokens:
            successes += 1
    return (1 - successes/n_trials) * 100

for T in [0.3, 0.7]:
    logits = np.random.randn(n_trials, V) * 0.8
    err_prompt = test_strategy(logits, T, use_logit_bias=False)
    err_bias = test_strategy(logits, T, use_logit_bias=True)
    print(f"T={T}: Prompt only={err_prompt:.0f}%, Prompt+LogitBias={err_bias:.0f}%")
    print(f"  Logit Bias reduces error by {err_prompt-err_bias:.0f} percentage points")
```

---

## 3. outlines 库 vs 手写 Logit Bias

```python
# 手写 Logit Bias（本章教学版）：~50 行
# 需要手动管理 JSON 状态（key/value/brace 位置）
# 需要理解 softmax 和 logit 层面的约束原理

# outlines 库（生产版）：~3 行
# import outlines
# schema = '{"name": "search", "arguments": {"query": string}}'
# generator = outlines.generate.json(model, schema)
# result = generator("今天天气怎么样")
# # result 保证是合法 JSON！

print("代码量对比:")
print("  手写 Logit Bias: ~50 lines (manual state tracking)")
print("  outlines 库:     ~3 lines (auto FSM)")
print("")
print("原理相同: 都是 logit 层面的约束 -> 保证输出格式")
print("差异: 手写版只限制单步 token, outlines 追踪完整语法状态")
print("建议: 理解原理用手写版, 生产环境用 outlines")
```

> **答案校验通过** — 2026-07-12
