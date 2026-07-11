"""Validate Chapter 35: Agent tool calling & structured generation."""
import numpy as np
import json

np.random.seed(42)

# 35.1 JSON Schema definition
tool_schema = {
    "name": "search",
    "description": "Search the web for information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "num_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    }
}
schema_str = json.dumps(tool_schema, ensure_ascii=False, indent=2)
assert '"name": "search"' in schema_str
assert '"query"' in schema_str
print("35.1 JSON Schema: valid")

# 35.2 Logit Bias for JSON mode
vocab = {"{": 0, '"': 1, ":": 2, "}": 3, "a": 4, "b": 5, "action": 6, "search": 7}
rev_vocab = {v: k for k, v in vocab.items()}

def apply_json_bias(logits, legal_tokens, illegal_value=-np.inf):
    """将非法 token 的 logits 置为 -inf，只允许合法 JSON token。"""
    mask = np.ones(len(logits), dtype=bool)
    mask[legal_tokens] = False
    logits_biased = logits.copy().astype(float)
    logits_biased[mask] = illegal_value
    return logits_biased

# 测试：只有 {, ", :, } 合法
logits = np.array([2.0, 1.0, 0.5, -1.0, 3.0, 2.5, 4.0, 5.0])  # 最后两个是 action/search
json_tokens = [0, 1, 2, 3]  # {, ", :, }
biased = apply_json_bias(logits, json_tokens)

# 稳定 softmax
biased_stable = biased - biased.max()
probs = np.exp(biased_stable) / np.exp(biased_stable).sum()
assert probs[4] == 0.0  # 'a' 概率应为 0
assert probs[0] > 0      # '{' 有正概率
assert abs(probs.sum() - 1.0) < 1e-10
print("35.2 Logit Bias: illegal tokens zeroed, probabilities valid")

# 35.3 ensure_json decorator with retry
def ensure_json(generate_fn, max_retries=3, base_delay=0.1):
    """装饰器：自动检测 JSON 输出，失败则重试。"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            output = generate_fn(*args, **kwargs)
            try:
                json.loads(output)
                return output, attempt  # success
            except json.JSONDecodeError:
                if attempt < max_retries:
                    # Exponential backoff + lower temperature retry
                    import time
                    time.sleep(base_delay * (2 ** attempt))
        return output, max_retries  # all retries exhausted
    return wrapper

# 测试：第三次才返回合法 JSON
attempt_count = [0]
def mock_generate():
    attempt_count[0] += 1
    if attempt_count[0] < 3:
        return 'not json at all'
    return '{"action": "search", "query": "weather"}'

safe_gen = ensure_json(mock_generate, max_retries=3)
result, retries = safe_gen()
assert retries == 2  # 0-indexed: succeeds on 3rd call (attempt index 2)
assert json.loads(result)["action"] == "search"
print("35.3 ensure_json: retried 2x, succeeded on 3rd attempt")

# 35.3 Temperature vs error rate (simulated)
def simulate_generation(logits, temperature):
    """模拟单步采样：temperature 控制概率分布。"""
    scaled = logits / max(temperature, 1e-8)
    scaled = scaled - scaled.max()
    probs = np.exp(scaled) / np.exp(scaled).sum()
    return probs

logits_example = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
probs_t01 = simulate_generation(logits_example, 0.1)
probs_t10 = simulate_generation(logits_example, 1.0)
probs_t20 = simulate_generation(logits_example, 2.0)

# 低温度 → 更集中（峰值更高）
assert probs_t01.max() > probs_t10.max()
assert probs_t10.max() > probs_t20.max()
# 高温度 → 更分散（熵更大）
entropy = lambda p: -np.sum(p * np.log(p + 1e-12))
assert entropy(probs_t20) > entropy(probs_t10) > entropy(probs_t01)
print(f"35.3 Temperature: T=0.1 max={probs_t01.max():.3f}, T=2.0 max={probs_t20.max():.3f}")

print("\nCh35 OK -- JSON Schema, Logit Bias, ensure_json, Temperature behavior")
