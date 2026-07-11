# 附录 A：NumPy / PyTorch / SciPy 核心数学函数速查表

> 本书每一章都在使用这些函数。此表按使用频率排序，方便快速查阅"这个操作对应哪个 API"。

---

## A.1 NumPy 核心函数

| 操作用途 | NumPy API | 示例 | 出现章节 |
|---------|-----------|------|---------|
| 创建全零数组 | `np.zeros(shape)` | `np.zeros((3, 4))` | Ch03 |
| 创建全一数组 | `np.ones(shape)` | `np.ones(5)` | Ch03 |
| 创建随机正态数组 | `np.random.randn(d0, d1, ...)` | `np.random.randn(100, 10)` | Ch03 |
| 创建均匀分布数组 | `np.random.uniform(low, high, size)` | `np.random.uniform(-1, 1, (3,3))` | Ch15 |
| 等间隔采样 | `np.linspace(start, stop, n)` | `np.linspace(0, 1, 100)` | Ch01 |
| 对数等间隔采样 | `np.logspace(start, stop, n)` | `np.logspace(-3, 0, 200)` | Ch19 |
| 设置随机种子 | `np.random.seed(n)` | `np.random.seed(42)` | 全书 |
| 数组形状 | `arr.shape` | `X.shape` → `(32, 784)` | Ch03 |
| 改变形状 | `arr.reshape(new_shape)` | `X.reshape(2, 5, 8, 64)` | Ch30 |
| 转置轴 | `arr.transpose(axes)` | `Q.transpose(0, 2, 1, 3)` | Ch30 |
| 矩阵乘法 | `A @ B` 或 `np.matmul(A, B)` | `scores = Q @ K.T` | Ch06 |
| 点积 | `np.dot(a, b)` | `np.dot(user_vec, movie_vec)` | Ch05 |
| 逐元素乘 | `A * B` | `mask * attention_weights` | Ch06 |
| 求和 | `arr.sum(axis=None)` | `probs.sum(axis=-1)` | Ch03 |
| 均值 | `arr.mean(axis=None)` | `x.mean(axis=-1, keepdims=True)` | Ch17 |
| 方差 | `arr.var(axis=None)` | `x.var(axis=-1, keepdims=True)` | Ch17 |
| 标准差 | `arr.std(axis=None)` | `data.std()` | Ch17 |
| 最大值 | `arr.max(axis=None)` | `x.max(axis=-1, keepdims=True)` | Ch22 |
| 最小值 | `arr.min(axis=None)` | `x.min()` | Ch22 |
| 最大值的索引 | `np.argmax(arr, axis=None)` | `np.argmax(logits, axis=1)` | Ch33 |
| 排序的索引 | `np.argsort(arr)` | `np.argsort(probs)[::-1]` | Ch33 |
| 累积和 | `np.cumsum(arr)` | `np.cumsum(probs[sorted_idx])` | Ch33 |
| 查找插入位置 | `np.searchsorted(a, v)` | `np.searchsorted(cumsum, p)` | Ch33 |
| 指数 | `np.exp(x)` | `np.exp(x - x.max())` | Ch22 |
| 自然对数 | `np.log(x)` | `-np.log(p_correct)` | Ch19 |
| 以2为底对数 | `np.log2(x)` | `-np.log2(0.5)` → 1.0 | Ch19 |
| 正弦/余弦 | `np.sin(x)` / `np.cos(x)` | `np.sin(pos / denom)` | Ch29 |
| 拼接 | `np.vstack([a,b])` / `np.hstack([a,b])` | `np.vstack([class0, class1])` | Ch17 |
| 按列拼接 | `np.column_stack([a, b])` | `np.column_stack([x1, x2])` | Ch17 |
| 协方差矩阵 | `np.cov(X, rowvar=False)` | `np.cov(X_centered)` | Ch17 |
| 相关系数 | `np.corrcoef(x, y)` | `np.corrcoef(height, weight)` | Ch17 |
| 上三角矩阵 | `np.triu(arr, k=1)` | `np.triu(np.ones((4,4)), k=1)` | Ch29 |
| 矩阵求逆 | `np.linalg.inv(A)` | `np.linalg.inv(X.T @ X)` | Ch07 |
| 伪逆 | `np.linalg.pinv(A)` | `np.linalg.pinv(X)` | Ch07 |
| L2 范数 | `np.linalg.norm(x)` | `np.linalg.norm(path[-1])` | Ch04 |
| 特征分解 | `np.linalg.eig(A)` | `eigvals, eigvecs = np.linalg.eig(cov)` | Ch08 |
| SVD 分解 | `np.linalg.svd(A)` | `U, S, Vt = np.linalg.svd(img)` | Ch09 |
| 条件判断筛选 | `np.where(condition, a, b)` | `np.where(probs == 0, 1e-12, probs)` | Ch19 |
| 布尔索引 | `arr[condition]` | `probs[probs > 0]` | Ch19 |
| 统计非零个数 | `np.count_nonzero(arr)` | `np.count_nonzero(topk)` | Ch33 |
| 数值检验全接近 | `np.allclose(a, b, atol=1e-5)` | `np.allclose(naive, stable)` | Ch22 |
| 设置打印选项 | `np.set_printoptions(precision, suppress)` | `np.set_printoptions(suppress=True)` | Ch20 |
| 数组数据类型 | `arr.astype(dtype)` | `x.astype(np.float32)` | Ch20 |
| 无穷大常量 | `np.inf` | `mask = np.ones(...) * -np.inf` | Ch29 |
| 列/行堆叠后分拆 | `np.split(arr, n, axis)` | — | Ch21 |

---

## A.2 PyTorch 核心函数

| 操作用途 | PyTorch API | 示例 | 出现章节 |
|---------|------------|------|---------|
| 创建张量 | `torch.tensor(data)` | `torch.tensor([1.0, 2.0, 3.0])` | Ch14 |
| 创建随机张量 | `torch.randn(size)` | `torch.randn(2, 5, 512)` | Ch14 |
| 设置随机种子 | `torch.manual_seed(n)` | `torch.manual_seed(42)` | 全书 |
| 需要梯度的张量 | `torch.tensor(data, requires_grad=True)` | `x = torch.tensor(2.0, requires_grad=True)` | Ch14 |
| 自动求导 | `loss.backward()` | `y.backward()` | Ch14 |
| 梯度值 | `tensor.grad` | `x.grad` | Ch14 |
| 全连接层 | `nn.Linear(in_features, out_features)` | `nn.Linear(784, 256)` | Ch25 |
| ReLU 激活 | `torch.relu(x)` 或 `F.relu(x)` | `F.relu(X @ W1 + b1)` | Ch28 |
| 交叉熵损失 | `F.cross_entropy(logits, labels)` | `F.cross_entropy(logits_t, y_t)` | Ch19 |
| Softmax | `F.softmax(x, dim=-1)` | `F.softmax(logits, dim=-1)` | Ch22 |
| Log-Softmax | `F.log_softmax(x, dim=-1)` | `F.log_softmax(logits, dim=-1)` | Ch22 |
| 梯度裁剪 | `nn.utils.clip_grad_norm_(params, max_norm)` | `clip_grad_norm_(model.parameters(), 1.0)` | Ch23 |
| 优化器 SGD | `optim.SGD(params, lr, momentum)` | `optim.SGD(model.parameters(), lr=0.01)` | Ch24 |
| 优化器 Adam | `optim.Adam(params, lr, betas)` | `optim.Adam(model.parameters(), lr=1e-3)` | Ch24 |
| 优化器 AdamW | `optim.AdamW(params, lr, weight_decay)` | HuggingFace Trainer 默认 | Ch24 |
| 学习率调度 | `optim.lr_scheduler.CosineAnnealingLR` | — | Ch24 |
| 参数遍历 | `model.named_parameters()` | `for name, p in model.named_parameters():` | Ch34 |
| 参数数量 | `sum(p.numel() for p in model.parameters())` | `sum(p.numel() for p in model.parameters())` | Ch34 |
| 保存模型 | `torch.save(model.state_dict(), path)` | — | Ch34 |
| 加载模型 | `model.load_state_dict(torch.load(path))` | — | Ch34 |
| 训练模式 | `model.train()` | Dropout 生效 | Ch31 |
| 推理模式 | `model.eval()` | Dropout 关闭 | Ch31 |
| 不记录梯度 | `with torch.no_grad():` | 推理时不建计算图 | Ch14 |
| 切断梯度 | `tensor.detach()` | 从计算图分离 | Ch14 |
| 梯度检查点 | `torch.utils.checkpoint.checkpoint(fn, x)` | — | Ch31 |
| 张量形状 | `tensor.shape` 或 `tensor.size()` | `X.shape` → `torch.Size([32,784])` | Ch03 |
| 改变形状 | `tensor.view(new_shape)` / `tensor.reshape(...)` | `Q.view(B, L, H, Dh)` | Ch30 |
| 转置 | `tensor.transpose(dim0, dim1)` | `K.transpose(-2, -1)` | Ch29 |
| 索引选取 | `tensor[torch.arange(n), indices]` | `probs[torch.arange(N), labels]` | Ch19 |

---

## A.3 SciPy / 其他常用函数

| 操作用途 | API | 示例 | 出现章节 |
|---------|-----|------|---------|
| 误差函数 | `math.erf(x)` | `normal_cdf` 的计算 | Ch15 |
| 浮点比较 | `math.isclose(a, b)` | `math.isclose(0.1 + 0.2, 0.3)` | Ch01 |
| 浮点比较容差法 | `abs(a - b) < 1e-8` | 通用模式 | Ch01 |
| 高等精度计算 | `decimal.Decimal('0.1')` | 教学演示 | Ch20 |
| Matplotlib 画线 | `plt.plot(x, y)` | 通用 | Ch01 |
| Matplotlib 散点 | `plt.scatter(x, y)` | PCA 降维可视化 | Ch17 |
| Matplotlib 柱状图 | `plt.bar(x, height)` | 概率分布 | Ch15 |
| Matplotlib 热力图 | `plt.imshow(matrix)` | 协方差矩阵 | Ch17 |
| Matplotlib 等高线 | `plt.contour(X, Y, Z)` | 梯度下降轨迹 | Ch12 |
| Matplotlib 箭袋图 | `plt.quiver(x, y, dx, dy)` | 向量可视化 | Ch04 |
| Matplotlib 动画 | `animation.FuncAnimation(fig, update, frames)` | 割线 → 切线 | Ch02 |

---

## A.4 HuggingFace 常用 API

| 操作用途 | API | 出现章节 |
|---------|-----|---------|
| 加载分词器 | `AutoTokenizer.from_pretrained('gpt2')` | Ch34 |
| 加载预训练模型 | `AutoModelForCausalLM.from_pretrained('gpt2')` | Ch34 |
| 分词 | `tokenizer(text, padding=True, truncation=True, return_tensors='pt')` | Ch34 |
| 解码 | `tokenizer.decode(token_ids)` | Ch34 |
| LoRA 配置 | `LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"])` | Ch34 |
| 注入 LoRA | `get_peft_model(model, lora_config)` | Ch34 |
| 合并 LoRA | `model.merge_and_unload()` | Ch34 |
| 生成文本 | `model.generate(input_ids, max_new_tokens, temperature, top_p)` | Ch33 |
| KV Cache 生成 | `model.generate(use_cache=True)` | Ch32 |
