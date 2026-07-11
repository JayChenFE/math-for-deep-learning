# 附录 C：全书术语一览表

> 按首次出现章节排序。每条术语包含：中文 | English | 一句话解释 | 首次出现章节。

---

## Part 1: 起点 (Ch01-02)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 浮点数 | Floating Point | IEEE 754 二进制近似存储实数的计算机格式，约 15-17 位有效数字 | 01 |
| 整数 | Integer | Python 中精确无误，无上限 | 01 |
| 布尔值 | Boolean | True/False，int 子类（True=1, False=0） | 01 |
| 列表 | List | Python 原生序列容器，元素可不同类型 | 01 |
| NumPy 数组 | ndarray | 同类型元素连续内存排列的多维数组，运算在 C 层量化执行 | 01 |
| 向量化 | Vectorization | 用数组级运算替代显式 for 循环，C 层 SIMD 并行加速 | 01 |
| 连续内存 | Contiguous Memory | 数组元素在 RAM 中紧密排列，CPU 缓存命中率高 | 01 |
| 函数 | Function | 输入→输出的映射规则，`def f(x): return ...` | 01 |
| 割线 | Secant Line | 穿过曲线上两点的直线 | 02 |
| 平均变化率 | Average Rate of Change | 区间上的总变化量 ÷ 区间长度 = 割线斜率 | 02 |
| 瞬时变化率 | Instantaneous Rate of Change | 当两点无限逼近时的割线斜率极限 = 导数 | 02 |
| 切线 | Tangent Line | 在一点的瞬时变化率方向上的直线 | 02 |
| 导数 | Derivative | 函数在某点的瞬时变化率，记为 f'(x) 或 df/dx | 02 |
| 数值微分 | Numerical Differentiation | 用 (f(x+h)−f(x))/h 近似导数，h 有限 | 02/10 |

---

## Part 2: 线性代数 (Ch03-09)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 标量 | Scalar | 0 维数字，单独一个数值 | 03 |
| 向量 | Vector | 1 维有序数组，有方向和大小 | 03 |
| 矩阵 | Matrix | 2 维数字表格，向量的"批量打包" | 03 |
| 张量 | Tensor | 3 维及以上的多维数组 | 03 |
| 轴 | Axis | 张量的"第几个维度"，axis=k 就是消灭第 k 维 | 03 |
| 形状 | Shape | 张量每个维度的大小的元组，如 (batch, seq_len, d_model) | 03 |
| 向量加法 | Vector Addition | 对应位置相加，几何上为平行四边形法则 | 04 |
| 向量数乘 | Scalar Multiplication | 每个分量乘以同一标量，方向不变长度缩放 | 04 |
| 残差连接 | Residual Connection | x = x + F(x)，梯度高速公路的工程实现 | 04/30 |
| 词嵌入 | Word Embedding | 将词映射为稠密向量的查表操作 | 04/29 |
| 点积 | Dot Product | 对应位置相乘再求和，a·b = Σaᵢbᵢ | 05 |
| 余弦相似度 | Cosine Similarity | cosθ = a·b/(||a||·||b||)，归一化到 [−1,1] 的相似度度量 | 05 |
| RAG | Retrieval-Augmented Generation | 用查询向量与知识库做点积检索 Top-K 文档作为生成上下文 | 05 |
| 矩阵乘法 | Matrix Multiplication | Aₘₓₙ·Bₙₓₚ = Cₘₓₚ，C[i,j] = Σ A[i,k]·B[k,j] | 06 |
| 转置 | Transpose | 行列互换，A[i,j] → Aᵀ[j,i] | 06 |
| 全连接层 | Fully Connected Layer | output = input @ weight.T + bias | 06 |
| 广播 | Broadcasting | 不同形状张量之间的自动形状对齐 | 06 |
| 连续批处理 | Continuous Batching | 多个用户的变长序列拼成锯齿形批量矩阵，用 mask 隔离 | 06 |
| 单位矩阵 | Identity Matrix | Iᵢⱼ = 1 if i=j else 0；"什么都不做"的变换 | 07 |
| 逆矩阵 | Inverse Matrix | A·A⁻¹ = I，"撤销变换" | 07 |
| 伪逆 | Pseudoinverse | pinv(A)：当精确解不存在时的最小二乘最佳近似 | 07 |
| 正规方程 | Normal Equation | w = (XᵀX)⁻¹Xᵀy，线性回归的闭式解 | 07 |
| 特征值 | Eigenvalue | A·v = λ·v 中的 λ—特征方向上的缩放倍数 | 08 |
| 特征向量 | Eigenvector | A·v = λ·v 中的 v—变换后方向不变的向量 | 08 |
| 特征分解 | Eigendecomposition | 将矩阵分解为特征值和特征向量：A = VΛV⁻¹ | 08 |
| 奇异值分解 | SVD | A = UΣVᵀ，旋转→拉伸→再旋转的几何三步骤 | 09 |
| 奇异值 | Singular Value | Σ 对角线上的值，按大小排序，衡量各方向的"重要度" | 09 |
| 低秩近似 | Low-Rank Approximation | 只保留前 k 个奇异值，用极少信息重构接近原始的矩阵 | 09 |
| LoRA | Low-Rank Adaptation | 用两个低秩矩阵 B(d×r)@A(r×d) 模拟权重更新 ΔW，压缩比 >99% | 09/34 |

---

## Part 3: 微积分 (Ch10-14)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 解析导数 | Analytical Derivative | 用公式推导出精确导数表达式（区别于数值微分） | 10 |
| 二阶导数 | Second Derivative | 导数的导数—描述"变化的变化"即加速度/凹凸性 | 10 |
| 子梯度 | Subgradient | 对不可导点（如 ReLU 在 x=0）任意选左右导数之一；工程上不影响训练 | 10 |
| 偏导数 | Partial Derivative | 对多变量函数按住其他变量只动一个：∂f/∂x | 11 |
| 梯度 | Gradient | 所有偏导数组成的向量 ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]；指向最陡上升方向 | 12 |
| 学习率 | Learning Rate | 梯度下降中控制步长的超参数，记为 lr 或 η | 12 |
| 链式法则 | Chain Rule | 嵌套函数的导数 = 外层导数 × 内层导数；反向传播的数学基础 | 13 |
| 计算图 | Computation Graph | 将数学表达式表示为有向无环图（节点=操作，边=数据流） | 14 |
| 前向传播 | Forward Propagation | 沿计算图从输入走到输出，计算每个节点的值 | 14 |
| 反向传播 | Backpropagation | 从输出往回走，用链式法则计算每个节点对最终输出的梯度 | 14 |
| 自动微分 | Automatic Differentiation (autograd) | 在代码执行时自动构建计算图并完成反向传播 | 14 |
| 动态计算图 | Dynamic Computation Graph | 每轮前向时重新构建计算图，适合变长输入（PyTorch 特色） | 14 |
| detach | detach() | 创建一个与原计算图无关的副本—切断梯度回传 | 14 |
| no_grad | torch.no_grad() | 整个代码块内不建计算图—用于推理，节省显存和计算 | 14 |

---

## Part 4: 概率论与统计 (Ch15-18)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 随机变量 | Random Variable | 取值由随机试验结果决定的变量 | 15 |
| 概率质量函数 | PMF (Probability Mass Function) | 离散随机变量取每个值的概率 | 15 |
| 概率密度函数 | PDF (Probability Density Function) | 连续随机变量在每点的概率密度 | 15 |
| 正态分布 | Normal Distribution | "钟形曲线"，由均值和标准差完全描述；CLT 使其成为终极分布 | 15 |
| 均匀分布 | Uniform Distribution | 所有取值等概率；权重初始化的常用分布 | 15 |
| 伯努利分布 | Bernoulli Distribution | 二选一（0 或 1，p 或 1−p）；Dropout 的数学基础 | 15 |
| 极大似然估计 | MLE (Maximum Likelihood Estimation) | 给定数据，找使数据出现概率最大的参数；等价于交叉熵最小化 | 15 |
| 采样策略 | Sampling Strategy | 从概率分布中选择具体 token 的方法：Greedy/Temperature/Top-k/Top-p | 15 |
| 温度 | Temperature | softmax(logits/T) 中的 T—T→0 退化 Greedy，T→∞ 均匀分布 | 15 |
| 条件概率 | Conditional Probability | P(A\|B)：在 B 发生的条件下 A 的概率 | 16 |
| 贝叶斯定理 | Bayes' Theorem | P(A\|B) = P(B\|A)·P(A)/P(B)；从果推因 | 16 |
| 朴素贝叶斯 | Naive Bayes | 假设特征独立的条件概率分类器；20 行代码 >90% 准确率 | 16 |
| 后验概率 | Posterior Probability | 观察到证据后更新的事件概率 | 16 |
| 期望 | Expected Value / Mean | E[X] = Σ x·P(X=x)；数据的"重心" | 17 |
| 方差 | Variance | Var[X] = E[(X−μ)²]；数据的"分散度" | 17 |
| 协方差 | Covariance | Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)]；两变量的"共变关系" | 17 |
| 协方差矩阵 | Covariance Matrix | Σ[i,j] = Cov(feature_i, feature_j)；PCA 的输入 | 17 |
| PCA | Principal Component Analysis | 协方差矩阵特征分解 → 取前 k 大特征值方向投影 → 降维 | 17 |
| 大数定律 | Law of Large Numbers | 样本均值随样本量增大收敛到期望 | 18 |
| 中心极限定理 | CLT (Central Limit Theorem) | 任何分布的样本均值都趋向正态分布；Mini-Batch SGD 的理论依据 | 18 |

---

## Part 5: 信息论 (Ch19)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 信息量 | Self-Information | I(p) = −log₂(p)；概率越低信息量越大 | 19 |
| 熵 | Entropy | H(P) = −Σ pᵢ log(pᵢ)；系统的"平均不确定度" | 19 |
| KL 散度 | KL Divergence | D_KL(P‖Q) = Σ pᵢ log(pᵢ/qᵢ)；用 Q 近似 P 多花的比特数 | 19 |
| 交叉熵 | Cross-Entropy | H(P,Q) = −Σ pᵢ log(qᵢ) = H(P) + D_KL(P‖Q)；分类的标准损失函数 | 19 |
| 困惑度 | Perplexity | PPL = exp(cross_entropy)；perp=10 = "模型从 10 个等概率词中猜" | 19 |

---

## Part 6: 数值计算 (Ch20-25)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| IEEE 754 | IEEE 754 | 浮点数的国际标准二进制存储格式 | 20 |
| float32 | 32-bit Float | 1 符号+8 指数+23 尾数；精度约 7 位有效数字 | 20 |
| float16 | 16-bit Float (Half) | 范围小(最大值 65504)，易溢出；用于推理加速 | 20 |
| bfloat16 | Brain Float 16 | 1 符号+8 指数+7 尾数；范围同 f32、精度更低；训练的"安全加速器" | 20 |
| 混合精度 | Mixed Precision | 前向/反向用 fp16，权重副本用 fp32；兼顾速度和稳定性 | 20 |
| 归一化 | Normalization | 将数据缩放/平移至特定范围的数值稳定技术 | 21 |
| BatchNorm | Batch Normalization | 跨 batch 维归一化；CNN 标配，Transformer 不用（序列变长+训练/推理不一致） | 21 |
| LayerNorm | Layer Normalization | 在特征维（last dim）内归一化；Transformer 的标配 γ*(x−μ)/σ+β | 21 |
| RMSNorm | RMS Normalization | 只除以 RMS 不减去均值；LLaMA/Mistral 的选择，比 LN 快约 15% | 21 |
| Stable Softmax | Numerically Stable Softmax | softmax(x−max(x))；一行代码防 NaN | 22 |
| Log-Sum-Exp | Log-Sum-Exp (LSE) Trick | c + log(Σ exp(x_i − c))；log(softmax) 的稳定实现 | 22 |
| Logit Bias | Logit Bias | 在 softmax 前将非法 token 的 logits 置为 −inf 以强制格式约束 | 22 |
| 梯度爆炸 | Gradient Explosion | 链式法则因子 >1 的层层相乘导致梯度指数增长 → NaN | 23 |
| 梯度消失 | Gradient Vanishing | 链式法则因子 <1 的层层相乘导致梯度指数衰减 → 参数不更新 | 23 |
| 梯度裁剪 | Gradient Clipping | 梯度范数超过阈值时等比缩放—方向不变步长受限的安全网 | 23 |
| SGD | Stochastic Gradient Descent | θ −= lr·∇L；最基础的梯度下降 | 24 |
| Momentum | Momentum / SGD + Momentum | v = β·v + (1−β)·∇L；给梯度加"惯性"抑制震荡 | 24 |
| Adam | Adaptive Moment Estimation | 一阶矩(动量) + 二阶矩(自适应学习率)；训练 Transformer 的事实标准 | 24 |
| AdamW | Adam with Decoupled Weight Decay | 将 λ·θ 从梯度中解耦，直接在参数上衰减；HuggingFace 默认 | 24 |
| Warmup | Learning Rate Warmup | 训练初将 lr 从 0 线性升至目标值，防止初期不稳定 | 24 |
| Cosine Decay | Cosine Learning Rate Decay | 余弦曲线平滑衰减到 0 | 24 |
| 梯度累积 | Gradient Accumulation | N 次 backward 后 1 次 step；模拟大 batch_size 的显存友好技巧 | 24 |
| 权重初始化 | Weight Initialization | 训练前赋予参数合理的起始值 | 25 |
| Xavier 初始化 | Xavier/Glorot Init | 保持前向/反向方差一致的初始化方案 | 25 |
| Kaiming 初始化 | Kaiming/He Init | 针对 ReLU 的改进—补偿负半轴清零导致的信息减半 | 25 |

---

## Part 7: 基石实战 (Ch26-28)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 线性回归 | Linear Regression | y = Xw + ε；最简单的参数模型 | 26 |
| 闭式解 | Closed-Form Solution | w = (XᵀX)⁻¹Xᵀy — 不需要迭代的一步到位解 | 26 |
| Sigmoid | Sigmoid Function | σ(z) = 1/(1+e^{-z})；将实数映射到 (0,1) | 27 |
| 逻辑回归 | Logistic Regression | 线性输出 + Sigmoid 激活 = 二分类概率模型 | 27 |
| 决策边界 | Decision Boundary | 模型划分不同类别的分界线/面 | 27 |
| 多层感知机 | MLP (Multi-Layer Perceptron) | 输入→隐藏(ReLU)→输出(Softmax)；最简单的深度网络 | 28 |
| 隐藏层 | Hidden Layer | 输入层和输出层之间的中间层；提取抽象特征 | 28 |
| 梯度检验 | Gradient Check | 用数值微分验证解析梯度的正确性；autograd 调试的终极手段 | 28 |

---

## Part 8: Transformer 解剖 (Ch29-32)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 自注意力 | Self-Attention | Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V；序列中每个 token 对所有 token 的加权聚合 | 29 |
| 位置编码 | Positional Encoding | 给 token 嵌入加上位置标签：Sinusoidal(固定)/Learned(可学)/RoPE(旋转) | 29 |
| 缩放点积注意力 | Scaled Dot-Product Attention | QKᵀ 除以 √d_k 防止 softmax 饱和 | 29 |
| 因果掩码 | Causal Mask | 上三角填 −inf 确保自回归生成时不能偷看未来 token | 29 |
| QKV 投影 | Q/K/V Projection | Q=X@W_Q, K=X@W_K, V=X@W_V；同一输入的三重角色分工 | 29 |
| 多头注意力 | Multi-Head Attention | 并行运行 h 个独立的 attention "头"，每头关注不同的语言关系 | 30 |
| 前馈网络 | FFN (Feed-Forward Network) | Linear→ReLU/GeLU→Linear；每个 token 独立变换，存储模型知识 | 30 |
| Pre-Norm | Pre-Layer Normalization | x = x + Sublayer(LN(x))—先归一化再进 Sublayer | 30 |
| Transformer Block | Transformer Block | Multi-Head Attention + FFN，各有 Pre-Norm + 残差连接 | 30 |
| 训练循环 | Training Loop | forward→loss→backward→clip→step→zero_grad 的完整周期 | 31 |
| zero_grad | optimizer.zero_grad() | 清空上一步的梯度—PyTorch 梯度默认累加而非覆盖 | 31 |
| train/eval | model.train() / model.eval() | 切换训练/推理模式—控制 Dropout 和 BatchNorm 行为 | 31 |
| 梯度检查点 | Gradient Checkpointing | 不保存中间激活，反向时重新前向计算—用 20-30% 时间换 50-70% 显存 | 31 |
| 自回归 | Autoregressive | 每步生成一个 token，输出作为下一步的输入 | 32 |
| KV Cache | Key-Value Cache | 推理时缓存历史 K/V，避免重复计算；O(N²) → O(N) | 32 |
| Prefill | Prefill Phase | 首次推理：输入完整 prompt → 一次性计算所有 K/V 并缓存 | 32 |
| Decode | Decode Phase | 逐个生成新 token：只算新 Q，复用缓存中的历史 K/V | 32 |
| PagedAttention | PagedAttention | 将 KV Cache 切成页，像 OS 虚拟内存一样灵活调度；vLLM 的底层秘密 | 32 |

---

## Part 9: Agent (Ch33-35)

| 中文 | English | 解释 | 章 |
|------|---------|------|----|
| 贪心解码 | Greedy Decoding | 每步选概率最高的 token—确定性输出但易陷入重复 | 33 |
| Top-k 采样 | Top-k Sampling | 只从概率前 k 高的 token 中采样；k 固定 → 截断损失 | 33 |
| 核采样 | Nucleus Sampling (Top-p) | 累加概率到 p 为止，候选集动态自适应 | 33 |
| 重复惩罚 | Repetition Penalty | 对已生成 token 的 logit 降权，防止模型说"车轱辘话" | 33 |
| 微调 | Fine-Tuning | 在预训练模型上继续训练以适应特定任务 | 34 |
| 灾难性遗忘 | Catastrophic Forgetting | 在新任务上微调时丧失预训练学到的通用能力 | 34 |
| 冻结微调 | Freeze Fine-Tuning | 只训练最后几层，底层保持预训练权重不变 | 34 |
| PEFT | Parameter-Efficient Fine-Tuning | 只训练极少参数（LoRA/Adapter/Prefix Tuning）的高效微调范式 | 34 |
| Early Stopping | Early Stopping | 验证 loss 不再下降时提前终止训练，防止过拟合 | 34 |
| Tool Calling | Function Calling | 模型输出结构化 JSON 以调用外部 API/函数/工具 | 35 |
| JSON Schema | JSON Schema | 定义工具接口的 JSON 格式规范：name, parameters, required | 35 |
| 结构化生成 | Structured Generation | 用 CFG/正则表达式在解码时强制格式匹配（Guidance/Outlines） | 35 |
| 指数退避 | Exponential Backoff | retry 间隔 = base × 2^attempt，JSON 解析失败时的重试策略 | 35 |
