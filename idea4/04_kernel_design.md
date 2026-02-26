# 核函数设计研究规划：超越 L2 指数核

**日期：** 2026-02-24
**优先级：** ⭐⭐⭐

---

## 1. 当前核函数设计分析

### 1.1 论文的核函数

论文使用基于 L2 距离的指数核：

$$k(x, y) = \exp\left(-\frac{\|x - y\|_2}{\tau}\right)$$

通过双向 softmax 归一化实现（几何平均归一化）：

$$W_\text{pos}(x_i, y^+_j) = \frac{\sqrt{A_\text{pos}(i,j) \cdot A_\text{neg\_sum}(i)}}{\text{normalization}}$$

其中 $A_\text{pos}(i,j) = \text{softmax}_j(-\text{dist}(x_i, y^+_j)/\tau)$，交叉加权保证反对称性。

### 1.2 消融实验数据

**核归一化消融（appendix_exp.tex）：**

| 归一化方式 | FID |
|---------|-----|
| x 和 y 轴双向 softmax（默认） | **8.46** |
| 仅 y 轴 softmax | 8.92 |
| 无归一化 | 10.54 |

**多温度消融：**

| 温度配置 | FID |
|--------|-----|
| τ = 0.02 | 10.62 |
| τ = 0.05 | 8.67 |
| τ = 0.2 | 8.96 |
| {0.02, 0.05, 0.2} 组合 | **8.46** |

**关键观察：**
- 双向 softmax 比单向好 5.4%，比无归一化好 24%
- 多温度组合与最优单温度持平，说明不同尺度的信息互补
- 论文只测试了等权组合，未探索自适应权重

### 1.3 理论框架的开放性

论文明确指出：

> "Our framework supports a broad class of functions $\mathcal{K}$, as long as $V=0$ when $p=q$."

这意味着任何满足反对称性条件的核函数都可以使用。当前的 L2 指数核是 mean-shift 的标准选择，但并非最优。

---

## 2. 方向一：学习型核函数

### 2.1 动机

固定的 L2 指数核假设特征空间中的距离是各向同性的，但实际上：
- 不同语义维度的重要性不同（颜色 vs 形状 vs 纹理）
- 不同类别的样本可能需要不同的距离度量
- 训练过程中最优的距离度量可能随生成质量变化

### 2.2 参数化核函数设计

**方案 A：马氏距离核（Mahalanobis Kernel）**

$$k_M(x, y) = \exp\left(-\frac{(x-y)^T M (x-y)}{\tau}\right)$$

其中 $M$ 是正定矩阵，可以学习。特殊情况：
- $M = I$：退化为 L2 核
- $M = \Sigma^{-1}$（协方差矩阵的逆）：白化核，消除特征相关性

**方案 B：神经网络核（Neural Kernel）**

用小网络参数化核函数：

```python
class NeuralKernel(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        # 将两个特征映射到标量相似度
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的相似度
        )

    def forward(self, x, y):
        # x: (B, D), y: (N, D)
        # 广播计算所有对
        x_exp = x.unsqueeze(1).expand(-1, y.size(0), -1)  # (B, N, D)
        y_exp = y.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, N, D)
        pairs = torch.cat([x_exp, y_exp], dim=-1)          # (B, N, 2D)
        return self.net(pairs).squeeze(-1)                  # (B, N)
```

**反对称性保证：** 神经网络核需要额外约束来保证 $V=0$ when $p=q$。一种方法是使用对称核：$k(x,y) = k(y,x)$，然后通过交叉加权机制自动保证反对称性。

**方案 C：注意力核（Attention Kernel）**

将核函数设计为 scaled dot-product attention：

$$k_\text{attn}(x, y) = \frac{\exp(q(x)^T k(y) / \sqrt{d})}{\sum_{y'} \exp(q(x)^T k(y') / \sqrt{d})}$$

其中 $q(\cdot)$ 和 $k(\cdot)$ 是可学习的线性投影。这本质上是将 Transformer 的注意力机制用作核函数。

### 2.3 训练稳定性分析

学习型核函数面临的主要风险：**核函数坍缩**——网络可能学到将所有样本映射到相同的相似度，使 V 退化。

缓解方案：
1. **正则化**：对核函数的输出分布施加熵正则化，防止过于尖锐或过于平坦
2. **停止梯度**：在计算核函数时，对"键"侧施加 stopgrad（类似 MoCo）
3. **辅助损失**：添加对比学习损失，确保核函数保持判别性

---

## 3. 方向二：多温度自适应权重

### 3.1 当前设计的局限

论文使用三个温度 $\tau \in \{0.02, 0.05, 0.2\}$ 等权求和：

$$V = \frac{1}{3}(V_{0.02} + V_{0.05} + V_{0.2})$$

等权假设三个尺度的信息同等重要，但实际上：
- 训练初期：生成分布与真实分布差距大，全局结构（高温）更重要
- 训练后期：细节对齐阶段，局部结构（低温）更重要

### 3.2 自适应权重设计

**方案 A：可学习静态权重**

$$V = \sum_\tau w_\tau V_\tau, \quad w_\tau = \text{softmax}(\alpha_\tau)$$

其中 $\alpha_\tau$ 是可学习参数，通过梯度下降优化。

**方案 B：条件自适应权重**

根据当前生成样本的质量动态调整权重：

```python
class AdaptiveTemperatureWeights(nn.Module):
    def __init__(self, feat_dim, num_temps=3):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_temps),
        )

    def forward(self, x_feat):
        # x_feat: (B, D) 当前生成样本的特征
        weights = F.softmax(self.weight_net(x_feat.mean(0)), dim=-1)  # (num_temps,)
        return weights
```

**方案 C：训练阶段自适应**

根据训练进度（epoch 比例）线性插值权重：

$$w_\tau(t) = w_\tau^\text{init} + t \cdot (w_\tau^\text{final} - w_\tau^\text{init})$$

其中 $t \in [0, 1]$ 是训练进度，初始权重偏向高温（全局），最终权重偏向低温（局部）。

### 3.3 实验设计

| 配置 | 权重策略 | 预期 FID |
|------|---------|---------|
| 基准 | 等权 {1/3, 1/3, 1/3} | 8.46 |
| 可学习静态 | softmax(α) | ~8.0 |
| 条件自适应 | 基于特征的动态权重 | ~7.5 |
| 训练阶段自适应 | 线性插值 | ~7.8 |

---

## 4. 方向三：非对称核函数

### 4.1 理论分析

当前核函数是对称的：$k(x, y) = k(y, x)$。非对称核可以捕捉更丰富的结构：

- **方向性相似度**：$k(x, y) \neq k(y, x)$ 允许"x 被 y 吸引"和"y 被 x 吸引"的强度不同
- **类比**：在信息检索中，查询-文档相似度通常是非对称的

**反对称性验证：**

对于非对称核 $k(x, y) \neq k(y, x)$，反对称性条件 $V_{p,q} = -V_{q,p}$ 是否仍然成立？

$$V_{p,q}(x) = \mathbb{E}_p[k(x, y^+)(y^+ - x)] - \mathbb{E}_q[k(x, y^-)(y^- - x)]$$

$$V_{q,p}(x) = \mathbb{E}_q[k(x, y^+)(y^+ - x)] - \mathbb{E}_p[k(x, y^-)(y^- - x)]$$

当 $p = q$ 时，$V_{p,q}(x) + V_{q,p}(x) = 0$ 当且仅当：

$$\mathbb{E}_p[k(x, y)(y - x)] = \mathbb{E}_p[k(x, y)(y - x)]$$

这对任意核函数都成立（两边相同）。因此，**非对称核不破坏反对称性**。

### 4.2 非对称核的设计

**方案：查询-键分离的核函数**

$$k_\text{asym}(x, y) = \frac{\exp(q(x)^T k(y) / \sqrt{d})}{\sum_{y'} \exp(q(x)^T k(y') / \sqrt{d})}$$

其中 $q(\cdot)$ 和 $k(\cdot)$ 是不同的线性投影（类似 Transformer 的 Q 和 K）。

这允许"从 x 的视角看 y 的相似度"与"从 y 的视角看 x 的相似度"不同，可能更好地捕捉语义层次结构。

---

## 5. 方向四：基于 MMD 的核函数

### 5.1 理论联系

论文附录指出，当使用归一化核时，漂移场 V 与最大均值差异（MMD）的梯度有深刻联系：

$$\text{MMD}^2(p, q) = \mathbb{E}_{x,x' \sim p}[k(x,x')] - 2\mathbb{E}_{x \sim p, y \sim q}[k(x,y)] + \mathbb{E}_{y,y' \sim q}[k(y,y')]$$

V 的方向与 $-\nabla_q \text{MMD}^2(p, q)$ 一致（在适当的归一化下）。

### 5.2 MMD 最优核的选择

MMD 的统计检验能力依赖于核函数的选择。最优核（在某种意义下）是：

$$k^*(x, y) = \frac{p(x)p(y)}{q(x)q(y)} \cdot k_\text{base}(x, y)$$

其中 $p/q$ 是密度比。这在实践中不可直接计算，但可以用密度比估计（Density Ratio Estimation）近似：

```python
class DensityRatioKernel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 密度比估计网络：输出 log(p(x)/q(x))
        self.ratio_net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.base_kernel = RBFKernel(tau=0.05)

    def forward(self, x, y):
        log_ratio_x = self.ratio_net(x)  # (B, 1)
        log_ratio_y = self.ratio_net(y)  # (N, 1)
        base_k = self.base_kernel(x, y)  # (B, N)
        # 加权核
        weight = (log_ratio_x + log_ratio_y.T).exp()  # (B, N)
        return base_k * weight
```

---

## 6. 实验优先级与资源估算

### 6.1 推荐实验顺序

**第一阶段（低成本，1–2 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| K0 | 多温度自适应权重（可学习静态） | ~50 | FID < 8.0 |
| K1 | 白化预处理（马氏距离核） | ~30 | FID < 8.0 |

**第二阶段（中等成本，2–4 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| K2 | 注意力核（Q-K 分离） | ~150 | FID < 7.5 |
| K3 | 条件自适应温度权重 | ~100 | FID < 7.5 |

**第三阶段（高成本，4–8 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| K4 | 神经网络核（完整参数化） | ~300 | FID < 7.0 |
| K5 | 密度比加权核 | ~400 | FID < 6.5 |

### 6.2 风险评估

- **最低风险**：多温度自适应权重（K0）——只增加 3 个可学习参数，不改变核函数结构
- **中等风险**：注意力核（K2）——改变核函数形式，但有 Transformer 的成功经验支撑
- **最高风险**：神经网络核（K4）——完全参数化，训练稳定性难以保证

### 6.3 与特征编码器改进的协同

核函数改进与特征编码器改进是互补的：
- 更好的特征编码器 → 特征空间更有判别性 → 任何核函数都更有效
- 更好的核函数 → 更准确的 V 估计 → 对特征编码器质量的要求降低

建议先完成特征编码器改进（idea2/01），再在更好的特征空间上测试核函数改进。

---

## 参考文献

- Gretton et al., "A Kernel Two-Sample Test," JMLR 2012
- Schölkopf et al., "Learning with Kernels," MIT Press 2002
- Sugiyama et al., "Density Ratio Estimation in Machine Learning," Cambridge 2012
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017
- Comaniciu & Meer, "Mean Shift: A Robust Approach toward Feature Space Analysis," TPAMI 2002
