# CFG 机制改进研究规划：重新理解训练时引导

**日期：** 2026-02-24
**优先级：** ⭐⭐⭐

---

## 1. 当前 CFG 设计分析

### 1.1 论文的 CFG 实现

论文的 CFG 与扩散模型的 CFG 在本质上不同。论文定义：

$$\tilde{q}(\cdot|c) \triangleq (1-\gamma)\,q_\theta(\cdot|c) + \gamma\,p_{\text{data}}(\cdot|\varnothing)$$

即在负样本中混入无条件真实数据（$\varnothing$ 表示无条件），混合比例 $\gamma$ 作为条件输入到网络（称为 $\alpha$）。

**实现细节（appendix_impl.tex）：**

```python
# 训练时：以概率 gamma 将负样本替换为无条件真实数据
if random.random() < gamma:
    y_neg = unconditional_real_samples  # 来自全局队列
else:
    y_neg = conditional_generated_samples  # 来自生成器

# alpha 作为条件输入到生成器
alpha = gamma  # 当前的混合比例
x = generator(z, class_label, alpha)
```

### 1.2 关键实验发现

**CFG 消融（appendix_exp.tex）：**

> "with our best model (L/2), the optimal FID is achieved at α=1.0, which is often regarded as 'w/o CFG' in diffusion-/flow-based models."

| 模型规模 | 最优 α | FID |
|---------|-------|-----|
| B/2（小模型） | ~1.5 | ~3.5 |
| L/2（大模型） | **1.0** | **1.54** |

**核心发现：最好的模型在推理时根本不需要 CFG（α=1.0）。**

这意味着：
1. 训练时的 CFG 信号（无条件真实数据作为额外负样本）已经被充分吸收进模型权重
2. 推理时的外插（α>1.0）对大模型反而有害
3. CFG 在 Drifting Models 中的作用机制与扩散模型根本不同

### 1.3 与扩散模型 CFG 的本质区别

| 维度 | 扩散模型 CFG | Drifting Models CFG |
|------|------------|-------------------|
| 作用时机 | 推理时（外插） | 训练时（负样本构造） |
| 数学形式 | $\epsilon_\theta(x,c) + \alpha(\epsilon_\theta(x,c) - \epsilon_\theta(x,\varnothing))$ | $\tilde{q} = (1-\gamma)q + \gamma p_\varnothing$ |
| 最优 α | 通常 7–10 | 1.0（大模型） |
| 物理意义 | 增强条件信号 | 提供额外排斥信号 |

---

## 2. 方向一：负样本构造策略的系统性探索

### 2.1 当前设计的局限

论文只测试了"无条件真实数据"作为额外负样本。但负样本的选择空间远不止于此。

**负样本的作用：** 负样本 $y^-$ 提供排斥信号，将生成样本推离"错误"区域。不同类型的负样本定义了不同的"错误"：
- 无条件真实数据：推离所有真实数据（防止过拟合到特定类别）
- 其他类别的生成样本：推离错误类别（增强类别区分度）
- 风格迁移样本：推离错误风格（增强风格一致性）
- 低质量生成样本：推离低质量区域（提升生成质量下限）

### 2.2 候选负样本类型

**类型 A：跨类别生成样本（Cross-class Negatives）**

$$y^-_\text{cross} = G(z, c'), \quad c' \neq c$$

将其他类别的生成样本作为负样本，强制生成器学习更清晰的类别边界。

```python
# 训练时：从其他类别采样负样本
other_classes = [c' for c' in range(num_classes) if c' != current_class]
neg_class = random.choice(other_classes)
y_neg_cross = generator(z_neg, neg_class, alpha=1.0)
```

**类型 B：增强负样本（Augmented Negatives）**

对真实数据施加强增强（颜色抖动、随机裁剪、噪声），生成"接近但不完全正确"的负样本：

$$y^-_\text{aug} = \text{StrongAug}(x_\text{real})$$

这迫使生成器学习更精确的数据分布，而不仅仅是粗略的语义结构。

**类型 C：历史生成样本（Historical Negatives）**

使用训练早期（质量较低）的生成样本作为负样本，类似课程学习：

$$y^-_\text{hist} = G_{\theta_{t-k}}(z), \quad k > 0$$

这提供了"当前模型应该超越的基准"，可能加速收敛。

**类型 D：插值负样本（Interpolated Negatives）**

在真实样本和生成样本之间插值：

$$y^-_\text{interp} = \lambda x_\text{real} + (1-\lambda) G(z), \quad \lambda \in (0, 1)$$

这在数据流形附近提供密集的排斥信号，可能改善生成样本的细节质量。

### 2.3 实验设计

| 实验 | 负样本类型 | 预期效果 | GPU 小时 |
|------|---------|---------|---------|
| C0 | 基准（无条件真实数据） | FID=8.46 | — |
| C1 | 跨类别生成样本 | 类别区分度提升 | ~100 |
| C2 | 增强负样本 | 细节质量提升 | ~100 |
| C3 | 混合策略（C0 + C1 + C2） | 综合提升 | ~150 |

---

## 3. 方向二：条件信号的增强

### 3.1 当前条件机制

论文将 $\alpha$（CFG 混合比例）作为标量条件输入到生成器，通过 adaLN-Zero 调制：

```python
# 生成器接受 (z, class_label, alpha) 三个输入
# alpha 通过 MLP 嵌入后与 class_label 嵌入相加
alpha_embed = self.alpha_mlp(alpha.unsqueeze(-1))  # (B, D)
class_embed = self.class_embed(class_label)         # (B, D)
condition = alpha_embed + class_embed               # (B, D)
```

### 3.2 更丰富的条件信号

**方案 A：训练进度条件（Training Progress Conditioning）**

将当前训练步数（归一化到 [0,1]）作为额外条件，让生成器感知自身的训练阶段：

$$x = G(z, c, \alpha, t/T)$$

这允许生成器在训练初期（$t/T$ 小）采用更保守的策略，在训练后期（$t/T$ 大）采用更精细的策略。

**方案 B：质量感知条件（Quality-Aware Conditioning）**

将当前批次的平均漂移场强度 $\|V\|$ 作为条件，让生成器感知当前的生成质量：

$$x = G(z, c, \alpha, \|V\|_\text{batch})$$

$\|V\|$ 大意味着生成分布与真实分布差距大，生成器应该做出更大的调整。

**方案 C：类别原型条件（Class Prototype Conditioning）**

除了类别标签，还将该类别的特征原型（真实样本的平均特征）作为条件：

$$x = G(z, c, \alpha, \bar{\phi}_c)$$

其中 $\bar{\phi}_c = \mathbb{E}_{x \sim p(\cdot|c)}[\phi(x)]$ 是类别 $c$ 的特征中心。

这提供了更丰富的类别信息，超越了离散标签的表达能力。

### 3.3 无分类器引导的替代方案

**方案：对比引导（Contrastive Guidance）**

在推理时，使用对比损失的梯度引导生成：

$$x_\text{guided} = x + \lambda \nabla_x \log \frac{p(c|x)}{p(c)}$$

其中 $p(c|x)$ 由特征编码器估计。这是一种无需训练额外分类器的引导方法。

---

## 4. 方向三：无条件生成的改进

### 4.1 当前无条件生成的问题

论文的无条件生成使用全局队列（1000 个样本），但：
- 全局队列混合了所有类别，无条件负样本的多样性可能不足
- 队列大小（1000）是固定的，未经充分消融

### 4.2 改进方案

**方案 A：类别平衡的无条件队列**

确保全局队列中每个类别的样本数量相等：

```python
class BalancedUnconditionalQueue:
    def __init__(self, num_classes, samples_per_class=100):
        self.queues = {c: deque(maxlen=samples_per_class) for c in range(num_classes)}

    def sample(self, n):
        # 从每个类别均匀采样
        per_class = n // self.num_classes
        samples = []
        for c in range(self.num_classes):
            samples.extend(random.sample(list(self.queues[c]), per_class))
        return torch.stack(samples)
```

**方案 B：动态队列大小**

根据训练进度动态调整队列大小：
- 训练初期：小队列（100），快速更新，反映当前生成质量
- 训练后期：大队列（10000），稳定估计，减少方差

**方案 C：在线负样本（Online Negatives）**

放弃队列，直接在每个训练步中生成负样本：

```python
# 每步生成新的负样本（无需队列）
with torch.no_grad():
    z_neg = torch.randn(N_neg, ...)
    y_neg = generator(z_neg, labels_neg, alpha=1.0)
```

优点：负样本始终反映当前生成质量，无队列延迟
缺点：每步需要额外的前向传播，计算成本增加 ~50%

---

## 5. 方向四：推理时 CFG 的重新设计

### 5.1 为什么大模型不需要推理时 CFG？

论文发现最优 α=1.0（即不使用推理时 CFG）。可能的解释：

1. **训练时 CFG 已经足够**：无条件真实数据作为负样本，已经将条件信息充分编码进模型权重
2. **推理时外插破坏平衡**：α>1.0 相当于在推理时破坏了训练时学到的平衡，导致质量下降
3. **模型容量足够**：L/2 模型足够大，能够直接学习条件分布，不需要推理时的额外引导

### 5.2 新的推理时引导策略

**方案 A：特征空间引导（Feature Space Guidance）**

在推理时，使用特征编码器的梯度引导生成：

$$x_{t+1} = x_t + \eta \nabla_{x_t} \log k(\phi(x_t), \bar{\phi}_c)$$

其中 $\bar{\phi}_c$ 是目标类别的特征原型。这是一种"软引导"，不破坏训练时学到的分布。

**方案 B：多步精炼（Multi-step Refinement）**

虽然 Drifting Models 是单步生成，但可以在推理时进行少量精炼步骤：

```python
# 单步生成
x = generator(z, c, alpha=1.0)

# 可选：少量精炼步骤（2-3步）
for _ in range(num_refine_steps):
    V = compute_V(x, real_samples_c, generated_samples)
    x = x + refine_lr * V
```

这保留了单步生成的效率优势，同时允许少量后处理改善质量。

**方案 C：温度缩放（Temperature Scaling）**

在推理时对生成器的输出施加温度缩放：

$$x_\text{scaled} = x / T, \quad T \in (0, 1]$$

低温度使生成样本更接近训练分布的高密度区域，可能提升质量但降低多样性。

---

## 6. 实验优先级与资源估算

### 6.1 推荐实验顺序

**第一阶段（低成本，1–2 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| G0 | 跨类别负样本（C1） | ~100 | FID < 8.0 |
| G1 | 类别平衡无条件队列 | ~50 | FID < 8.2 |

**第二阶段（中等成本，2–4 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| G2 | 混合负样本策略（C0+C1+C2） | ~150 | FID < 7.5 |
| G3 | 类别原型条件 | ~150 | FID < 7.5 |

**第三阶段（高成本，4–8 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| G4 | 在线负样本（无队列） | ~300 | FID < 7.0 |
| G5 | 多步精炼（2-3步） | ~200 | FID < 6.5 |

### 6.2 核心假设与验证

**假设 1：** 跨类别负样本能提升类别条件生成的精确度
- 验证方法：计算每个类别的 FID，观察类别间混淆是否减少

**假设 2：** 训练时 CFG 的作用是提供额外的排斥多样性，而非增强条件信号
- 验证方法：对比"无条件真实数据"vs"其他类别生成样本"作为负样本的效果

**假设 3：** 大模型不需要推理时 CFG 是因为模型容量足够，而非训练时 CFG 的特殊性
- 验证方法：在小模型（B/2）上测试不同负样本策略，观察是否仍需推理时 CFG

### 6.3 与其他方向的协同

CFG 机制改进与特征编码器改进（idea2/01）有直接协同：
- 更好的特征编码器 → 更准确的类别原型 $\bar{\phi}_c$ → 类别原型条件更有效
- 更好的特征编码器 → 跨类别负样本的特征区分度更高 → 跨类别排斥更精准

建议在完成特征编码器改进后，再系统测试 CFG 机制改进。

---

## 7. 理论分析：CFG 在 Drifting Models 中的信息论解释

### 7.1 无条件负样本的作用

从信息论角度，无条件真实数据作为负样本，相当于在漂移场中引入了一个"全局排斥项"：

$$V_\text{cfg}(x|c) = V(x|c) - \gamma \cdot V_\text{uncond}(x)$$

其中 $V_\text{uncond}(x) = \mathbb{E}_{y \sim p_\text{data}}[k(x,y)(y-x)]$ 是无条件吸引场。

这个全局排斥项的作用是：**防止条件生成器过度拟合到无条件分布**，保持条件信息的有效性。

### 7.2 最优 α=1.0 的信息论解释

当模型足够大时，生成器能够直接学习条件分布 $p(x|c)$，无需推理时的额外引导。此时：

$$G(z, c, \alpha=1.0) \approx \text{sample from } p(x|c)$$

推理时的 CFG 外插（α>1.0）相当于在已经正确的条件分布上施加额外偏置，反而引入误差。

这与扩散模型的情况不同：扩散模型的生成器学习的是分数函数 $\nabla_x \log p(x|c)$，推理时的 CFG 是对分数函数的外插，可以增强条件信号而不破坏分布。

---

## 参考文献

- Ho & Salimans, "Classifier-Free Diffusion Guidance," NeurIPS Workshop 2021
- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS 2021
- Kang et al., "Scaling up GANs for Text-to-Image Synthesis," CVPR 2023
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022
- Chen et al., "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning," ICLR 2023
