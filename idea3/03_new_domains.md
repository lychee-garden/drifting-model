# 新领域扩展研究规划：Drifting Models 的领域无关性验证

**日期：** 2026-02-24
**优先级：** ⭐⭐⭐⭐

---

## 1. 理论基础：为什么 Drifting Models 具有领域无关性？

### 1.1 核心算法的抽象性

Drifting Models 的核心是漂移场 $V$：

$$V(x_i) = \sum_j W_\text{pos}(x_i, y^+_j) \cdot y^+_j - \sum_k W_\text{neg}(x_i, y^-_k) \cdot y^-_k$$

这个公式对数据类型没有任何假设——$x, y^+, y^-$ 可以是任意向量空间中的元素。反对称性条件 $V_{p,q} = -V_{q,p}$ 同样与数据类型无关，只依赖于核函数的对称性。

### 1.2 机器人控制的成功案例

论文已经在机器人控制任务（Push-T, Block Pushing）上验证了方法的领域无关性：

> "We directly compute drifting loss on the raw representations for control, using no feature space."

关键发现：
- **低维状态空间不需要特征编码器**：机器人状态（关节角度、末端位置）维度低（~20维），核函数天然有效
- **超越 100-NFE 扩散策略**：单步生成的 Drifting Models 在控制任务上超过了需要 100 步迭代的 Diffusion Policy
- **时序一致性**：通过条件生成（以历史观测为条件）实现了动作序列的时序一致性

### 1.3 各领域的核心挑战

| 领域 | 数据空间 | 核心挑战 | 特征编码器需求 |
|------|---------|---------|-------------|
| 视频生成 | 时序图像序列 | 时序一致性、运动建模 | 时空特征编码器 |
| 分子生成 | 3D 原子坐标 + 化学键 | SE(3) 等变性、化学有效性 | 等变图神经网络 |
| 文本生成 | 离散 token 序列 | 离散空间的漂移场定义 | 语言模型编码器 |
| 音频生成 | 时序波形/频谱 | 时序结构、感知质量 | 音频特征编码器 |

---

## 2. 方向一：视频生成

### 2.1 问题定义

视频生成的目标是学习分布 $p(\mathbf{x}_{1:T})$，其中 $\mathbf{x}_t \in \mathbb{R}^{C \times H \times W}$ 是第 $t$ 帧。

**Drifting Models 的自然扩展：**

将视频视为时空体积（spatiotemporal volume）$\mathbf{X} \in \mathbb{R}^{T \times C \times H \times W}$，直接在视频空间定义漂移场：

$$V(\mathbf{X}_i) = \sum_j W_\text{pos}(\mathbf{X}_i, \mathbf{Y}^+_j) \cdot \mathbf{Y}^+_j - \sum_k W_\text{neg}(\mathbf{X}_i, \mathbf{Y}^-_k) \cdot \mathbf{Y}^-_k$$

### 2.2 时序一致性的保证

**挑战：** 逐帧独立生成会导致时序不一致（闪烁、运动不连贯）。

**方案 A：时空特征编码器**

使用 3D 卷积或时空 Transformer 提取视频特征，使核函数能够感知时序结构：

```python
class VideoFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的视频理解模型
        self.backbone = VideoMAE_ViT_L()  # 或 InternVideo2

    def forward(self, video):
        # video: (B, T, C, H, W)
        feats = self.backbone(video)  # (B, D) 时空全局特征
        return F.normalize(feats, dim=-1)
```

**方案 B：自回归条件生成**

以前 $k$ 帧为条件，生成下一帧：

$$V(\mathbf{x}_t | \mathbf{x}_{1:t-1}) = \sum_j W_\text{pos}(\mathbf{x}_t, \mathbf{y}^+_{t,j} | \mathbf{x}_{1:t-1}) \cdot \mathbf{y}^+_{t,j} - \ldots$$

这与论文的条件生成框架（类别条件）完全兼容，只需将条件从类别标签扩展为历史帧序列。

**方案 C：联合时空漂移**

同时在帧内（空间）和帧间（时间）计算漂移损失：

$$\mathcal{L}_\text{video} = \mathcal{L}_\text{spatial} + \lambda \mathcal{L}_\text{temporal}$$

其中 $\mathcal{L}_\text{temporal}$ 使用光流或帧差特征作为时序一致性约束。

### 2.3 实验设计

**数据集：** UCF-101（101类，13,320视频，16帧，64×64）

**基准对比：**
- DDPM-based video generation（FVD ~2500）
- VideoLDM（FVD ~550）
- 目标：FVD < 500

**实验阶段：**
1. 短视频（16帧，64×64）：验证基本可行性
2. 中等视频（32帧，128×128）：测试时序一致性
3. 长视频（64帧，256×256）：测试运动连贯性

---

## 3. 方向二：分子生成

### 3.1 问题定义

分子生成的目标是学习 3D 分子结构的分布 $p(\mathcal{M})$，其中 $\mathcal{M} = \{(\mathbf{r}_i, z_i)\}_{i=1}^N$ 包含原子坐标 $\mathbf{r}_i \in \mathbb{R}^3$ 和原子类型 $z_i$。

**核心约束：SE(3) 等变性**

分子的物理性质在旋转、平移、反射下不变。漂移场必须满足：

$$V(R\mathbf{r} + \mathbf{t}) = R \cdot V(\mathbf{r}) \quad \forall R \in SO(3), \mathbf{t} \in \mathbb{R}^3$$

### 3.2 等变漂移场设计

**方案：使用等变图神经网络作为特征编码器**

```python
class EquivariantMolEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # EGNN 或 SE(3)-Transformer
        self.gnn = EGNN(in_node_nf=5, hidden_nf=256, out_node_nf=256)

    def forward(self, coords, atom_types, edges):
        # coords: (N_atoms, 3) — 原子坐标
        # atom_types: (N_atoms,) — 原子类型
        node_feats, _ = self.gnn(atom_types, coords, edges)
        # 全局池化得到分子级特征
        mol_feat = node_feats.mean(dim=0)  # (256,)
        return F.normalize(mol_feat, dim=-1)
```

**漂移场的等变性保证：**

若特征编码器 $\phi$ 是等变的，则核函数 $k(\phi(\mathcal{M}_i), \phi(\mathcal{M}_j))$ 是旋转不变的，漂移场 $V$ 自然满足等变性。

### 3.3 低维状态空间的优势

与图像生成不同，分子的状态空间维度较低（典型分子 ~50 个原子，150 维坐标）。类似机器人控制任务，可能不需要复杂的特征编码器，直接在原子坐标空间计算漂移损失。

**实验设计：**
- 数据集：QM9（13.4万小分子，最多 9 个重原子）
- 评估指标：有效性（Validity）、唯一性（Uniqueness）、新颖性（Novelty）
- 基准：DDPM（Validity=85.6%）、EDM（Validity=91.9%）
- 目标：Validity > 95%，同时保持高唯一性

---

## 4. 方向三：文本生成

### 4.1 离散空间的核心挑战

文本是离散的 token 序列，而 Drifting Models 的漂移场定义在连续空间中。直接在 token 空间定义漂移场面临：

1. **离散性**：token 之间没有自然的距离度量（"cat" 和 "dog" 的距离是多少？）
2. **变长序列**：不同文本的长度不同，无法直接计算向量差
3. **组合爆炸**：词汇表大小 ~50,000，序列长度 ~512，状态空间极大

### 4.2 连续嵌入空间的漂移

**方案 A：在 token 嵌入空间定义漂移**

将文本映射到连续嵌入空间，在嵌入空间中计算漂移场，然后通过最近邻解码回 token：

```python
# 训练时：在嵌入空间计算漂移
x_embed = embedding_layer(x_tokens)  # (B, T, D)
V = compute_V(x_embed, y_pos_embed, y_neg_embed, phi)

# 推理时：从噪声嵌入生成，然后解码
z_embed = torch.randn(B, T, D)
x_embed_gen = generator(z_embed, condition)
x_tokens_gen = nearest_neighbor_decode(x_embed_gen, embedding_layer.weight)
```

**方案 B：扩散语言模型框架下的漂移**

结合 MDLM（Masked Diffusion Language Model）的思路，在掩码预测框架下定义漂移：
- 正样本：真实文本的 BERT 风格掩码预测
- 负样本：生成文本的掩码预测
- 漂移场：在掩码 token 的概率分布空间中定义

**方案 C：潜在空间文本生成**

使用 VAE 将文本压缩到连续潜在空间（类似 Optimus、DELLA），在潜在空间中应用 Drifting Models：
- 优势：直接复用图像生成的框架
- 缺点：依赖文本 VAE 的质量，且文本 VAE 的训练本身是难题

### 4.3 评估与基准

- 数据集：WikiText-103、OpenWebText
- 评估指标：困惑度（Perplexity）、MAUVE 分数、生成多样性
- 基准：GPT-2（PPL=18.3）、MDLM（PPL=7.0）
- 目标：PPL < 10，MAUVE > 0.9

---

## 5. 方向四：音频生成

### 5.1 音频的特殊性

音频信号具有以下特点：
- **时序结构强**：音频的时序依赖比图像更强（语音的音素序列、音乐的节拍结构）
- **感知非线性**：人耳对频率的感知是对数尺度的（Mel 频谱）
- **多尺度结构**：从毫秒级的音色到秒级的旋律，跨越多个时间尺度

### 5.2 频谱空间的漂移

**方案：在 Mel 频谱空间定义漂移**

将音频转换为 Mel 频谱图（2D 时频表示），然后应用图像生成的框架：

```python
# 音频 → Mel 频谱图
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256
)(audio)  # (B, 80, T)

# 在频谱图空间应用 Drifting Models
V = compute_V(mel_spec, y_pos_mel, y_neg_mel, phi_audio)
```

**特征编码器选择：**
- EnCodec（Meta）：音频神经编解码器，提供高质量的音频潜在表示
- AudioMAE：音频领域的 MAE 预训练模型
- CLAP：对比语言-音频预训练，类似 CLIP 的音频版本

### 5.3 实验设计

- 数据集：LJSpeech（单说话人 TTS）、VCTK（多说话人）
- 评估指标：MOS（Mean Opinion Score）、UTMOS（自动 MOS 估计）
- 基准：WaveGrad（MOS=4.2）、DiffWave（MOS=4.4）
- 目标：MOS > 4.3，单步生成

---

## 6. 优先级分析与资源估算

### 6.1 各领域的可行性评估

| 领域 | 技术难度 | 理论风险 | 预期收益 | 综合优先级 |
|------|---------|---------|---------|----------|
| 视频生成 | 中 | 低（直接扩展） | 高 | ⭐⭐⭐⭐ |
| 分子生成 | 中 | 低（低维空间） | 高（应用价值） | ⭐⭐⭐⭐ |
| 音频生成 | 低 | 低（频谱图≈图像） | 中 | ⭐⭐⭐ |
| 文本生成 | 高 | 高（离散空间） | 高（影响力） | ⭐⭐⭐ |

### 6.2 推荐实验顺序

**第一阶段（验证可行性，2–4 周）：**

| 实验 | 内容 | GPU 小时 |
|------|------|---------|
| D0 | 音频生成（LJSpeech，Mel 频谱） | ~100 |
| D1 | 分子生成（QM9，直接坐标空间） | ~50 |

**第二阶段（核心实验，4–8 周）：**

| 实验 | 内容 | GPU 小时 |
|------|------|---------|
| D2 | 视频生成（UCF-101，16帧，64×64） | ~300 |
| D3 | 分子生成（GEOM-Drug，更大分子） | ~200 |

**第三阶段（高难度，8–16 周）：**

| 实验 | 内容 | GPU 小时 |
|------|------|---------|
| D4 | 文本生成（连续嵌入空间方案） | ~500 |
| D5 | 视频生成（长视频，256×256） | ~800 |

### 6.3 最低成本验证策略

**分子生成是最低成本的新领域验证：**
- QM9 数据集小（13.4万分子），训练快
- 低维状态空间（~150维），不需要特征编码器
- 评估指标明确（有效性、唯一性）
- 预期 GPU 小时：~50

若分子生成成功，可以作为"Drifting Models 的领域无关性"的有力证据，支撑后续更大规模的视频和文本生成实验。

---

## 7. 统一框架设计

### 7.1 领域无关的 Drifting Models 接口

```python
class DriftingModel(nn.Module):
    """
    领域无关的 Drifting Models 基类。
    子类只需实现 feature_encoder 和 generator。
    """
    def __init__(self, feature_encoder, generator, kernel_temps=[0.02, 0.05, 0.2]):
        super().__init__()
        self.phi = feature_encoder  # 可以是 None（低维空间）
        self.G = generator
        self.temps = kernel_temps

    def compute_drift_loss(self, x_gen, y_pos, y_neg):
        if self.phi is not None:
            x_feat = self.phi(x_gen)
            y_pos_feat = self.phi(y_pos)
            y_neg_feat = self.phi(y_neg)
        else:
            x_feat, y_pos_feat, y_neg_feat = x_gen, y_pos, y_neg

        V = compute_V(x_feat, y_pos_feat, y_neg_feat, self.temps)
        loss = (x_gen * V.detach()).sum(dim=-1).mean()
        return loss

    def generate(self, noise, condition=None):
        return self.G(noise, condition)
```

### 7.2 各领域的实例化

```python
# 图像生成
image_model = DriftingModel(
    feature_encoder=MultiScaleFeatureEncoder(),
    generator=DriftDiT_Small()
)

# 分子生成
molecule_model = DriftingModel(
    feature_encoder=None,  # 直接在坐标空间
    generator=EGNN_Generator()
)

# 视频生成
video_model = DriftingModel(
    feature_encoder=VideoMAE_Encoder(),
    generator=VideoTransformer()
)
```

---

## 参考文献

- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023
- Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D," ICML 2022
- Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces," NeurIPS 2021
- Kong et al., "DiffWave: A Versatile Diffusion Model for Audio Synthesis," ICLR 2021
- Ho et al., "Video Diffusion Models," NeurIPS 2022
- Saharia et al., "Imagen Video: High Definition Video Generation with Diffusion Models," 2022
