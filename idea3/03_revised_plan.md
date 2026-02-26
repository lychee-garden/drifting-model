# 新领域扩展方案（修订版）

**基于 rebuttal 的全面修订**
**日期：** 2026-02-25
**状态：** 修订版 v2

---

## 修订说明

本方案完整吸收审稿人五项批评及作者回应，核心修订如下：

1. **重新定位核心论点**：从"领域无关性"改为"1-NFE 约束下的跨模态适用性"
2. **聚焦单一领域深入**：放弃四领域并行的浅层调研，优先深入分子生成和视频生成
3. **修正分子生成的平移等变性错误**：采用基于相对坐标的 V 构造
4. **重新设计视频生成方案**：使用 patch-level token 提供帧级约束
5. **明确区分数学层与工程层**：删除"数学上统一"的过度主张

---

## 1. 核心论点重新定位

### 1.1 从"领域无关性"到"1-NFE 跨模态适用性"

原始提案的"领域无关性"论点是平凡命题——MMD、SVGD 等方法同样具有此性质。

**修订后的核心论点：**

> **在 1-NFE（单步推理）约束下，Drifting Models 能否在多个领域实现与多步方法竞争的生成质量？**

这一论点的独特性在于：
- 扩散模型在视频、分子、音频领域均需要 50–1000 步迭代
- Drifting Models 的训练时漂移机制天然支持单步推理
- 在推理效率敏感的场景（实时控制、在线生成），1-NFE 优势显著

### 1.2 机器人控制的证明力分析

论文已验证：20 维状态空间中，1-NFE Drifting Policy 超越 100-NFE Diffusion Policy。

**审稿人批评**：20 维对高维领域无证明力。

**修订立场**：
- 20 维验证了"低维空间无需特征编码器"的可行性
- 高维领域（视频、分子）需要特征编码器，是独立的研究问题
- **补充高维验证**：在 RL 策略空间（维度 ≥ 200）验证，作为低维到高维的过渡

---

## 2. 优先方向一：分子生成（最高优先级）

### 2.1 为什么分子生成是最优先方向

| 优势 | 说明 |
|------|------|
| 低维状态空间 | QM9 分子 ~50 原子，150 维坐标，接近机器人控制的成功场景 |
| 明确评估指标 | 有效性、唯一性、新颖性，无需主观评估 |
| 计算成本低 | QM9 数据集小（13.4 万分子），训练快（~50 GPU 小时） |
| 1-NFE 价值清晰 | 分子构象采样需要大量样本，单步生成效率优势显著 |

### 2.2 平移等变性的数学修正（关键修订）

**原始提案的错误**：声称"等变特征编码器使漂移场自动满足 SE(3) 等变性"，但忽略了平移等变性。

**审稿人的正确推导**：若分子平移 $\mathbf{t}$，则：
$$y^+_j \to y^+_j + \mathbf{t}, \quad V \to V + \mathbf{t} \cdot \left(\sum_j W^+_j - \sum_k W^-_k\right)$$

由于 $\sum W^+_j = \sum W^-_k = 1$（各自归一化），差值不为零，平移等变性不满足。

**修正方案：基于相对坐标的 V 构造**

$$V = \sum_{j} W^+_j (y^+_j - \bar{y}^+) - \sum_{k} W^-_k (y^-_k - \bar{y}^-)$$

其中 $\bar{y}^+ = \sum_j W^+_j y^+_j$，$\bar{y}^- = \sum_k W^-_k y^-_k$ 分别为正负样本的加权质心。

**平移等变性证明**：若所有坐标平移 $\mathbf{t}$：
$$y^+_j - \bar{y}^+ \to (y^+_j + \mathbf{t}) - (\bar{y}^+ + \mathbf{t}) = y^+_j - \bar{y}^+$$

差值不变，$V$ 不变，平移等变性严格成立。✓

**旋转等变性**：若旋转矩阵 $R \in SO(3)$：
$$y^+_j - \bar{y}^+ \to R(y^+_j - \bar{y}^+), \quad V \to RV$$

$V$ 在旋转下协变，旋转等变性成立。✓

### 2.3 等变生成器设计

```python
class MolecularDriftingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 等变特征编码器（用于核函数计算）
        self.encoder = EGNN(
            in_node_nf=5,      # 原子类型 one-hot
            hidden_nf=256,
            out_node_nf=256,
            n_layers=6
        )
        # 等变生成器（直接在坐标空间生成）
        self.generator = EquivariantGenerator(
            noise_dim=64,
            hidden_nf=256,
            n_layers=8
        )

    def compute_V(self, x_coords, x_types, y_pos_coords, y_pos_types,
                  y_neg_coords, y_neg_types):
        # 提取特征（用于核函数）
        x_feat = self.encoder(x_coords, x_types)       # (B, 256)
        yp_feat = self.encoder(y_pos_coords, y_pos_types)  # (N_pos, 256)
        yn_feat = self.encoder(y_neg_coords, y_neg_types)  # (N_neg, 256)

        # 计算核权重（旋转不变）
        W_pos, W_neg = compute_kernel_weights(x_feat, yp_feat, yn_feat)

        # 基于相对坐标计算 V（平移等变）
        y_pos_center = (W_pos.unsqueeze(-1) * y_pos_coords).sum(dim=1)  # 加权质心
        y_neg_center = (W_neg.unsqueeze(-1) * y_neg_coords).sum(dim=1)

        V = ((W_pos.unsqueeze(-1) * (y_pos_coords - y_pos_center.unsqueeze(1))).sum(dim=1)
           - (W_neg.unsqueeze(-1) * (y_neg_coords - y_neg_center.unsqueeze(1))).sum(dim=1))
        return V  # (B, N_atoms, 3)，等变向量场
```

### 2.4 实验设计

**数据集**：QM9（13.4 万小分子，最多 9 个重原子）

**评估指标**：
- 有效性（Validity）：RDKit 化学有效性检验
- 唯一性（Uniqueness）：生成样本中不重复的比例
- 新颖性（Novelty）：不在训练集中的比例

**竞争基准**（同等 NFE 约束）：

| 方法 | NFE | Validity |
|------|-----|---------|
| DDPM（EDM） | 1000 | 91.9% |
| Flow Matching | 100 | 93.2% |
| **目标：Drifting Policy** | **1** | **>90%** |

**实验阶段**：
1. **M1**（~50 GPU 小时）：QM9，直接坐标空间，无特征编码器，验证基本可行性
2. **M2**（~100 GPU 小时）：QM9，EGNN 特征编码器，测试等变性修正效果
3. **M3**（~200 GPU 小时）：GEOM-Drug（更大分子），测试扩展性

---

## 3. 优先方向二：视频生成（次优先级）

### 3.1 单步生成与时序一致性的矛盾分析

**审稿人批评**：单步生成必须在一次前向传播中同时生成所有帧，时序一致性难以保证。

**修订立场**：时序一致性约束应编码进漂移场的构造中，而非依赖自回归展开。

**核心思路**：将时序一致性作为漂移场的**软约束**，通过帧级特征的时序相关性来实现。

### 3.2 帧级特征约束方案（替换全局特征）

**原始提案的问题**：VideoMAE 的 [CLS] token 是全局聚合特征，无法提供逐帧约束。

**修订方案**：使用 VideoMAE 的 **patch-level token** 作为帧级特征。

```python
class VideoFrameLevelEncoder(nn.Module):
    """
    使用 VideoMAE 的 patch-level token 提供帧级特征约束。
    """
    def __init__(self):
        super().__init__()
        self.backbone = VideoMAE_ViT_L()  # 预训练 VideoMAE

    def forward(self, video):
        # video: (B, T, C, H, W)
        # VideoMAE 输出 patch tokens: (B, T*N_patches, D)
        patch_tokens = self.backbone.get_patch_tokens(video)

        # 按帧聚合：每帧的 patch tokens 平均
        T = video.shape[1]
        N_patches_per_frame = patch_tokens.shape[1] // T
        frame_tokens = patch_tokens.view(
            patch_tokens.shape[0], T, N_patches_per_frame, -1
        ).mean(dim=2)  # (B, T, D)

        return frame_tokens  # 保留时序维度

    def get_temporal_consistency_loss(self, gen_video, real_video):
        """
        计算帧间特征一致性损失，作为时序约束。
        """
        gen_feats = self.forward(gen_video)   # (B, T, D)
        real_feats = self.forward(real_video)  # (B, T, D)

        # 帧间差分特征（捕捉运动）
        gen_motion = gen_feats[:, 1:] - gen_feats[:, :-1]   # (B, T-1, D)
        real_motion = real_feats[:, 1:] - real_feats[:, :-1]  # (B, T-1, D)

        # 运动一致性损失
        return F.mse_loss(gen_motion, real_motion.detach())
```

### 3.3 联合损失设计

$$\mathcal{L}_\text{video} = \mathcal{L}_\text{drift} + \lambda_\text{temp} \mathcal{L}_\text{temporal}$$

其中：
- $\mathcal{L}_\text{drift}$：标准漂移损失（在视频级特征空间计算）
- $\mathcal{L}_\text{temporal}$：帧间运动一致性损失（软约束）
- $\lambda_\text{temp}$：权衡系数，通过消融确定

### 3.4 放弃自回归方案

原始提案的自回归方案（逐帧生成）将 NFE 从 1 增加到 T，放弃了 1-NFE 优势。**修订版完全删除此方案**，专注于单步视频生成。

### 3.5 竞争基准重新定位

**放弃**"FVD < 500"的绝对目标（落后 SOTA 两个数量级）。

**修订目标**：在相同参数量级的**单步视频生成方法**中建立基线。

| 方法 | NFE | FVD（UCF-101） |
|------|-----|--------------|
| DDPM-based | 1000 | ~2500 |
| VideoLDM | 50 | ~550 |
| **目标：Drifting Model（单步）** | **1** | **<1000（初步）** |

初步目标是建立单步视频生成的可行性基线，而非与大规模多步方法竞争。

### 3.6 实验设计

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| V1 | UCF-101，16帧，64×64，全局特征 | ~200 | FVD < 2000（可行性） |
| V2 | UCF-101，16帧，64×64，帧级特征 | ~300 | FVD < 1500（时序改善） |
| V3 | UCF-101，16帧，64×64，+时序损失 | ~300 | FVD < 1000（目标） |

---

## 4. 降级方向：音频生成

### 4.1 为什么音频是低风险验证

- Mel 频谱图是 2D 时频表示，可直接复用图像生成框架
- 音频特征编码器（EnCodec、AudioMAE）已有成熟实现
- 数据集小（LJSpeech 24 小时），训练快

### 4.2 简化方案

```python
# 音频 → Mel 频谱图 → 图像生成框架
mel_spec = MelSpectrogram(n_mels=80)(audio)  # (B, 80, T)
# 视为 80×T 的"图像"，直接应用 Drifting Models
```

**目标**：在 LJSpeech 上验证可行性（MOS > 4.0），作为"领域扩展"的低成本证据。

---

## 5. 暂缓方向：文本生成

### 5.1 为什么暂缓文本生成

审稿人的批评完全成立：
- 嵌入空间漂移不保证解码后文本连贯（语义鸿沟）
- 掩码扩散方案已偏离 Drifting Models 框架
- 文本 VAE 构造本身是开放问题

**修订决定**：文本生成方向暂缓，待分子生成和视频生成验证成功后再考虑。

若未来重启文本生成，核心问题需先解决：
> 如何在离散 token 空间定义有意义的"距离"，使核函数能够捕捉语义相似性？

---

## 6. 统一框架的重新定位

### 6.1 删除"数学上统一"的主张

原始提案的统一框架（`DriftingModel` 基类）声称数学上统一，但：
- 分子生成（`feature_encoder=None`）：V 在原始坐标空间计算，需要等变性约束
- 视频生成（`feature_encoder=VideoMAE`）：V 在语义特征空间计算，无等变性要求

两者数学性质本质不同，不能用同一接口抹平。

### 6.2 修订后的两层框架

**数学层**（各领域独立）：

| 领域 | V 的计算空间 | 等变性要求 | 特征编码器 |
|------|------------|----------|----------|
| 机器人控制 | 原始状态空间 | 无 | 无 |
| 分子生成 | 原子坐标空间 | SE(3) 等变 | EGNN（可选） |
| 视频生成 | 语义特征空间 | 无 | VideoMAE |
| 音频生成 | Mel 频谱空间 | 无 | AudioMAE |

**工程层**（统一接口，仅用于代码复用）：

```python
class DriftingModelBase(nn.Module):
    """
    工程层统一接口，仅保证训练循环和推理接口的一致性。
    各领域的数学性质（等变性、特征空间）由子类负责。
    """
    def compute_drift_loss(self, x_gen, y_pos, y_neg):
        raise NotImplementedError  # 子类实现，各领域数学不同

    def generate(self, noise, condition=None):
        raise NotImplementedError  # 子类实现
```

---

## 7. 实验优先级与资源估算（修订版）

| 优先级 | 实验 | 领域 | GPU 小时 | 成功标准 |
|--------|------|------|---------|---------|
| 1（最高） | M1 | 分子（QM9，无编码器） | ~50 | Validity > 85% |
| 1 | M2 | 分子（QM9，EGNN） | ~100 | Validity > 90% |
| 2 | D0 | 音频（LJSpeech） | ~100 | MOS > 4.0 |
| 2 | V1 | 视频（UCF-101，基线） | ~200 | FVD < 2000 |
| 3 | M3 | 分子（GEOM-Drug） | ~200 | Validity > 85% |
| 3 | V2-V3 | 视频（帧级特征+时序损失） | ~600 | FVD < 1000 |
| 暂缓 | — | 文本生成 | — | — |

---

## 参考文献

- Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D," ICML 2022
- Satorras et al., "E(n) Equivariant Graph Neural Networks," ICML 2021
- Tong et al., "Improving and Generalizing Flow-Matching for 3D Molecule Generation," 2023
- Wang et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training," NeurIPS 2022
- Kong et al., "DiffWave: A Versatile Diffusion Model for Audio Synthesis," ICLR 2021
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023
