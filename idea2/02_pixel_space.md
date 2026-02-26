# 像素空间生成研究规划：突破潜在空间依赖

**日期：** 2026-02-24
**优先级：** ⭐⭐⭐⭐

---

## 1. 问题分析：像素空间 vs 潜在空间

### 1.1 当前实验数据对比

| 空间 | 编码器配置 | 训练轮数 | FID |
|------|-----------|----------|-----|
| 潜在空间（SD-VAE 32×32×4） | latent-MAE w=640 + cls ft | 1280 | **1.54** |
| 像素空间（256×256×3） | MAE w=640 + cls ft | 640 | 1.61 |
| 像素空间（256×256×3） | MAE w=256 | 100 | 32.11 |
| 像素空间（256×256×3） | MAE w=640 + cls ft | 100 | 9.35 |
| 像素空间（256×256×3） | MAE w=640 + cls ft + ConvNeXt-V2 | 100 | **3.70** |

**关键发现：**
- 像素空间 FID=1.61 vs 潜在空间 FID=1.54，差距仅 4.5%
- 但像素空间只训练了 640 轮（潜在空间训练了 1280 轮）
- 论文明确指出："Due to limited time, we train pixel-space models for 640 epochs (vs. the latent counterpart's 1280); **we expect that longer training would yield further improvements.**"
- 多编码器组合（ResNet-MAE + ConvNeXt-V2）在 100 轮时就达到 FID=3.70，潜力巨大

### 1.2 像素空间的核心挑战

**维度诅咒（Curse of Dimensionality）：**

像素空间维度 = 256×256×3 = 196,608，而潜在空间维度 = 32×32×4 = 4,096。

在高维空间中，核函数的 flat kernel 问题更严重：

$$k(x_i, x_j) = \exp\left(-\frac{\|\phi(x_i) - \phi(x_j)\|_2}{\tau}\right)$$

若 $\phi$ 的判别能力不足，高维像素空间中所有样本对的距离趋于集中（Johnson-Lindenstrauss 现象），核函数退化为常数的风险更高。

**VAE 信息瓶颈的缺失：**

潜在空间生成依赖 SD-VAE 的编码器将高频噪声压缩掉，生成器只需学习语义结构。像素空间生成器必须同时学习高频细节和低频语义，任务难度更高。

**特征编码器的适配问题：**

当前 latent-MAE 是在潜在空间（32×32×4）上预训练的，直接用于像素空间（256×256×3）需要重新设计。像素空间的 MAE 预训练需要更大的模型和更多计算。

### 1.3 像素空间生成的独特价值

1. **端到端纯粹性**：不依赖 VAE，避免了 VAE 引入的重建误差和信息瓶颈
2. **理论完整性**：直接在数据空间验证 Drifting Models 的理论，无需假设 VAE 的质量
3. **应用灵活性**：不需要预训练 VAE，可以直接应用于新数据集
4. **与扩散模型的公平对比**：许多扩散模型基准是在像素空间建立的

---

## 2. 方向一：延长训练轮数

### 2.1 理论依据

论文的像素空间实验只训练了 640 轮，而潜在空间训练了 1280 轮。从特征编码器的消融数据可以看出，训练轮数对 FID 有显著影响：

| 训练轮数 | FID（latent-MAE w=640） |
|----------|------------------------|
| 192 | 6.30 |
| 1280 | 4.28 |

从 192→1280 轮，FID 提升 32%，且曲线尚未饱和。像素空间从 640→1280 轮，预期有类似幅度的提升。

### 2.2 实验设计

**基准实验（E0）：**
- 配置：像素空间，MAE w=640 + cls ft，640 轮（复现论文结果）
- 预期 FID：~1.61

**延长训练（E1）：**
- 配置：像素空间，MAE w=640 + cls ft，1280 轮
- 预期 FID：1.3–1.5（基于潜在空间的缩放规律估计）
- GPU 小时：~400（假设与潜在空间训练成本相当）

**超长训练（E2）：**
- 配置：像素空间，MAE w=640 + cls ft，2560 轮
- 预期 FID：1.1–1.3（若曲线仍未饱和）
- GPU 小时：~800

### 2.3 成功标准

- E1 达到 FID < 1.5：证明像素空间可以与潜在空间持平
- E2 达到 FID < 1.3：证明像素空间在充分训练后可以超越潜在空间

---

## 3. 方向二：更强的像素空间特征编码器

### 3.1 当前编码器的局限

当前像素空间使用的 MAE（ResNet-basic, w=640）是在像素空间自监督预训练的，但：
- ResNet 架构的感受野有限，难以捕捉全局语义
- MAE 的重建目标偏向低级纹理，语义聚类性弱于对比学习
- 宽度 640 已经是当前设计的上限，进一步扩展需要改变架构

### 3.2 候选编码器

**DINOv2 (ViT-L/14) 用于像素空间：**

- 输入：256×256 图像（与 DINOv2 期望的 224×224 接近，可直接使用或轻微裁剪）
- 特征：CLS token（1024 维）或 patch tokens（256 个 patch × 1024 维）
- 优势：大规模自蒸馏训练，特征空间各向同性好，ImageNet 线性探测 86.1%
- 接入方式：

```python
class DINOv2PixelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # 256x256 → 18x18 patches (14px stride)
        # 使用 CLS token 作为全局特征

    def forward(self, x):
        # x: (B, 3, 256, 256)
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats = self.backbone(x_resized)  # (B, 1024) CLS token
        return F.normalize(feats, dim=-1)
```

**SAM (Segment Anything Model) 编码器：**

- 特点：专为像素级理解设计，特征具有强空间对应性
- 优势：对局部结构的感知能力强，适合像素空间的细粒度漂移
- 缺点：主要用于分割任务，语义聚类性未经充分验证

**ConvNeXt-V2-L（论文已验证有效）：**

论文像素空间实验中，加入 ConvNeXt-V2 后 FID 从 9.35 → 3.70（100 轮），提升 60%。这是已验证的有效方向。

- 接入方式：与 ResNet-MAE 并行，特征拼接后计算漂移损失
- 预期 FID（100 轮）：~3.70（复现）
- 预期 FID（640 轮）：~1.8–2.2

### 3.3 多编码器融合策略

论文已经证明 ResNet-MAE + ConvNeXt-V2 的组合有效。可以进一步探索：

**策略 A：特征拼接（Feature Concatenation）**

$$\phi_\text{fused}(x) = \text{normalize}([\phi_\text{MAE}(x); \phi_\text{ConvNeXt}(x); \phi_\text{DINOv2}(x)])$$

三个编码器的特征拼接后归一化，计算统一的核函数。

**策略 B：损失加权求和（Loss Aggregation）**

$$\mathcal{L}_\text{drift} = \lambda_1 \mathcal{L}^\text{MAE}_\text{drift} + \lambda_2 \mathcal{L}^\text{ConvNeXt}_\text{drift} + \lambda_3 \mathcal{L}^\text{DINOv2}_\text{drift}$$

每个编码器独立计算漂移损失，加权求和。权重 $\lambda_i$ 可以固定（均等）或学习。

**策略 C：层次化融合（Hierarchical Fusion）**

- 低层编码器（ConvNeXt 浅层）：捕捉纹理和边缘
- 高层编码器（DINOv2 CLS token）：捕捉语义结构
- 分别计算不同尺度的漂移损失，类似多温度策略

---

## 4. 方向三：多尺度 Patch 策略

### 4.1 当前设计的局限

当前像素空间的特征提取使用固定的 patch 大小（16×16），这导致：
- 细粒度纹理（如毛发、纹理）被平均掉
- 全局结构（如整体姿态）需要多层感受野才能捕捉

### 4.2 多尺度 Patch 设计

**并行多尺度策略：**

```python
class MultiScalePixelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 4×4 patch：捕捉细粒度纹理（256×256 → 64×64 patches）
        self.fine_encoder = PatchEncoder(patch_size=4, embed_dim=256)
        # 8×8 patch：中等尺度结构（256×256 → 32×32 patches）
        self.mid_encoder = PatchEncoder(patch_size=8, embed_dim=512)
        # 16×16 patch：粗粒度语义（256×256 → 16×16 patches）
        self.coarse_encoder = PatchEncoder(patch_size=16, embed_dim=1024)

    def forward(self, x):
        fine_feats = self.fine_encoder(x)    # (B, 64*64, 256)
        mid_feats = self.mid_encoder(x)      # (B, 32*32, 512)
        coarse_feats = self.coarse_encoder(x) # (B, 16*16, 1024)

        # 全局池化后拼接
        fine_global = fine_feats.mean(dim=1)
        mid_global = mid_feats.mean(dim=1)
        coarse_global = coarse_feats.mean(dim=1)

        return F.normalize(torch.cat([fine_global, mid_global, coarse_global], dim=-1), dim=-1)
```

**与多温度核函数的对应关系：**

多尺度 patch 与多温度核函数在概念上是对偶的：
- 小 patch + 低温度 τ：局部细粒度结构
- 大 patch + 高温度 τ：全局粗粒度结构

两者可以协同设计，形成更完整的多尺度漂移场。

### 4.3 实验设计

| 配置 | Patch 尺度 | 预期 FID（640 轮） |
|------|-----------|-----------------|
| 基准 | 16×16 only | ~1.61 |
| 双尺度 | 8×8 + 16×16 | ~1.4 |
| 三尺度 | 4×4 + 8×8 + 16×16 | ~1.2 |

---

## 5. 方向四：生成器架构优化

### 5.1 当前像素空间生成器

当前使用 DiT（Diffusion Transformer）架构，但针对像素空间的优化不足：
- 全局自注意力的计算复杂度为 O(N²)，256×256 图像的 patch 数量 N=256（16×16 patch），计算量可接受
- 但若使用更小的 patch（如 8×8），N=1024，全局注意力计算量增加 16 倍

### 5.2 局部-全局混合注意力

**设计方案：**

```python
class LocalGlobalDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size=8):
        super().__init__()
        # 局部窗口注意力：O(N × window_size²)
        self.local_attn = WindowAttention(hidden_size, num_heads, window_size)
        # 全局稀疏注意力：每隔 stride 个 patch 采样
        self.global_attn = StridedAttention(hidden_size, num_heads, stride=4)
        # 前馈网络
        self.ffn = SwiGLU(hidden_size)

    def forward(self, x, c):
        x = x + self.local_attn(x)   # 局部细节
        x = x + self.global_attn(x)  # 全局结构
        x = x + self.ffn(x, c)       # 条件调制
        return x
```

**优势：**
- 局部注意力捕捉纹理细节，全局注意力保证整体一致性
- 计算复杂度从 O(N²) 降至 O(N × window_size² + N × stride)

### 5.3 U-Net 风格的跳跃连接

DiT 是纯 Transformer 架构，缺乏 U-Net 的多尺度跳跃连接。对于像素空间生成，引入跳跃连接可以：
- 保留高频细节（通过浅层特征的直接传递）
- 减少深层网络的信息瓶颈

---

## 6. 实验优先级与资源估算

### 6.1 推荐实验顺序

**第一阶段（低成本，1–2 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| P0 | 复现论文像素空间结果（FID=1.61） | ~200 | FID < 1.7 |
| P1 | 延长训练至 1280 轮 | ~400 | FID < 1.5 |

**第二阶段（中等成本，2–4 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| P2 | 加入 ConvNeXt-V2 编码器（复现 3.70，然后延长训练） | ~300 | FID < 1.4 |
| P3 | DINOv2 替换（冻结，像素空间） | ~300 | FID < 1.3 |

**第三阶段（高成本，4–8 周）：**

| 实验 | 内容 | GPU 小时 | 成功标准 |
|------|------|---------|---------|
| P4 | 三编码器融合（MAE + ConvNeXt-V2 + DINOv2） | ~500 | FID < 1.0 |
| P5 | 多尺度 patch + 局部-全局注意力 | ~600 | FID < 0.9 |

### 6.2 Go/No-Go 决策标准

- P1 未达到 FID < 1.5：说明像素空间的瓶颈不在训练轮数，转向编码器改进
- P2 未达到 FID < 1.4：说明 ConvNeXt-V2 的增益在长训练下减弱，转向 DINOv2
- P3 未达到 FID < 1.3：说明冻结编码器已达上限，需要联合训练

### 6.3 与潜在空间的对比基准

所有像素空间实验的最终目标：**在相同训练轮数下，像素空间 FID ≤ 潜在空间 FID（1.54）**。

若 P4 达到 FID < 1.0，则像素空间生成将成为 Drifting Models 的新 SOTA，证明 VAE 不是必要的。

---

## 7. 理论分析：为什么像素空间可以超越潜在空间？

### 7.1 VAE 的信息损失

SD-VAE 的重建误差（LPIPS ~0.05）意味着潜在空间生成的图像在解码时会引入额外的模糊。像素空间生成直接优化像素级分布，没有这个误差源。

### 7.2 特征编码器的适配性

潜在空间的特征编码器是在 32×32×4 的潜在表示上预训练的，而像素空间的编码器可以直接使用在 ImageNet 上预训练的大规模模型（DINOv2、ConvNeXt-V2），这些模型的特征质量远高于从头训练的 latent-MAE。

### 7.3 多编码器融合的协同效应

论文已经证明 ResNet-MAE + ConvNeXt-V2 的组合在 100 轮时就达到 FID=3.70，而单独使用 MAE 需要 640 轮才能达到 FID=9.35。这说明不同架构的编码器提供了互补的特征，融合后的漂移场更准确。

---

## 参考文献

- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," 2023
- Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders," CVPR 2023
- Kirillov et al., "Segment Anything," ICCV 2023
- Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution," CVPR 2022
