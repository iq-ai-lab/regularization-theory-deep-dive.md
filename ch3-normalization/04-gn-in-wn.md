# 04. Group / Instance / Weight Normalization

## 🎯 핵심 질문

- **GroupNorm** (Wu & He 2018)은 BN의 small-batch 문제를 어떻게 해결하는가?
- **InstanceNorm** (Ulyanov 2016)이 style transfer에서 효과적인 이유는?
- **WeightNorm** (Salimans 2016)은 왜 **activation이 아닌 weight를 재매개변수화**하는가?
- 이 세 기법의 **정규화 축 분류**는?

---

## 🔍 왜 또 다른 normalization이 필요한가

BN/LN이 커버하지 못하는 영역들:

1. **Object detection, segmentation** — batch size 2~4 제한 (메모리). BN이 unstable, LN은 너무 공격적.
2. **Style transfer, GAN generator** — sample간 스타일 무관, sample별 통계로 정규화.
3. **Generative models** — batch 통계에 따라 output이 달라지면 문제.
4. **Activation 대신 weight을 건드리는 접근** — WeightNorm은 다른 philosophy.

이 세 기법은 각각 이 틈을 메운다.

---

## 📐 수학적 선행 조건

- Ch3-01, Ch3-02, Ch3-03
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): CNN의 channel/spatial 구조
- 기본 확률: 조건부 분포, 독립성

---

## 📖 직관적 이해

### 정규화 축 분류 (Wu & He 2018의 4축 그림)

Conv feature $x \in \mathbb{R}^{B \times C \times H \times W}$:

| 기법 | 정규화 범위 | 차원 |
|------|----------|------|
| **BatchNorm** | $B \times H \times W$ per channel | $(B, \cdot, H, W)$ |
| **LayerNorm** | $C \times H \times W$ per sample | $(\cdot, C, H, W)$ |
| **InstanceNorm** | $H \times W$ per (sample, channel) | $(\cdot, \cdot, H, W)$ |
| **GroupNorm** | $H \times W \times (C/G)$ per (sample, group) | $(\cdot, C/G, H, W)$ |

- **BN**: batch 축 포함 → batch 크기 의존.
- **LN**: 전체 channel 포함 → "사진 전체"가 통일 scale.
- **IN**: channel별 독립 → "각 채널이 독립 스타일".
- **GN**: BN과 LN의 중간 — $G$ 개의 채널 그룹.

### GroupNorm의 아이디어

CNN의 학습된 채널은 종종 **그룹으로** 작동 (e.g. 엣지 detector 그룹, texture 그룹). GN은 이 자연스러운 그룹화를 이용:

- $G = 1$: LayerNorm과 동치.
- $G = C$: InstanceNorm과 동치.
- $G = 32$ (표준): BN보다 안정, LN보다 표현력.

### InstanceNorm의 style transfer 적용

Image의 **content**는 spatial 패턴, **style**은 channel별 통계. IN은 각 (sample, channel) 독립 정규화 → **style을 제거**.

Style transfer의 **AdaIN** (Huang 2017): content의 IN으로 스타일 제거 → style image의 IN 통계 주입으로 스타일 교체.

### WeightNorm의 접근

Activation을 만지지 않고 **weight 자체를 reparameterize**:

$$w = g \cdot \frac{v}{\|v\|}$$

$g$ (scalar magnitude) + $v$ (direction vector). 두 분리 최적화:
- $g$: scalar → 간단한 scaling.
- $v$: unit vector 방향.

**효과**: gradient의 scale/direction이 분리 → condition number 개선, BN의 lightweight 대안.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — GroupNorm (Wu & He 2018)

Input $x \in \mathbb{R}^{B \times C \times H \times W}$. Channel을 $G$ 그룹 $(C = G \cdot C_g)$로 분할. 각 $(b, g)$ pair에서:

$$\mu_{b, g} = \frac{1}{C_g H W}\sum_{c \in g, h, w} x_{b, c, h, w}$$

마찬가지로 $\sigma^2_{b, g}$. 정규화 + affine:

$$\hat{x}_{b, c, h, w} = \frac{x_{b, c, h, w} - \mu_{b, g(c)}}{\sqrt{\sigma^2_{b, g(c)} + \epsilon}}, \quad y = \gamma_c \hat{x} + \beta_c$$

$\gamma, \beta \in \mathbb{R}^C$ channel별 affine.

### 정의 4.2 — InstanceNorm (Ulyanov 2016)

$$\mu_{b, c} = \frac{1}{HW}\sum_{h, w} x_{b, c, h, w}, \quad \sigma^2_{b, c} = \frac{1}{HW}\sum (x_{b, c, h, w} - \mu_{b, c})^2$$

각 (sample, channel)별로 spatial 통계.

### 정의 4.3 — WeightNorm (Salimans & Kingma 2016)

Linear layer $y = W x + b$에서 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Row별 reparameterization:

$$W_i = g_i \cdot \frac{V_i}{\|V_i\|}, \quad i = 1, \ldots, d_{\text{out}}$$

Learnable: $g \in \mathbb{R}^{d_{\text{out}}}$, $V \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Forward: $(V, g)$를 통해 $W$ 계산.

### 정리 4.4 — GroupNorm의 BN/LN/IN 한계 case

- $G = 1$: 모든 채널을 하나의 그룹 → LayerNorm.
- $G = C$: 각 채널 독립 → InstanceNorm.

표준 $G = 32$는 두 극단 사이의 **interpolation**.

### 정리 4.5 — WeightNorm의 Gradient Decomposition

$\nabla_{g_i} L = \frac{\nabla_{W_i} L \cdot V_i}{\|V_i\|}$  (magnitude direction)

$\nabla_{V_i} L = \frac{g_i}{\|V_i\|}\left(\nabla_{W_i} L - \frac{(\nabla_{W_i} L \cdot V_i) V_i}{\|V_i\|^2}\right)$  (projection onto $V_i^\perp$)

**$g, V$의 gradient가 orthogonal** — magnitude와 direction을 독립적으로 학습.

### 정리 4.6 — Small batch에서 GN vs BN 오차

Batch size $B$에서 통계 추정 오차:

$$\text{Var}(\hat{\mu}_{BN}) = \sigma^2 / B, \quad \text{Var}(\hat{\mu}_{GN}) \text{ 독립}$$

BN 오차가 $1/\sqrt{B}$로 증가 → $B = 2$에서 매우 noisy. GN은 **batch와 무관**.

---

## 🔬 수학적 유도

### GroupNorm이 LN, IN으로 환원되는 과정

$G = 1$: 모든 channel을 한 그룹 → $\mu_{b, 1} = \frac{1}{CHW}\sum x_{b, c, h, w}$ = LN의 정의 (채널 + 공간 전체).

$G = C$: 각 channel 독립 → $\mu_{b, c} = \frac{1}{HW}\sum_{h, w} x_{b, c, h, w}$ = IN.

$G = 32$ (C=128): 4 channels per group, 각 그룹별 정규화. 통계의 샘플 수 $4 \cdot H \cdot W$.

### WeightNorm Gradient 유도

$W_i = g_i V_i / \|V_i\|$, $L$에 대한 chain rule:

$\partial L/\partial g_i = (\partial L / \partial W_i) \cdot (V_i / \|V_i\|)$ (dot product, scalar 결과).

$\partial L/\partial V_i$: Quotient rule.

$W_i = g_i V_i \|V_i\|^{-1}$, $\partial W_i/\partial V_{ij} = g_i [\delta_{ij}/\|V\| - V_i V_j/\|V\|^3]$.

$(\partial L / \partial V_{ij}) = \sum_k (\partial L / \partial W_{ik})(\partial W_{ik}/\partial V_{ij})$

$= g_i [(\partial L/\partial W_{ij})/\|V\| - V_j (\partial L/\partial W_i \cdot V_i)/\|V\|^3]$

$= (g_i/\|V\|)[\partial L/\partial W_{ij} - V_j(\partial L/\partial W_i \cdot V_i)/\|V\|^2]$

즉 $V_i$의 gradient는 $\partial L/\partial W_i$에서 $V_i$ 방향 성분을 **빼고** 수직 성분만 남김. **Direction change**만 $V_i$ 업데이트에 영향.

---

## 💻 실험으로 효과 검증

### 실험 1 — Small batch에서 BN vs GN 비교

```python
import torch
import torch.nn as nn

class TestCNN(nn.Module):
    def __init__(self, norm='BN'):
        super().__init__()
        if norm == 'BN':
            nlayer = lambda c: nn.BatchNorm2d(c)
        elif norm == 'GN':
            nlayer = lambda c: nn.GroupNorm(8, c)
        elif norm == 'LN':
            nlayer = lambda c: nn.GroupNorm(1, c)  # LN as GN(G=1)
        elif norm == 'IN':
            nlayer = lambda c: nn.GroupNorm(c, c)  # IN as GN(G=C)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nlayer(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nlayer(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)

# Batch size 1, 2, 4, 8, 32로 CIFAR-10 훈련 후 test accuracy
# → BN: batch 1, 2에서 성능 급락
# → GN: 모든 batch size에서 일관된 성능
```

### 실험 2 — InstanceNorm for style transfer 직관

```python
# IN이 channel별 통계 제거 → "style normalization"
x = torch.randn(1, 3, 64, 64)     # 1 image, 3 channels, 64x64
# 각 채널이 다른 mean/std로 생성되었다고 가정
x[:, 0] = x[:, 0] * 2 + 5          # channel 0: mean=5, std=2
x[:, 1] = x[:, 1] * 0.5 - 1        # channel 1: mean=-1, std=0.5

in_layer = nn.InstanceNorm2d(3, affine=False)
y = in_layer(x)
print("After IN, each channel stats:")
for c in range(3):
    print(f"  channel {c}: mean={y[0, c].mean():.4f}, std={y[0, c].std():.4f}")
# → 모든 채널 (0, 1) stats — style 제거
```

### 실험 3 — WeightNorm 구현과 BN 비교

```python
class WeightNormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.V = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.g = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        V_norm = self.V.norm(dim=1, keepdim=True) + 1e-8
        W = self.g.unsqueeze(1) * self.V / V_norm
        return x @ W.T + self.bias

# PyTorch 내장: torch.nn.utils.weight_norm
linear = nn.Linear(10, 5)
wn_linear = torch.nn.utils.weight_norm(nn.Linear(10, 5))
print("wn_linear parameters:", [n for n, _ in wn_linear.named_parameters()])
# → weight_v, weight_g, bias
```

### 실험 4 — GN의 그룹 수 $G$ 영향

```python
import matplotlib.pyplot as plt

# Feature channels = 128, 다양한 G에서 성능 비교
G_list = [1, 2, 4, 8, 16, 32, 64, 128]   # G=1 (LN), G=128 (IN)
# CIFAR-10에서 각 $G$로 훈련, test error를 plot
# 논문 Fig 1: G=32가 sweet spot, 양 극단에서 성능 약간 저하
```

---

## 🔗 실전 활용

### 선택 가이드

| 상황 | 권장 |
|------|------|
| Large batch (>=32) CNN | BN |
| Small batch (detection, segmentation) | **GN** (G=32) |
| Style transfer, CycleGAN | **IN** |
| Transformer, RNN | LN |
| FC의 간단한 대안 | **WN** (BN의 $1/3$ cost) |
| Video model (spatiotemporal) | GN 또는 LN |

### Detection 예시

- **Mask R-CNN**: GroupNorm이 표준 (Wu & He 2018: GN-equipped Mask R-CNN → ResNet 성능 개선).
- **DETR**: LN + positional encoding.
- **YOLO v5-v8**: 부분적으로 GN과 BN 혼용.

### GAN에서의 IN

- **CycleGAN**: generator에 IN (style 제거), discriminator에 BN 또는 LN.
- **StyleGAN**: AdaIN (IN + learned $\gamma, \beta$) — style 주입 메커니즘.

### WeightNorm의 장단점

- **장점**: BN의 ~$1/3$ 계산 비용, running stats 불필요, small batch에서도 작동.
- **단점**: BN만큼 landscape smoothing 효과 없음, 깊은 네트워크에서 한계.
- **실전**: Generative 모델, RNN에서 가끔 (e.g. parallel wavenet).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| GN의 group 수가 임의 | $G = 32$ 기본값이 모든 task에 최적인 것은 아님 |
| IN의 channel 독립성 | Discriminative task에서 channel 정보 소실 가능 |
| WN의 weight-space 접근 | BN의 activation smoothing 효과 누락 |
| Affine $(\gamma, \beta)$ 추가 | Normalization 후 표현력 복원 필수 |

---

## 📌 핵심 정리

$$\boxed{\text{BN (batch)} \quad \text{LN (all)} \quad \text{IN (per-chan)} \quad \text{GN (groups)} \quad \text{WN (weight)}}$$

| 기법 | Axis | 대표 사용처 |
|------|------|-----------|
| **BN** | $(B, H, W)$ | Large-batch CNN |
| **LN** | $(C, H, W)$ | Transformer, RNN |
| **IN** | $(H, W)$ | Style transfer |
| **GN** | $(C/G, H, W)$ | Small-batch CNN |
| **WN** | weight-space | FC layer lightweight alternative |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $C = 64$, $G = 16$일 때 각 그룹의 channel 수는? $G = 1$과 $G = 64$는 각각 어떤 normalization과 동치?

<details>
<summary>힌트 및 해설</summary>

- $C/G = 64/16 = 4$ channels per group.
- $G = 1$: 전체 64 channel 한 그룹 → **LayerNorm**.
- $G = 64$: 각 channel 독립 → **InstanceNorm**.

GN은 이 두 극단 사이의 연속적 spectrum을 제공. ResNet-50 실험(Wu & He)에서 $G = 32$이 ImageNet top-1 accuracy 최고.

</details>

**문제 2** (심화): WeightNorm의 gradient decomposition (정리 4.5)에서 "$V_i$ gradient는 $V_i^\perp$ 성분만 있다"의 의미를 기하적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

$V_i$는 **unit-direction**의 scaled 버전 ($\hat{V}_i = V_i / \|V_i\|$). $V_i$의 길이($\|V_i\|$)는 의미가 없고 **direction** $\hat{V}_i$만 중요 — $g_i$와 함께 $W_i = g_i \hat{V}_i$로 결정.

정리 4.5의 gradient는 $V_i$의 **radial 성분**(length 변화)을 **제거**, **tangential 성분**(direction 변화)만 남긴다. 즉 $V_i$ 업데이트가 direction만 돌리고 length는 유지.

이것이 **reparameterization invariance** — $(V_i, g_i) = (2V_i, g_i)$ 같은 equivalent pair가 gradient 관점에서 unique하게 처리되지 않음. Optimizer가 길이 자유도를 낭비하지 않고 오직 direction만 optimize.

이 decomposition이 "optimizer scale invariance"의 이론적 근거. Natural gradient와 연결된다.

</details>

**문제 3** (이론-실전): GroupNorm은 BN보다 detection task에서 더 나은 이유는 batch size뿐만 아니라 **transferability** 관점도 있다. 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Detection task 특성**:
1. **Small batch** (2~4, COCO): BN noise 큼.
2. **Diverse image sizes**: BN은 spatial-axis까지 통계, image size 변화에 민감.
3. **Fine-tuning from ImageNet**: BN의 running stats가 ImageNet 분포에 fitted — COCO 이미지는 다름.

**GN의 장점**:
- Running stats 없음 → **train/eval 일관성**.
- Batch 통계 안 씀 → **ImageNet → COCO transfer 시 stats mismatch 없음**.
- Channel 그룹 구조 → semantic feature grouping 유지 (localization에 유리).

Wu & He 2018 Table 3: ImageNet pre-trained BN vs GN on Mask R-CNN:
- BN: 36.5 AP (batch=2로 훈련), BN은 동결해야 함.
- GN: 40.3 AP, end-to-end 훈련 가능.

이는 단순히 batch size 효과가 아니라 **normalization 자체의 transfer 속성** 차이. 현대 detection pipeline에서 GN이 거의 표준.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Layer Normalization](./03-layer-norm.md) | [📚 README로 돌아가기](../README.md) | [05. Fixup · SkipInit ▶](./05-fixup-skipinit.md) |

</div>
