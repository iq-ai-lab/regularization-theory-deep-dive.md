# 04. CutMix · CutOut · RandAugment

## 🎯 핵심 질문

- **CutMix** (Yun 2019)는 Mixup과 어떻게 다른가? 왜 object detection에서 더 자연스러운가?
- **CutOut** (DeVries 2017)의 random erasing이 정보 이론적으로 주는 효과는?
- **AutoAugment** (Cubuk 2018)의 RL-based policy search가 왜 2 billion GPU-hour를 쓰는가?
- **RandAugment** (Cubuk 2020)가 AutoAugment를 어떻게 2-parameter simplification으로 대체했는가?

---

## 🔍 왜 이 변종들이 필요한가

Mixup(Ch4-03)의 두 약점:

1. **이미지 품질**: 두 이미지의 alpha-blend는 **unnatural** — 자연 데이터 분포에서 멀다.
2. **Localization 손실**: detection, segmentation 같은 spatial task에서 object의 정확한 위치가 섞여 사라진다.

**CutMix의 해결**: patch 교환. 한 이미지의 일부 영역을 다른 이미지로 교체, 라벨은 **영역 비율로 mix**. 각 region이 **원본 이미지 pixel**을 유지 → natural.

**CutOut의 해결**: random erasing (그냥 일부 영역 검은색으로). 단순하지만 robustness 개선.

**AutoAugment / RandAugment**: augmentation **policy** (어떤 augmentation을 어느 강도로, 어떤 확률로) 자동화. AutoAugment는 RL로 expensive, RandAugment는 **2 parameter (num_ops, magnitude)**만.

이 네 기법은 현대 CNN 훈련 recipe의 핵심.

---

## 📐 수학적 선행 조건

- Ch4-01, Ch4-03: VRM, Mixup
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): uniform 분포 over rectangles
- RL 기초 (AutoAugment의 policy search 이해)

---

## 📖 직관적 이해

### CutMix

**정의**: 이미지 $x_i$의 랜덤 직사각형 영역을 $x_j$의 해당 영역으로 교체. 라벨은 **영역 비율**로 보간:

$$\tilde{x} = M \odot x_i + (1 - M) \odot x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

$M$은 binary mask, $\lambda$ = ratio of ones in $M$.

### Mixup vs CutMix 비교

| | Mixup | CutMix |
|---|-------|--------|
| 이미지 조합 | Alpha blending | Patch replacement |
| Natural-ness | Low (blurry) | High (자연 이미지 patch) |
| Object detection | Bad (object 흐려짐) | Good (object 일부 가려짐) |
| Classification | 둘 다 OK | CutMix가 약간 우수 (보통) |

### CutOut

**정의**: 이미지 $x_i$에 랜덤 직사각형 영역을 **0**으로 채우기 (또는 dataset mean):

$$\tilde{x} = M \odot x_i$$

라벨은 변하지 않음. 극단적으로 간단하지만 효과 있음.

**직관**: Object의 일부가 가려진 상황에서도 분류해야 — real world의 occlusion에 robustness.

### AutoAugment / RandAugment

**Augmentation policy**: "rotation 10°, AutoContrast, sharpness 0.3" 같은 여러 augmentation의 순서·강도·확률.

- **AutoAugment** (Cubuk 2018): RL controller가 policy space 탐색, validation accuracy를 reward. **Expensive** (5000 GPU-hours).
- **RandAugment** (Cubuk 2020): 복잡한 search 포기, 2 parameter만:
  - $N$: 적용할 augmentation 수 (e.g. 2).
  - $M$: magnitude (0~30).
  - 각 단계에서 15개 augmentation 중 $N$개 uniform random 선택.

RandAugment는 AutoAugment와 거의 같은 성능을 **10배 빠른 search**로 달성.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — CutMix (Yun et al. 2019)

Images $x_i, x_j \in \mathbb{R}^{C \times H \times W}$. Rectangle $R = [r_x, r_x + r_w] \times [r_y, r_y + r_h]$ (uniform random), mask $M = \mathbb{1}_R$ (1 inside, 0 outside):

$$\tilde{x} = (1 - M) \odot x_i + M \odot x_j$$

Label ratio:

$$\lambda = 1 - \frac{r_w r_h}{HW}, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

$r_w = W\sqrt{1 - \lambda}, r_h = H\sqrt{1 - \lambda}$로 **random patch 크기를 $\lambda$와 일관**되게 선택.

### 정의 4.2 — CutOut (DeVries & Taylor 2017)

Random rectangle $R$ 크기는 fixed (e.g. $16 \times 16$ for CIFAR). Mask $M = \mathbb{1}_R$:

$$\tilde{x} = (1 - M) \odot x_i + M \odot c_{\text{fill}}$$

$c_{\text{fill}}$은 보통 dataset mean (0 after normalization). Label 변화 없음.

### 정의 4.3 — AutoAugment Policy (Cubuk et al. 2018)

**Sub-policy**: 2 ops, 각 op은 (augmentation type, probability $p$, magnitude $m$). 전체 policy는 25 sub-policies, 각 이미지에 random 하나 적용.

**Search**: RNN controller (1 billion+ parameter search over $\sim 10^{32}$ possible policies). Validation acc → reward for REINFORCE.

### 정의 4.4 — RandAugment (Cubuk et al. 2020)

두 hyperparameter:
- $N$: number of augmentations to apply sequentially (e.g. 1, 2, 3).
- $M$: magnitude for all augmentations (0~30 scale).

Augmentation pool (14 transformations):
```
AutoContrast, Equalize, Invert, Rotate, Posterize, Solarize,
Color, Contrast, Brightness, Sharpness, ShearX, ShearY, TranslateX, TranslateY
```

각 이미지에 $N$개 uniform random sample, 각각 magnitude $M$으로 적용.

### 정리 4.5 — CutMix의 Mixup 비교 우위 (Yun 2019)

CutMix가 Mixup보다 accuracy 개선 (표준 ResNet-50 ImageNet):
- ImageNet top-1: ERM 76.3% → Mixup 77.4% → CutMix **78.6%**.
- CIFAR-100: 78.2% → 80.5% → **81.7%**.

특히 **localization task** (weakly-supervised localization) 에서 CutMix >> Mixup >> ERM.

### 정리 4.6 — RandAugment의 Efficiency

AutoAugment의 policy search cost: $\sim 5000$ GPU-hours. RandAugment: **$\sim 10$ GPU-hours** (validation grid $N, M$).

동등 성능: ImageNet EfficientNet-B7에서 AutoAugment 84.0% vs RandAugment 83.7% (difference $< 0.5\%$).

---

## 🔬 수학적 유도

### CutMix의 VRM Formulation

Vicinity:

$$\mathcal{D}_{x_i, y_i}^{\text{CutMix}} = \mathbb{E}_j \mathbb{E}_R [\delta_{(x_{i,j,R}, \lambda_R y_i + (1 - \lambda_R) y_j)}]$$

$x_{i,j,R}$은 $x_i$에 $x_j$의 $R$-patch 삽입. Mixup과 달리 "**spatial locality 유지**" — 각 pixel은 $x_i$ 또는 $x_j$의 **정확한 pixel**.

### CutOut의 Dropout과의 관계

CutOut은 **input-space dropout** 중 spatial 구조화:
- Element-wise input dropout: random individual pixels.
- Spatial dropout on input: random spatial regions.
- CutOut: rectangular erasing.

모두 VRM 관점에서 "특정 input patch를 잃은 augmentation" vicinity.

### RandAugment의 정당화

AutoAugment가 찾은 policy의 **다양성**이 효과적이라는 관찰 → "policy 자체를 정확히 학습할 필요 없다". **Uniform sampling over augmentation pool**이 sufficient diversity 제공.

이는 "curse of dimensionality"를 역으로 이용 — augmentation space의 dimension이 크면 **random이 좋은 coverage** 제공.

---

## 💻 실험으로 효과 검증

### 실험 1 — CutMix 구현

```python
import numpy as np
import torch

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))

    B, C, H, W = x.shape
    r_w = int(W * np.sqrt(1 - lam))
    r_h = int(H * np.sqrt(1 - lam))
    r_x = np.random.randint(W)
    r_y = np.random.randint(H)

    # bounding box clipped to image
    x1 = max(0, r_x - r_w // 2)
    x2 = min(W, r_x + r_w // 2)
    y1 = max(0, r_y - r_h // 2)
    y2 = min(H, r_y + r_h // 2)

    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    # adjust lambda (actual ratio of remaining original pixels)
    lam_adj = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    return x_mix, y, y[idx], lam_adj

# Visualization
import matplotlib.pyplot as plt
x = torch.randn(2, 3, 224, 224).clamp(-1, 1)
x_mix, y_a, y_b, lam = cutmix(x, torch.tensor([0, 1]), alpha=1.0)
plt.imshow(x_mix[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)
plt.title(f'CutMix: 이미지 0 base + 이미지 1 patch (λ={lam:.2f})')
plt.show()
```

### 실험 2 — CutOut 구현

```python
def cutout(x, size=16, fill=0):
    B, C, H, W = x.shape
    x_cut = x.clone()
    for b in range(B):
        y_c, x_c = np.random.randint(H), np.random.randint(W)
        y1 = max(0, y_c - size // 2)
        y2 = min(H, y_c + size // 2)
        x1 = max(0, x_c - size // 2)
        x2 = min(W, x_c + size // 2)
        x_cut[b, :, y1:y2, x1:x2] = fill
    return x_cut
```

### 실험 3 — RandAugment 구현 (간략)

```python
import torchvision.transforms.functional as TF
from PIL import Image

class RandAugment:
    def __init__(self, N=2, M=9):
        self.N = N
        self.M = M   # magnitude 0~30

    def __call__(self, img):
        ops_pool = [
            ('AutoContrast', self._auto_contrast),
            ('Rotate', self._rotate),
            ('Sharpness', self._sharpness),
            ('ShearX', self._shear_x),
            ('TranslateY', self._translate_y),
            # ... 15개 total
        ]
        ops = np.random.choice(len(ops_pool), self.N, replace=True)
        for i in ops:
            name, fn = ops_pool[i]
            img = fn(img, self.M)
        return img

    def _rotate(self, img, m):
        angle = m * 30 / 30  # scale M to angle
        return TF.rotate(img, angle)
    def _sharpness(self, img, m):
        factor = 0.1 + m / 30 * 1.8
        return TF.adjust_sharpness(img, factor)
    # ... (other implementations)

# PyTorch 내장: torchvision.transforms.RandAugment
from torchvision.transforms import RandAugment as RA
ra = RA(num_ops=2, magnitude=9)  # PyTorch standard
```

### 실험 4 — CIFAR-10에서 기법 비교

```python
# 동일 ResNet-18, 동일 lr schedule, 다음 augmentation 조합으로 훈련:
configs = [
    'No augment',                    # baseline
    'Random crop + Flip',             # standard
    'Standard + CutOut (16)',         # +cutout
    'Standard + Mixup (α=1)',         # +mixup
    'Standard + CutMix (α=1)',        # +cutmix
    'Standard + RandAugment (N=2, M=9)', # +randaug
    'All combined',                   # 모든 것
]

# 전형적 결과 (CIFAR-10 ResNet-18, 200 epochs):
#   baseline: 88.5%
#   +crop+flip: 93.0%
#   +CutOut: 94.0%
#   +Mixup: 95.0%
#   +CutMix: 95.5%
#   +RandAug: 96.0%
#   All: 96.5%
```

### 실험 5 — AutoAugment vs RandAugment 결과 (ImageNet)

```python
# EfficientNet-B7 ImageNet top-1:
#   ERM + standard:    83.0%
#   + AutoAugment:     84.0%  (5000 GPU-hours search)
#   + RandAugment:     83.7%  (10 GPU-hours validation)

# RandAugment가 AutoAugment와 거의 같은 성능을 훨씬 적은 cost로
```

---

## 🔗 실전 활용

### 현대 CNN Recipe (2024 기준)

**CIFAR-10**:
```python
transforms = [
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    CutOut(16),
    ToTensor(), Normalize(),
    # 선택: Mixup or CutMix (in training loop)
]
```

**ImageNet**:
```python
transforms = [
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    ToTensor(), Normalize(),
    # + Mixup(α=0.2) or CutMix(α=1.0) in training
]
```

### 조합 효과

- **CutMix + Mixup alternating**: 매 epoch 랜덤 선택, 다양성 ↑.
- **CutMix + RandAugment**: 공간 mix + 색상/기하 변환 조합.
- **CutOut + Mixup**: 작은 모델에서 조심 — 너무 많은 augmentation은 underfit.

### Task별 권장

| Task | 권장 |
|------|------|
| ImageNet classification | RandAugment + CutMix |
| CIFAR-10/100 | CutOut + Mixup 또는 CutMix |
| Object detection | Copy-paste augmentation (CutMix extension) |
| Segmentation | CutMix (mask도 교환), CutOut 주의 |
| Medical imaging (small data) | 보수적 — flip, small rotation, CutOut만 |

### Implementation Tips

- **CutMix label scale**: 실제 region size로 $\lambda$ 계산 (bounding box clipping 후).
- **RandAugment M**: 큰 모델 (ResNet-200+)에서는 M=10~15, 작은 모델은 M=5~9.
- **Magnitude scheduling**: Epoch마다 M을 서서히 증가 (curriculum) — 가끔 효과.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| CutMix의 patch가 label-informative | 배경 patch로 바뀌면 label mismatch |
| CutOut size가 적절 | 너무 크면 object 완전 제거, 너무 작으면 noise 수준 |
| RandAugment pool이 충분 | 특수 domain (medical) 에서는 다른 pool 필요 |
| Augmentation이 independent | Sequential effect 고려 안 됨 |
| Fixed $M$ throughout training | Adaptive $M$이 더 좋을 수도 |

---

## 📌 핵심 정리

$$\boxed{\text{CutMix: patch swap} \quad \text{CutOut: erase} \quad \text{RandAugment: 2-param policy}}$$

| 기법 | 메커니즘 | 특징 |
|------|---------|------|
| **CutMix** | Patch 교환 + label ratio | Spatial information 보존 |
| **CutOut** | Random erasing | 단순, Occlusion robustness |
| **AutoAugment** | RL policy search | Expensive, optimal policy |
| **RandAugment** | Random (N, M) | Cost-effective, AutoAug와 거의 동등 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 224×224 이미지에서 CutMix $\lambda = 0.6$일 때 patch 크기는?

<details>
<summary>힌트 및 해설</summary>

$r_w = W\sqrt{1 - \lambda} = 224 \sqrt{0.4} \approx 224 \times 0.632 \approx 141.7 \approx 142$.

$r_h = H\sqrt{1 - \lambda} \approx 142$ (같은 비율).

Patch area $= 142 \times 142 \approx 20,164$ pixels.  
Total area $= 224 \times 224 = 50,176$.  
Area ratio $= 20,164 / 50,176 \approx 0.402$ ≈ $1 - \lambda = 0.4$ ✓.

</details>

**문제 2** (심화): CutMix의 $\lambda$는 이미지 patch의 **area ratio**, Mixup의 $\lambda$는 linear mix coefficient. 두 $\lambda$가 같은 의미를 갖는가?

<details>
<summary>힌트 및 해설</summary>

**엄밀히는 다르다**.

- **Mixup**: $\tilde x$의 각 pixel이 **$\lambda x_i + (1-\lambda) x_j$**. 모든 pixel이 양쪽 정보 섞여 있음.
- **CutMix**: $\tilde x$의 각 pixel은 **정확히 $x_i$ 또는 $x_j$ 중 하나**. $\lambda$는 $x_i$의 pixel 비율.

그러나 **기댓값 관점**에서 동일:
- Mixup: $\mathbb{E}[\tilde x \text{의 한 pixel}] = \lambda x_i + (1-\lambda) x_j$.
- CutMix: $P(\text{한 pixel이 } x_i\text{에서 옴}) = \lambda$ → $\mathbb{E}[\tilde x \text{의 한 pixel}] = \lambda x_i + (1-\lambda) x_j$.

즉 **평균적 information content**는 같지만 **공간적 coherence**가 다름:
- Mixup: "모든 곳이 50/50"이라 어떤 object도 **흐림**.
- CutMix: "왼쪽은 image A, 오른쪽은 image B"라 각 object가 **선명**.

이 차이가 localization, detection 같은 spatial task에서 CutMix 우위로 나타남.

</details>

**문제 3** (이론-실전): RandAugment의 $(N, M) = (2, 9)$이 ImageNet의 EfficientNet에 표준이다. 더 큰 모델에서는 더 큰 $M$을 쓰는 이유를 **overfitting/underfitting** 관점으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**큰 모델 = capacity ↑ = overfit 경향**. 강한 augmentation이 필요.

**작은 모델**: capacity 제한 → underfit 위험. 강한 augmentation은 **learn되어야 할 signal을 noise로 변형** → 더 떨어뜨림.

**RandAugment paper Table 2**:
- ResNet-50: $M = 9$ 최적.
- ResNet-200: $M = 13$ 최적.
- EfficientNet-B7: $M = 15$ 최적 (더 큰 모델일수록 큰 $M$).

일반 경험: **Training accuracy - Validation accuracy gap**이 크면 $M$ 증가, gap 작거나 training acc 낮으면 $M$ 감소.

**$N$ (number of ops)**: $N = 2$가 거의 모든 경우 최적. $N = 3, 4$는 diminishing returns, $N = 1$은 약함.

**실전 실용적 조언**: 새 task에서는 (N=2, M=9)로 시작, generalization gap 관찰 후 조정. EfficientNet / RegNet 레시피 참고.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Mixup](./03-mixup.md) | [📚 README로 돌아가기](../README.md) | [05. Contrastive Learning ▶](./05-contrastive.md) |

</div>
