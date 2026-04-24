# 05. Contrastive Learning과 Augmentation

## 🎯 핵심 질문

- **SimCLR** (Chen 2020)의 "두 augmented view" 프레임워크는 어떻게 작동하는가?
- **InfoNCE loss**는 왜 augmentation을 **semantic invariance**로 해석하게 하는가?
- Augmentation 선택이 representation quality에 **얼마나 critical**한가?
- **MoCo, BYOL, DINO**는 SimCLR와 어떻게 다른가?

---

## 🔍 왜 contrastive learning이 augmentation의 한계 주제인가

Ch4-01~04의 augmentation은 **supervised** context에서의 regularization:
- Label이 있고, augmentation이 "invariance를 주입".

**Contrastive learning**은 **self-supervised**:
- **Label 없음**. Augmentation이 **학습 신호 자체**를 만든다.
- "같은 이미지의 두 augmented view를 가깝게, 다른 이미지의 view를 멀게".

이 설정에서 augmentation은 단순한 regularizer가 아닌 **semantic definition의 결정 요소**:
- Color jitter 포함 → 색상 무관 feature 학습.
- Heavy crop → global vs local structure trade-off.
- Rotation 제외 → 방향 정보 보존.

SimCLR (Chen 2020)가 이 아이디어를 규모 있게 검증하면서 self-supervised learning 혁명을 시작. 현재 (2024) vision 분야에서 CLIP, DINO, MAE 등 self-supervised 모델이 supervised와 경쟁.

이 문서는 contrastive augmentation의 **loss, theory, 실전 선택**을 정리.

---

## 📐 수학적 선행 조건

- Ch4-01, Ch4-02: VRM, invariance
- 정보이론: entropy, mutual information $I(X; Y)$, InfoNCE와 MI의 관계
- Softmax / cross-entropy
- Cosine similarity

---

## 📖 직관적 이해

### SimCLR Framework

1. Image $x$에서 **두 random augmentation** $t_1, t_2 \in T$ 샘플 → $x^{(1)} = t_1(x), x^{(2)} = t_2(x)$.
2. Encoder $f$로 representation $h^{(k)} = f(x^{(k)})$.
3. Projection head $g$ (2-layer MLP)로 $z^{(k)} = g(h^{(k)})$.
4. **Contrastive loss**: $(z^{(1)}, z^{(2)})$가 같은 이미지에서 왔음을 맞추도록 학습.

**"Same image" pair = positive, different images = negative**.

### InfoNCE (van den Oord 2018)

Batch of $N$ images, $2N$ augmented views. Anchor $z_a$에 대해:

$$\mathcal{L}_{\text{InfoNCE}}(z_a) = -\log \frac{\exp(\text{sim}(z_a, z_p)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}[k \neq a] \exp(\text{sim}(z_a, z_k)/\tau)}$$

- $z_p$: anchor의 positive pair (같은 original image에서 온 다른 view).
- $z_k$ (otherwise): negative pair.
- $\text{sim}$: cosine similarity.
- $\tau$: temperature (e.g. 0.1).

**직관**: positive similarity를 상대적으로 크게. "$z_a$가 $z_p$와 같은 image에서 왔다는 걸 맞추기" — classification task처럼 해석 가능.

### Augmentation Choice의 Critical성

Chen 2020 Figure 5: 각 augmentation 조합별 ImageNet linear eval accuracy.

- Random crop만: 낮음.
- Crop + color jitter: 크게 개선.
- Crop + color + Gaussian blur: 최고.

**결론**: 여러 augmentation의 **조합**이 효과적. 단일 augmentation은 약함.

### MoCo, BYOL, DINO

**MoCo** (He 2020): Momentum encoder로 **negative queue** 유지. 큰 batch 없이도 많은 negative 사용.

**BYOL** (Grill 2020): **Negative 없음**! Target encoder의 EMA로 collapse 방지.

**DINO** (Caron 2021): Self-distillation + centering. Teacher/student arch, no explicit negatives.

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Augmentation Distribution $\mathcal{T}$

Image transformation 집합 $\mathcal{T}$ (e.g. {crop, flip, color jitter, blur, ...}). 각 $t \in \mathcal{T}$가 특정 확률 분포로 샘플.

### 정의 5.2 — SimCLR Augmentation Pipeline (Chen 2020)

```
1. RandomResizedCrop(224, scale=(0.08, 1.0))
2. RandomHorizontalFlip(p=0.5)
3. ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
4. RandomGrayscale(p=0.2)
5. GaussianBlur(kernel=23, sigma=(0.1, 2.0), p=0.5)
```

### 정의 5.3 — Contrastive Loss (NT-Xent, Chen 2020)

Mini-batch of $N$ images. 각 image에 two random augmentations → $2N$ representations. Positive pair: $(z_{2i-1}, z_{2i})$ ($i$-th image의 두 view).

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}[k \neq i] \exp(\text{sim}(z_i, z_k)/\tau)}$$

$$\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N}\sum_{i=1}^N [\ell_{2i-1, 2i} + \ell_{2i, 2i-1}]$$

**Normalized Temperature-scaled Cross Entropy**.

### 정리 5.4 — InfoNCE와 Mutual Information Lower Bound (Poole 2019)

$$I(X; Y) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$$

즉 InfoNCE 최소화 = MI 하한 최대화. **Augmented views 간 MI 최대화**.

### 정리 5.5 — Augmentation Invariance Learning (HaoChen 2021)

Feature $f$가 InfoNCE로 훈련되면:

$$f^* \to \text{Aug-invariant: } f(t(x)) = f(x) \ \forall t \sim \mathcal{T}$$

수학적으로: $\mathcal{T}$ 하 orbit의 projection $\pi_{\mathcal{T}}$가 $f^*$의 속성.

### 정의 5.6 — BYOL (Bootstrap Your Own Latent, Grill et al. 2020)

**Online** encoder $f_\theta$ + **Target** encoder $f_\xi$ (EMA of online: $\xi \leftarrow \tau \xi + (1-\tau) \theta$).

Loss:

$$\mathcal{L}_{\text{BYOL}} = \|p_\theta(f_\theta(x^{(1)})) - \text{stopgrad}(f_\xi(x^{(2)}))\|^2$$

$p_\theta$는 predictor MLP. **Negative pair 불필요** — target encoder의 느린 변화가 collapse 방지.

---

## 🔬 수학적 유도

### InfoNCE → MI Lower Bound 유도

Poole et al. 2019: 각 positive에 대해 negative $N-1$개 있으면:

$\mathcal{L}_{\text{NCE}} = -\mathbb{E}\log \frac{p(y | x)}{p(y | x) + (N-1) \bar{p}(y)}$

$\bar{p}(y) = \mathbb{E}_{x'}[p(y | x')]$. 이 loss가 $\log N - I(X; Y)$의 추정자.

**함의**:
- $N$ 커짐 (많은 negative) → bound tighter.
- $\mathcal{L}_{\text{NCE}}$ 최소화 → $I(X; Y)$ 최대화.
- Positive pair = 같은 image의 두 view → augmentation이 "**같은 latent semantic에서 온 view**"라는 가정.

### Augmentation이 invariance 결정

**Key insight** (Chen 2020): $f^*$의 invariance는 **$\mathcal{T}$에 의해 결정**.

- Color jitter 포함 → color-invariant feature.
- Crop 포함 → scale-invariant, spatially-local feature.
- Blur 포함 → detail-invariant, semantic-level feature.

이는 **semantic을 dataset이 아닌 augmentation으로 정의**. Supervised와 결정적 차이 (supervised는 label이 semantic 정의).

### BYOL의 Collapse-Avoidance 미스터리

BYOL이 negative 없이 왜 작동하는가? Grill의 초기 설명: target encoder의 EMA가 다양성 보장. 후속 분석:

- **Richemond 2020**: Predictor $p_\theta$와 EMA의 조합이 암묵적 contrastive signal.
- **Tian 2021**: BatchNorm의 이미지 간 implicit contrastive 역할.
- **Chen 2021 (SimSiam)**: EMA도 필요 없음, stop-gradient만으로 가능.

**아직 완전히 이해되지 않은 현상** — 열린 연구.

---

## 💻 실험으로 효과 검증

### 실험 1 — SimCLR 훈련 (간략)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

simclr_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
    transforms.ToTensor(),
])

class SimCLRModel(nn.Module):
    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.backbone = backbone  # e.g. ResNet-50 without final fc
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, proj_dim))
    def forward(self, x):
        h = self.backbone(x)       # representation
        z = self.projector(h)       # projection
        return F.normalize(z, dim=-1)

def nt_xent_loss(z1, z2, tau=0.1):
    """z1, z2: (B, proj_dim), L2-normalized."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)   # (2N, dim)
    sim = torch.mm(z, z.T) / tau     # (2N, 2N)
    
    # Mask out self-similarity
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -float('inf'))
    
    # Positive pairs: i ↔ i+N
    pos_idx = torch.arange(2*N, device=z.device)
    pos_idx[:N] += N; pos_idx[N:] -= N
    
    logits = sim
    labels = pos_idx
    return F.cross_entropy(logits, labels)

# 훈련 루프
for x, _ in loader:   # label 무시
    x1 = torch.stack([simclr_transforms(im) for im in x])
    x2 = torch.stack([simclr_transforms(im) for im in x])
    z1, z2 = model(x1), model(x2)
    loss = nt_xent_loss(z1, z2, tau=0.1)
    loss.backward(); opt.step()
```

### 실험 2 — Augmentation ablation (Chen 2020 Figure 5 재현)

```python
# 각 augmentation 조합별 linear eval accuracy
# ImageNet subset에서 SimCLR 훈련 후 linear classifier로 평가

augmentation_configs = [
    ['Crop'],
    ['Crop', 'Color jitter'],
    ['Crop', 'Color jitter', 'Blur'],
    ['Crop', 'Color jitter', 'Blur', 'Grayscale'],
]

# 전형적 결과 (linear eval on ImageNet):
# Crop alone:                     25%
# Crop + Color jitter:            55%
# Crop + Color jitter + Blur:     65%
# 전체 SimCLR aug:                 69%

# 단일 augmentation과 조합 effect 차이가 매우 크다 (non-additive)
```

### 실험 3 — MoCo 구현 스케치

```python
class MoCoModel(nn.Module):
    def __init__(self, backbone, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.f_q = backbone          # online encoder
        self.f_k = copy.deepcopy(backbone)  # momentum encoder
        self.m = m                   # momentum coefficient
        self.T = T                   # temperature
        # Queue of negatives
        self.register_buffer('queue', F.normalize(torch.randn(128, K), dim=0))
        self.K = K
        
        for p in self.f_k.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1 - self.m)
    
    def forward(self, x1, x2):
        q = F.normalize(self.f_q(x1), dim=-1)  # query
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.f_k(x2), dim=-1)  # key (positive)
        
        # Positive logits: (B, 1)
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
        # Negative logits: (B, K)
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone())
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self.queue = torch.cat([k.T, self.queue[:, :-k.size(0)]], dim=1)
        return loss
```

### 실험 4 — Augmentation strength의 contrastive 효과

```python
# Augmentation 강도 (e.g. ColorJitter strength 0.0 ~ 1.0)에 대한 downstream accuracy
# → 중간 강도 (0.4~0.6)에서 최적
# → Too weak: views가 너무 유사 (trivial 해)
# → Too strong: views가 semantically 다름 (noise)
```

---

## 🔗 실전 활용

### Self-Supervised vs Supervised 선택

| Situation | 권장 |
|-----------|------|
| Labeled data 많음 (ImageNet) | Supervised + augmentation |
| Large unlabeled + small labeled | SimCLR/MoCo pretrain → fine-tune |
| Domain transfer | Contrastive pretrain이 특히 효과적 |
| Very large pretraining (billion+ images) | CLIP-style text-image contrastive |

### Contrastive 기법 선택

- **SimCLR**: 단순, large batch 필요 (4096+).
- **MoCo v2**: Large batch 불필요 (queue로 대체), 더 실용적.
- **BYOL**: Negative 없어도 작동, SimCLR 대체.
- **DINO**: Vision Transformer에 특히 효과적.
- **SimSiam**: Simplest (no momentum, no negative, no projection MLP 최소).

### Augmentation Design Principles

1. **Crop이 항상 critical**: spatial-local feature 학습.
2. **Color augmentation**: color-invariant semantic 학습.
3. **Blur**: detail 무시, semantic-level feature.
4. **Multi-scale crops** (DINO): local + global views의 조합.

### Foundation Models

CLIP, ALIGN 같은 vision-language model: 이미지와 **텍스트**를 contrastive pair로. Augmentation은 덜 critical, 대신 **large dataset** (400M+ pairs)가 핵심.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Augmentation이 semantic-preserving | Hard augmentation이 label을 바꿀 수 있음 |
| Negative pair는 "정말 다른" | Same image의 다른 crop을 negative로 쓰면 noise |
| Large batch (SimCLR) | Small batch에서 SimCLR 성능 저하 — MoCo 사용 |
| Vision에 특화 | NLP에서는 text augmentation 전략 다름 |
| Downstream task agnostic | Task-specific augmentation이 더 유용할 수 있음 |

**주의**: Contrastive learning의 성공은 **augmentation + architecture + batch size + training time**의 복합 결과. 단순 replication이 어려움.

---

## 📌 핵심 정리

$$\boxed{\text{Contrastive Aug: } \text{sim}(z^{(1)}, z^{(2)}) \uparrow \text{ for positive, } \downarrow \text{ for negative}}$$

| 개념 | 의미 |
|------|------|
| **Two augmented views** | Positive pair의 정의 |
| **InfoNCE** | MI lower bound 최대화 → augmentation-invariance |
| **Augmentation 선택** | Representation의 **semantic**을 결정 |
| **SimCLR / MoCo / BYOL / DINO** | 각각 다른 접근, 공통: aug-based invariance |
| **Ch4 마무리** | VRM → invariance → structured mix → contrastive |

---

## 🤔 생각해볼 문제

**문제 1** (기초): SimCLR에서 batch size 128, 각 이미지에 2 augmentation이면 한 anchor의 negative 수는?

<details>
<summary>힌트 및 해설</summary>

총 view: $2 \times 128 = 256$.  
한 anchor의 positive: 1 (같은 이미지의 다른 view).  
Anchor 자체 제외: 1.  
**Negative: $256 - 1 - 1 = 254$**.

SimCLR는 batch size가 중요한 이유: $N$이 클수록 InfoNCE의 MI lower bound가 tighter. 그래서 SimCLR v1은 4096+ batch size 사용.

</details>

**문제 2** (심화): InfoNCE loss $\mathcal{L} = -\log \frac{\exp(s_p/\tau)}{\sum \exp(s_k/\tau)}$에서 temperature $\tau$의 역할은? 너무 작거나 큰 $\tau$의 효과를 설명하라.

<details>
<summary>힌트 및 해설</summary>

$\tau$는 softmax의 "sharpness"를 조절:

- **$\tau \to 0$**: softmax가 argmax에 가까워짐. 가장 hard negative에만 집중 → gradient 매우 sharp.
  - 장점: hard negative에 강한 signal.
  - 단점: easy negative는 무시 → representation diversity 부족.
- **$\tau$ 크다 (e.g. 1.0)**: softmax가 uniform에 가까움. 모든 negative에 고르게 집중.
  - 장점: gradient smooth.
  - 단점: hard negative 구별 못함 → 약한 학습 신호.
- **$\tau = 0.1$ (SimCLR default)**: 중간. Hard negative에 focus하면서도 다양성 유지.

**$\tau$의 Temperature와 Regularization 해석** (Wang 2021):
- 작은 $\tau$ = strong hardness-aware loss.
- 큰 $\tau$ = weak / easier loss, gradient variance 작음.

**Ablation (SimCLR Table)**:
- $\tau = 0.05$: 67.0% linear eval.
- $\tau = 0.1$: **69.3%** (best).
- $\tau = 0.5$: 57.5%.

Temperature는 CLIP 등 large-scale contrastive에서도 **learned parameter**로 두어 자동 조정 (CLIP's temperature scale 학습).

</details>

**문제 3** (이론-실전): BYOL이 negative 없이 작동하는 핵심은 무엇인가? "Trivial collapse" (모든 $f$가 constant로 수렴)을 어떻게 방지하는가?

<details>
<summary>힌트 및 해설</summary>

**Trivial collapse**: $f(x) = c$ (constant) for all $x$면 두 view의 representation이 identical → loss = 0. 하지만 representation이 informative하지 않음.

**BYOL의 non-collapse 메커니즘**:

1. **Asymmetric design**: online encoder $f_\theta$ + target encoder $f_\xi$가 다르다. Target은 EMA (느린 업데이트), online은 gradient update (빠른 업데이트).
2. **Predictor $p_\theta$**: online에만 존재하는 추가 MLP. $p_\theta$가 "target을 예측"하려 함.
3. **Stop-gradient on target**: target을 학습시키지 않고 fixed로 취급.

**왜 collapse 안 되는가**:
- 만약 online이 collapse하려 하면, target은 EMA로 과거 non-collapse 상태를 기억 → online은 "과거 자신"에 fit해야 함 → collapse 막힘.
- Predictor가 identity가 아니므로 $f_\theta = f_\xi$로 수렴하지 않고 "예측 task"를 학습.

**후속 연구** (Chen 2021 SimSiam): Stop-gradient만으로도 BYOL 수준 — EMA 불필요. 이는 **더 단순한 collapse-avoidance** mechanism이 있음을 시사.

**직관**: Contrastive learning의 "negative pair"가 explicit diversity를 강제한다면, BYOL/SimSiam는 **asymmetric architecture**가 implicit diversity 강제.

**실전 고려**:
- BYOL은 SimCLR 대비 batch size 작아도 OK (256도 효과).
- 복잡한 hyperparameter tuning 필요 (predictor 구조, EMA rate 등).
- Downstream 성능 비슷, 상황별 선택.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. CutMix · RandAugment](./04-cutmix-randaugment.md) | [📚 README로 돌아가기](../README.md) | [Chapter 5 → 01. Label Smoothing ▶](../ch5-label-calibration/01-label-smoothing.md) |

</div>
