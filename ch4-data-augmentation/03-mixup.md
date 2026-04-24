# 03. Mixup (Zhang et al. 2018)

## 🎯 핵심 질문

- Mixup의 정의 $\tilde{x} = \lambda x_i + (1-\lambda) x_j$는 어떻게 VRM의 특수 경우로 기술되는가?
- 왜 **Beta(α, α)** 분포가 $\lambda$의 sampling 분포로 선택되는가?
- Mixup이 어떻게 **linear decision boundary**를 강제하는가?
- **Calibration 개선** (Thulasidasan 2019)의 메커니즘은?
- **Manifold Mixup** (Verma 2019)은 hidden layer에서 보간 — 왜 더 강력한가?

---

## 🔍 왜 Mixup이 특별한가

Ch4-01은 VRM framework을, Ch4-02는 group-based invariance를 주었다. **Mixup**은 이 둘 어디에도 딱 들어맞지 않는 **"가상의 data point"** augmentation:

1. **Group action이 아님** — $\lambda x + (1-\lambda) x'$는 어떤 group의 orbit도 아니다.
2. **원본 manifold을 떠남** — 두 이미지의 알파 블렌딩은 "자연 이미지"가 아님.
3. 그럼에도 **generalization 개선, calibration 개선**을 경험적으로 보임.

Zhang et al. 2018의 설명: Mixup은 **linear interpolation을 강제**하는 vicinity.

이 문서는 Mixup의 네 측면:
1. **VRM 기술**: convex combination vicinity.
2. **Linear boundary**: 왜 smooth한지.
3. **Calibration**: confidence overestimation 방지.
4. **Manifold Mixup**: hidden representation에서의 보간.

---

## 📐 수학적 선행 조건

- Ch4-01: VRM framework
- Beta 분포 $\text{Beta}(\alpha, \beta)$의 density와 성질
- Cross-entropy loss
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Beta 분포의 모드, 평균

---

## 📖 직관적 이해

### Mixup의 정의

두 random sample $(x_i, y_i), (x_j, y_j)$와 $\lambda \sim \text{Beta}(\alpha, \alpha)$:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

**라벨도 mix**. $y$가 one-hot이면 $\tilde{y}$는 soft label (두 class에 확률 mass 분배).

Loss: $\ell(f(\tilde{x}), \tilde{y})$ — cross-entropy는 soft label과 자연스럽게 호환.

### Beta(α, α)의 shape

| $\alpha$ | 분포 모양 |
|----------|---------|
| $\alpha \to 0$ | $\text{Bernoulli}(0.5)$ 근사 — 0 또는 1에 mass (mix 없음) |
| $\alpha = 1$ | Uniform(0, 1) |
| $\alpha = 0.2$ (Mixup default) | U-shape — 0이나 1에 가까운 $\lambda$ 선호 (약한 mix) |
| $\alpha = 1.0$ | 평평 — 어떤 $\lambda$도 동등 |
| $\alpha \to \infty$ | $\delta_{0.5}$ — 균등 mix |

$\alpha = 0.2$가 ImageNet에서 최적 (Zhang 2018). CIFAR에서는 $\alpha = 1$이 자주 씀. 보통 $\alpha \in [0.1, 0.4]$.

### Linear Decision Boundary

$\tilde{x}$가 두 sample 사이 선분 위에 있고, $\tilde{y}$가 라벨의 선형 보간이면:

$$f(\lambda x_i + (1-\lambda) x_j) \approx \lambda f(x_i) + (1-\lambda) f(x_j)$$

즉 $f$는 선분 위에서 **linear** — decision boundary가 두 cluster 사이 **매끄럽게** 변화. Sharp cliff이 없어 adversarial/corruption robustness 증가.

### Calibration 개선

현대 NN은 over-confident (Ch5-04). Mixup은:
- Soft label $\tilde{y} = \lambda y_i + (1-\lambda) y_j$로 훈련 → softmax가 "**섞인 sample**"에도 일관된 확률 출력.
- Cross-entropy가 $(1, 0)$에 비해 $(0.7, 0.3)$ 같은 soft target을 갖게 됨 → target logit을 **무한히 밀지 않음**.
- 결과: ECE (Expected Calibration Error) 감소.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Mixup Operator

Training set $S = \{(x_i, y_i)\}_{i=1}^n$, $\alpha > 0$. Mixup sample:

$$(x_\lambda, y_\lambda) := (\lambda x_i + (1-\lambda) x_j, \lambda y_i + (1-\lambda) y_j), \quad \lambda \sim \text{Beta}(\alpha, \alpha), \ j \sim U([n])$$

Mixup loss:

$$\hat{L}_{\text{Mixup}}(f) = \mathbb{E}_{i, j, \lambda}[\ell(f(x_\lambda), y_\lambda)]$$

### 정의 3.2 — Mixup as VRM

Vicinity:

$$\mathcal{D}_{x_i, y_i}^{\text{Mixup}} = \mathbb{E}_j \mathbb{E}_\lambda [\delta_{(\lambda x_i + (1-\lambda) x_j, \lambda y_i + (1-\lambda) y_j)}]$$

각 $x_i$의 vicinity는 "**다른 sample과의 convex combination의 분포**".

### 정리 3.3 — Mixup의 Linear Interpolation Property

훈련 후 $f^*$에 대해 다음이 근사적으로 성립 (Zhang 2018 Figure 3):

$$f^*(\lambda x_i + (1-\lambda) x_j) \approx \lambda f^*(x_i) + (1-\lambda) f^*(x_j)$$

증명: Loss $\ell(f(x_\lambda), y_\lambda)$를 최소화하는 $f$는 $f$가 linear일 때 0. 따라서 $f$를 모든 $\lambda$에 대해 linear로 유도.

### 정리 3.4 — Guo-Pleiss 2017 Calibration Framework

Mixup 훈련 모델의 ECE:

$$\text{ECE}(f_{\text{Mixup}}) < \text{ECE}(f_{\text{ERM}})$$

경험적으로 CIFAR-10 ResNet-50에서 ECE 15% → 3% (Thulasidasan et al. 2019).

### 정의 3.5 — Manifold Mixup (Verma et al. 2019)

Hidden representation에서 보간:

$$\tilde{h}_\ell = \lambda h_\ell(x_i) + (1-\lambda) h_\ell(x_j)$$

$\ell$은 random layer. 이는 representation space에서 VRM — "learned feature이 convex combination에 대해 smooth"를 강제.

### 정리 3.6 — Mixup과 Adversarial Robustness의 관계 (Zhang 2018)

Mixup 훈련 모델은 small adversarial perturbation에 더 robust:

$$\text{Accuracy under }\varepsilon\text{-perturbation}(f_{\text{Mixup}}) > \text{Accuracy}(f_{\text{ERM}})$$

이유: Linear decision boundary가 일반적으로 margin이 큼.

---

## 🔬 수학적 유도

### Mixup의 Regularizing Effect — Taylor 전개

$\lambda \sim \text{Beta}(\alpha, \alpha)$이면 $\mathbb{E}[\lambda] = 0.5$, $\text{Var}(\lambda) = 1/(4(2\alpha+1))$. 작은 $\alpha$에서 variance 큼.

$f(x_\lambda)$를 $\lambda = 0$ 또는 $1$ 주변에서 Taylor:

$f(\lambda x_i + (1-\lambda) x_j) \approx f(x_i) + (1-\lambda)(x_j - x_i) \cdot \nabla f(x_i) + \frac{(1-\lambda)^2}{2}(x_j - x_i)^T H_f(x_i)(x_j - x_i) + \cdots$

Loss Taylor expansion in $\ell(f, y)$ (supposing $\ell = -\log f_y$):

Extra regularization term (Zhang 2018 Appendix): $\mathbb{E}_{i,j,\lambda}[\ell''(\cdot) \cdot (\text{interpolation curvature})^2]$.

즉 Mixup은 **second-order curvature penalty** (Dao 2019의 first-order Jacobian과 유사하지만 different axis).

### Why Beta(α, α) with small α

**$\alpha \to 0$**: $\lambda \in \{0, 1\}$ 거의 확실 → no mixing (거의 ERM).  
**$\alpha \to \infty$**: $\lambda \approx 0.5$ → 균등 mixing, **soft label always (0.5, 0.5)** — 정보 손실.  
**$\alpha \approx 0.2$**: $\lambda$가 대부분 0.1 이하 또는 0.9 이상 → 약한 mixing. $(x_i, y_i)$를 대부분 주로 유지하면서 가끔 강하게 mix.

이 "**gentle interpolation**"이 generalization 효과적.

### Manifold Mixup의 표현력

원 Mixup: input space $\mathcal{X}$에서 선형 보간 → pixel-level mix (unnatural).

Manifold Mixup: 학습된 hidden $h$에서 보간 → **semantic-level mix**. Hidden manifold에서의 linear interpolation은 훨씬 natural (학습된 feature가 smooth).

Verma 2019: Manifold Mixup으로 훈련된 모델의 **hidden representation이 더 flat** — generalization 개선 + adversarial robustness 증가.

---

## 💻 실험으로 효과 검증

### 실험 1 — Mixup 구현 (classification)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mixup_batch(x, y, alpha=0.2):
    """Mixup with Beta(α, α). y는 class index (not one-hot)."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam

def mixup_loss(logits, y_a, y_b, lam):
    return lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)

# 훈련 루프
for x, y in loader:
    x_mix, y_a, y_b, lam = mixup_batch(x, y, alpha=0.2)
    logits = model(x_mix)
    loss = mixup_loss(logits, y_a, y_b, lam)
    loss.backward(); opt.step()
```

### 실험 2 — Toy 2D: Mixup의 decision boundary smoothing

```python
import matplotlib.pyplot as plt

torch.manual_seed(0)
# 2 cluster
X_a = torch.randn(50, 2) + torch.tensor([2., 2.])
X_b = torch.randn(50, 2) + torch.tensor([-2., -2.])
X = torch.cat([X_a, X_b])
y = torch.cat([torch.zeros(50), torch.ones(50)]).long()

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, 2))
    def forward(self, x): return self.net(x)

def train(use_mixup=False, alpha=0.2, epochs=1000):
    net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        if use_mixup:
            x_m, y_a, y_b, lam = mixup_batch(X, y, alpha)
            loss = mixup_loss(net(x_m), y_a, y_b, lam)
        else:
            loss = F.cross_entropy(net(X), y)
        loss.backward(); opt.step()
    return net

def plot_boundary(net, title, ax):
    xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
    pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        probs = torch.softmax(net(pts), -1)[:, 1].view(200, 200).numpy()
    ax.contourf(xx, yy, probs, levels=20, cmap='coolwarm', alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=30)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_boundary(train(False), 'ERM', axes[0])
plot_boundary(train(True, alpha=0.4), 'Mixup (α=0.4)', axes[1])
plt.tight_layout(); plt.show()
```

**관찰**: ERM의 boundary는 data에 "달라붙음" sharp. Mixup의 boundary는 두 cluster 사이 중앙을 **smoothly** 가로지름.

### 실험 3 — Calibration measurement

```python
def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    preds = probs.argmax(1)
    confidences = probs.max(1)
    accuracies = (preds == labels).float()
    ece = 0
    for b in range(n_bins):
        mask = (confidences > bins[b]) & (confidences <= bins[b+1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean().item()
            avg_acc = accuracies[mask].mean().item()
            ece += mask.float().mean().item() * abs(avg_conf - avg_acc)
    return ece

# ERM vs Mixup 모델의 ECE 비교
# ECE_ERM >> ECE_Mixup (일반적으로 3-5배 차이)
```

### 실험 4 — Beta 분포 $\alpha$ 영향

```python
alphas = [0.1, 0.4, 1.0, 4.0]
fig, axes = plt.subplots(1, 4, figsize=(15, 3))
for ax, alpha in zip(axes, alphas):
    samples = np.random.beta(alpha, alpha, 10000)
    ax.hist(samples, bins=50, density=True)
    ax.set_title(f'Beta({alpha}, {alpha})')
    ax.set_xlim(0, 1)
plt.show()
# α=0.1: 강한 U-shape (0 또는 1에 몰림)
# α=4: bell 모양 (0.5 근처)
```

### 실험 5 — Manifold Mixup 구현

```python
class ManifoldMixupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 2)
        ])
    def forward(self, x, mixup_layer=None, lam=None, idx=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == mixup_layer:
                x = lam * x + (1 - lam) * x[idx]
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

# Manifold Mixup: random layer에서 보간
def train_manifold_mixup(alpha=0.2, epochs=1000):
    net = ManifoldMixupNet(); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(X.size(0))
        mix_layer = np.random.randint(0, len(net.layers))
        logits = net(X, mix_layer, lam, idx)
        loss = mixup_loss(logits, y, y[idx], lam)
        loss.backward(); opt.step()
    return net
```

---

## 🔗 실전 활용

### Mixup 활용 recipe

| Task | Mixup $\alpha$ | 기타 augmentation |
|------|---------------|--------------------|
| CIFAR-10 | 1.0 | Random crop + flip |
| CIFAR-100 | 0.4 | 같음 |
| ImageNet | 0.2 | RandAugment + CutMix도 함께 |
| Medical (small data) | 0.4~1.0 | 보수적 flip만 |
| NLP (label smoothing으로 대체) | — | Not typical |

### 언제 Mixup이 harmful

- **Small discrete class** (e.g. digit classification 0-9): 섞인 디짓이 의미 없음, 훈련 불안정.
- **Time series / sequential**: 두 시계열의 alpha blend는 해석 불가.
- **Object detection**: Mixup은 이미지 구조 파괴 → regression target 왜곡. CutMix가 선호.

### Mixup과 다른 기법의 상호작용

- **Label Smoothing (Ch5-01)**: Mixup의 soft label과 중복 — $\alpha$ 작게 두거나 LS 제외.
- **BatchNorm**: Mixed image의 batch statistics가 단일 이미지와 다름 → BN running stats 수렴 느릴 수 있음.
- **CutMix**: 둘 함께 쓰기도 (매 epoch 랜덤 선택) — 다양성 확보.

### Advanced Mixup 변종

- **Manifold Mixup** (Ch 본 문서): hidden interpolation.
- **PuzzleMix** (Kim 2020): saliency-based patch 조합.
- **MixMo** (Ramé 2021): multi-input multi-output ensemble.
- **Adversarial Mixup Resampling** (Beckham 2019): adversarial로 hard mix sample 생성.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear interpolation이 의미 | Object detection, segmentation에서 unnatural |
| One-hot label 가정 | Hierarchical / multi-label에서는 label mixing 미묘 |
| Random pair 선택 | Hard example mining과 결합 시 더 나을 수도 |
| Input-space mixing | Manifold mixup이 더 자연스럽지만 구현 복잡 |
| $\alpha$ fixed | Curriculum style dynamic $\alpha$도 연구 중 |

**주의**: "Mixup으로 모든 것이 개선된다"는 허구. Object detection, face recognition 같은 task에서는 CutMix 나 전혀 쓰지 않음이 나을 수도.

---

## 📌 핵심 정리

$$\boxed{(\tilde x, \tilde y) = \lambda(x_i, y_i) + (1-\lambda)(x_j, y_j), \ \lambda \sim \text{Beta}(\alpha, \alpha)}$$

| 개념 | 의미 |
|------|------|
| **Mixup** | VRM의 convex combination vicinity |
| **Beta(α, α)** | $\alpha$ 작을수록 약한 mix (2018 default α=0.2) |
| **Linear boundary** | 훈련 목표가 $f$를 선분 위에서 linear로 |
| **Calibration 개선** | Soft label → ECE 감소 |
| **Manifold Mixup** | Hidden에서 보간 — 더 강력 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $x_i = (1, 2), x_j = (3, 4), y_i = 0, y_j = 1, \lambda = 0.7$일 때 Mixup sample $(\tilde x, \tilde y)$는? ($y$는 one-hot으로 표현)

<details>
<summary>힌트 및 해설</summary>

- $\tilde x = 0.7 \cdot (1, 2) + 0.3 \cdot (3, 4) = (0.7 + 0.9, 1.4 + 1.2) = (1.6, 2.6)$.
- $y_i = (1, 0), y_j = (0, 1)$ one-hot. $\tilde y = 0.7 \cdot (1, 0) + 0.3 \cdot (0, 1) = (0.7, 0.3)$.

**Soft label** $(0.7, 0.3)$. Cross-entropy $= -0.7 \log p_0 - 0.3 \log p_1$ where $p$ is model's softmax output.

이 sample $(\tilde x, \tilde y)$로 훈련하면 $f(\tilde x)$가 class 0에 70%, class 1에 30% 확률을 주도록 유도.

</details>

**문제 2** (심화): Beta(0.2, 0.2)는 $\lambda$가 대부분 0이나 1에 가까움 — 왜 ImageNet에서 최적인가? 더 큰 $\alpha$(e.g. 1.0)는 왜 harmful할 수 있는가?

<details>
<summary>힌트 및 해설</summary>

**$\alpha = 0.2$의 이점**:
- 대부분 $\lambda \approx 0$ 또는 $\approx 1$ → sample이 거의 원본 $x_i$ 또는 $x_j$에 가까움.
- 가끔 중간 $\lambda \approx 0.5$도 나오지만 드묾 → 강한 mix는 희귀.
- Training set의 "natural distribution"에 가까움. $P$에서 크게 벗어나지 않음.

**$\alpha = 1.0$ (uniform)**:
- $\lambda$가 0.5 근처일 확률 커짐 → **완전 섞인 sample**이 많아짐.
- "두 이미지 50:50 mix"는 자연 분포에 없음 → $P$에서 멀어짐.
- ImageNet 고해상도 이미지에서는 정보 손실 큼.

**$\alpha = 4.0$ 이상**:
- $\lambda \approx 0.5$ 거의 deterministic → 모든 sample이 "중앙 mix".
- 매우 strong regularization이지만 실제 이미지 정보 부족 → underfit.

이는 VRM 관점에서 "**vicinity가 $P$에 가까워야 한다**"는 원칙의 구체화.

CIFAR-10의 작은 이미지 $32 \times 32$에서는 mix artifact가 덜 두드러져 더 강한 $\alpha$ ($0.4 \sim 1.0$)도 OK.

</details>

**문제 3** (이론-실전): Mixup이 adversarial robustness를 향상한다는 주장을 실험으로 검증하려면? FGSM attack을 예로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**FGSM (Fast Gradient Sign Method)**: 입력 $x$에 대해 gradient sign 방향으로 $\epsilon$ perturbation:

$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x L(f(x), y))$

**실험 설정**:
1. 같은 arch (ResNet)을 ERM과 Mixup으로 각각 훈련.
2. 각 모델의 FGSM attack에 대한 accuracy 측정 (다양한 $\epsilon$).
3. Comparison plot: accuracy vs $\epsilon$.

**예상 결과** (Zhang 2018 Table 4 based):
- $\epsilon = 0$ (no attack): 두 모델 비슷.
- $\epsilon = 2/255$: ERM 20% 감소, Mixup 10% 감소 — Mixup이 robust.
- $\epsilon = 8/255$: ERM accuracy 매우 낮음, Mixup 여전히 유의미.

**이유**:
- Mixup의 linear decision boundary → larger margin.
- Input-space smoothness → gradient가 덜 sharp → FGSM's sign gradient가 작은 효과.

**주의**: Mixup은 "certified robustness"가 아님 — strong adversarial attack (PGD 등)에서 여전히 broken. Proper adversarial training (Madry 2018)이 필요. 그러나 Mixup은 low-cost으로 기본 robustness 개선.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Invariance Injection](./02-invariance-injection.md) | [📚 README로 돌아가기](../README.md) | [04. CutMix · RandAugment ▶](./04-cutmix-randaugment.md) |

</div>
