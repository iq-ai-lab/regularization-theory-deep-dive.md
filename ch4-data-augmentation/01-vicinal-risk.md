# 01. Vicinal Risk Minimization (Chapelle et al. 2000)

## 🎯 핵심 질문

- **Empirical Risk Minimization (ERM)** vs **Vicinal Risk Minimization (VRM)** — 무엇이 다른가?
- 왜 data augmentation이 **vicinity distribution**의 특수 경우로 통일되는가?
- Rotation/flip augmentation과 Mixup이 같은 VRM framework에서 어떻게 기술되는가?
- VRM이 ERM보다 generalization이 좋다는 이론적 근거는?

---

## 🔍 왜 이 framework가 필요한가

Data augmentation은 실전에서 광범위하게 쓰이지만, **이론적 통일**이 부족하다:

- Random crop, flip, rotation, color jitter — 각각을 **독립 trick**으로 배운다.
- Mixup, CutMix — "보간이 왜 유용한가"가 임시 설명.
- Self-supervised의 두 view — "augmentation이 invariance 주입"이라는 말은 정성적.

Chapelle, Weston, Bottou, Vapnik 2000의 **VRM**은 이 모두를 **측도론적으로 통일**한다:

1. ERM: 데이터 분포를 **Dirac delta의 empirical 합**으로 근사.
2. VRM: Dirac delta 대신 **vicinity distribution** — 각 data point 주변의 "확산".
3. Augmentation: "Vicinity를 어떻게 정의하는가"의 선택.

**결과**:
- Random flip = "각 이미지의 horizontal flip을 vicinity에 포함".
- Mixup = "두 sample을 잇는 선분을 vicinity".
- SimCLR = "augmentation set의 모든 조합이 vicinity".

이 문서는 VRM의 measure-theoretic framework을 세우고 이후 장(Mixup, CutMix, Contrastive)에서 이를 적용한다.

---

## 📐 수학적 선행 조건

- 측도론 기초: Dirac delta measure $\delta_x$, empirical measure $\hat{P}_n$
- [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive): ERM, generalization bound
- 확률: distribution mixture, convolution
- Vapnik의 statistical learning theory 배경

---

## 📖 직관적 이해

### ERM: 점을 그대로 쓰기

Training set $\{(x_i, y_i)\}_{i=1}^n \sim P$. ERM:

$$\hat{L}_n(f) = \frac{1}{n}\sum_i \ell(f(x_i), y_i)$$

수학적으로 $\hat{L}_n(f) = \mathbb{E}_{(x, y) \sim \hat{P}_\delta}[\ell(f(x), y)]$, 여기서:

$$\hat{P}_\delta = \frac{1}{n}\sum_i \delta_{(x_i, y_i)}$$

— **empirical delta measure**. 각 data point가 Dirac delta(점 mass)로.

### VRM: 점을 퍼뜨리기

VRM은 $\delta_{(x_i, y_i)}$ 대신 **vicinity distribution** $\mathcal{D}_{x_i, y_i}$를 쓴다:

$$\hat{P}_{\text{VRM}} = \frac{1}{n}\sum_i \mathcal{D}_{x_i, y_i}$$

각 data point 주변의 "비슷한 점들"의 분포. VRM:

$$\hat{L}_{\text{VRM}}(f) = \frac{1}{n}\sum_i \mathbb{E}_{(\tilde x, \tilde y) \sim \mathcal{D}_{x_i, y_i}}[\ell(f(\tilde x), \tilde y)]$$

### Augmentation은 $\mathcal{D}$의 선택

| Augmentation | Vicinity $\mathcal{D}_{x, y}$ |
|-------------|----------------------------|
| No augment | $\delta_{(x, y)}$ (ERM) |
| Gaussian noise | $(\mathcal{N}(x, \sigma^2), \delta_y)$ |
| Horizontal flip | $\tfrac{1}{2}\delta_{(x, y)} + \tfrac{1}{2}\delta_{(\text{flip}(x), y)}$ |
| Rotation | $\delta_{(\text{rot}_\theta(x), y)}$ for $\theta \sim U(-\alpha, \alpha)$ |
| Mixup | $\{(x_i \lambda + x_j (1-\lambda), y_i \lambda + y_j (1-\lambda)) : \lambda \sim \text{Beta}(\alpha, \alpha), j \in [n]\}$ |

모든 augmentation이 **vicinity distribution의 한 선택**으로 정식화.

### 왜 VRM이 더 좋은가

ERM은 point mass에서만 좋은 fit을 요구 → overfitting. VRM은 **"neighborhood에서 일관된 예측"** 을 요구 → smoother decision boundary → generalization.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Empirical Delta Measure

Training set $S = \{(x_i, y_i)\}$에 대해:

$$\hat{P}_\delta(A) := \frac{1}{n}\sum_i \mathbb{1}[(x_i, y_i) \in A] = \frac{1}{n}\sum_i \delta_{(x_i, y_i)}(A)$$

### 정의 1.2 — Vicinity Distribution

각 $(x_i, y_i) \in S$에 대한 확률측도 $\mathcal{D}_{x_i, y_i}$를 "**vicinity**"라 한다. 조건:
- $\mathcal{D}_{x_i, y_i}$는 $\mathcal{X} \times \mathcal{Y}$ 위의 measure.
- 보통 $(x_i, y_i)$에 **concentrated** — 하지만 exact $\delta$일 필요 없음.

### 정의 1.3 — Vicinal Distribution and Risk

$$\hat{P}_{\text{VRM}} := \frac{1}{n}\sum_i \mathcal{D}_{x_i, y_i}$$

$$\hat{L}_{\text{VRM}}(f) := \mathbb{E}_{(x, y) \sim \hat{P}_{\text{VRM}}}[\ell(f(x), y)] = \frac{1}{n}\sum_i \mathbb{E}_{(\tilde x, \tilde y) \sim \mathcal{D}_{x_i, y_i}}[\ell(f(\tilde x), \tilde y)]$$

### 정의 1.4 — Augmentation as VRM

Augmentation $A: \mathcal{X} \to 2^{\mathcal{X}}$ (data point를 possible augmented version의 집합으로 매핑). Vicinity:

$$\mathcal{D}_{x_i, y_i}(\cdot) = \text{Uniform}_{A(x_i)}(\cdot) \otimes \delta_{y_i}$$

### 정리 1.5 — VRM Generalization Bound

특정 조건(Chapelle 2000 Theorem 1) 하에:

$$L(f) \leq \hat{L}_{\text{VRM}}(f) + O\left(\frac{\text{complexity}}{\sqrt{n}}\right) + \underbrace{d(\hat{P}_{\text{VRM}}, P)}_{\text{VRM과 true dist의 차이}}$$

VRM이 true distribution $P$에 더 가까우면 bound가 tighter. **좋은 vicinity 선택** 이 generalization을 개선.

### 정의 1.6 — Mixup as VRM (Zhang et al. 2018)

$$\mathcal{D}_{x_i, y_i}^{\text{Mixup}} = \mathbb{E}_{\lambda \sim \text{Beta}(\alpha, \alpha)}[\text{Uniform}_j \delta_{(\lambda x_i + (1-\lambda)x_j, \lambda y_i + (1-\lambda)y_j)}]$$

**Mixup은 VRM의 특수 경우** — vicinity를 "다른 sample과의 선분"으로 정의.

---

## 🔬 수학적 유도

### ERM = VRM with Delta Vicinity

$\mathcal{D}_{x_i, y_i} = \delta_{(x_i, y_i)}$면:

$\mathbb{E}_{(\tilde x, \tilde y) \sim \delta_{(x_i, y_i)}}[\ell(f(\tilde x), \tilde y)] = \ell(f(x_i), y_i)$.

$\hat{L}_{\text{VRM}}(f) = \frac{1}{n}\sum_i \ell(f(x_i), y_i) = \hat{L}_n(f) = $ ERM loss. $\square$

### Gaussian Noise = Convolution in Input Space

$\mathcal{D}_{x, y} = \mathcal{N}(x, \sigma^2 I) \otimes \delta_y$:

$$\hat{L}_{\text{VRM}}(f) = \frac{1}{n}\sum_i \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, \sigma^2 I)}[\ell(f(x_i + \varepsilon), y_i)]$$

Taylor 확장 ($\ell$ smooth):

$\approx \frac{1}{n}\sum_i [\ell(f(x_i), y_i) + \tfrac{\sigma^2}{2} \nabla_x^2 \ell(\cdot) \text{ trace}]$

**Gaussian augmentation = ERM + L2-like input-gradient penalty**. 이는 adversarial training의 first-order approximation과 연결.

### Mixup의 "Linear Decision Boundary"

Zhang 2018: Mixup으로 훈련된 분류기는 **두 sample 사이 선분 위에서 linear interpolation된 라벨 예측**. 이는 "**smoothness constraint**"로 작용:

$$f(\lambda x_i + (1-\lambda) x_j) \approx \lambda f(x_i) + (1-\lambda) f(x_j)$$

Decision boundary가 두 클러스터 사이에서 **smoothly transition**. Sharp cliff 대신 gradual slope → robustness.

---

## 💻 실험으로 효과 검증

### 실험 1 — ERM vs 단순 Gaussian augmentation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Toy: 2D classification
torch.manual_seed(0)
n = 100
X = torch.cat([torch.randn(n, 2) + torch.tensor([2., 2.]),
               torch.randn(n, 2) + torch.tensor([-2., -2.])])
y = torch.cat([torch.zeros(n), torch.ones(n)]).long()

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

def train_erm(epochs=500):
    net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(net(X), y)
        loss.backward(); opt.step()
    return net

def train_vrm_gaussian(sigma=0.5, epochs=500):
    net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        # augmentation: Gaussian noise
        X_aug = X + sigma * torch.randn_like(X)
        loss = nn.CrossEntropyLoss()(net(X_aug), y)
        loss.backward(); opt.step()
    return net

def plot_boundary(net, title):
    xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
    pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        probs = torch.softmax(net(pts), -1)[:, 1].view(200, 200).numpy()
    plt.contourf(xx, yy, probs, levels=20, cmap='coolwarm', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)

net_erm = train_erm()
net_vrm = train_vrm_gaussian(sigma=0.3)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(axes[0]); plot_boundary(net_erm, 'ERM')
plt.sca(axes[1]); plot_boundary(net_vrm, 'VRM (Gaussian aug)')
plt.tight_layout(); plt.show()
# → ERM은 sharper boundary, VRM은 smoother (overfitting 완화)
```

### 실험 2 — Empirical measure → Vicinal measure의 시각화

```python
# Empirical delta: 각 point가 점
# Vicinal: 각 point 주변에 "분포"
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ERM (delta)
axes[0].scatter(X[:, 0], X[:, 1], c=y, s=50)
axes[0].set_title(r'ERM: $\hat P_\delta$')
axes[0].set_xlim(-6, 6); axes[0].set_ylim(-6, 6)

# Gaussian VRM
for xi, yi in zip(X, y):
    for _ in range(5):
        noise = 0.3 * np.random.randn(2)
        axes[1].scatter(xi[0] + noise[0], xi[1] + noise[1], c=['red' if yi==1 else 'blue'], s=5, alpha=0.4)
axes[1].scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='black')
axes[1].set_title(r'Gaussian VRM: $\mathcal{N}(x_i, \sigma^2 I)$')

# Mixup VRM (선분으로 연결)
for _ in range(200):
    i, j = np.random.randint(0, len(X), 2)
    lam = np.random.beta(0.4, 0.4)
    x_new = lam * X[i] + (1 - lam) * X[j]
    axes[2].scatter(x_new[0], x_new[1], s=5, alpha=0.2, c='purple')
axes[2].scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='black')
axes[2].set_title('Mixup VRM: convex combinations')

plt.tight_layout(); plt.show()
```

### 실험 3 — Rotation augmentation을 VRM으로 formalize

```python
from torchvision.transforms import RandomRotation

# MNIST에 대해 rotation augmentation
# Vicinity = {rot_theta(x) : theta ~ Uniform(-10°, 10°)}
# VRM loss: E_theta[loss(f(rot_theta(x)), y)]

# 수학적 equivalence 확인:
# f_VRM(x) = argmin E_theta[loss(f, rot(x))]
# 해석: f가 rotation invariant한 feature를 학습
```

---

## 🔗 실전 활용

### Vicinity 설계 가이드

**좋은 $\mathcal{D}_{x, y}$의 조건**:
1. **Label-preserving**: $\tilde{x}$가 original label $y$를 공유해야 함 (rotation, crop, noise).
2. **True distribution $P$에 가까움**: Vicinity가 natural data variation을 반영.
3. **Computationally cheap**: 훈련 중 매 iteration에서 샘플 가능.

### 예시별 분석

- **Random crop**: vicinity = "crop된 모든 version". $P$에 가까움 (자연 이미지는 crop-invariant 정보 많음).
- **Mixup**: vicinity = "다른 sample과의 선분". $P$에서 벗어남 (실제로 두 이미지의 mix는 not natural). 그러나 smoothness inductive bias로 유용.
- **CutMix**: vicinity = "patch 교환". $P$에 약간 더 가까움 (real 이미지의 일부로 보임).
- **AutoAugment**: RL로 $\mathcal{D}$ 자체를 학습.

### VRM이 ERM보다 좋을 때 (그리고 그렇지 않을 때)

**좋은 경우**:
- Data가 적음 (augmentation이 effective dataset size 증가).
- Natural invariance가 task에 맞음 (vision의 rotation, flip).
- Smooth decision boundary 선호 (Mixup).

**나쁜 경우**:
- Augmentation이 label-changing (digit rotation 180°이면 6→9, 9→6).
- Vicinity가 $P$와 멀어 semantic mismatch.
- Overfit이 문제 아니라 underfit이 문제 (small model).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Vicinity가 true distribution 근사 | 잘못 설계하면 harmful inductive bias |
| Label preserving augmentation | 예외 많음 (mirror로 text 뒤집기 등) |
| Independent augmentation per sample | 실전에서는 batch-level correlation 가능 |
| Symmetric measure $\mathcal{D}_{x, y}$ | Non-symmetric도 가능 하지만 이론 복잡 |
| Static vicinity during training | Dynamic vicinity (curriculum, RL-learned)도 연구 활발 |

---

## 📌 핵심 정리

$$\boxed{\hat{L}_{\text{VRM}}(f) = \frac{1}{n}\sum_i \mathbb{E}_{(\tilde x, \tilde y) \sim \mathcal{D}_{x_i, y_i}}[\ell(f(\tilde x), \tilde y)]}$$

| 개념 | 의미 |
|------|------|
| **ERM** | $\mathcal{D} = \delta$ (Dirac delta vicinity) |
| **VRM** | Generic vicinity distribution $\mathcal{D}_{x, y}$ |
| **Augmentation** | $\mathcal{D}$의 구체적 선택 |
| **Generalization** | 좋은 $\mathcal{D}$가 $P$ 근사 → tighter bound |
| **다음 질문** | Augmentation의 invariance 주입 효과 → Ch4-02 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Gaussian noise augmentation $\mathcal{D}_{x, y} = \mathcal{N}(x, \sigma^2 I) \otimes \delta_y$에서 $\sigma = 0$이면 VRM은 무엇과 같은가?

<details>
<summary>힌트 및 해설</summary>

$\mathcal{N}(x, 0 \cdot I) = \delta_x$. 따라서 $\mathcal{D}_{x, y} = \delta_{(x, y)}$ → VRM = ERM.

Gaussian noise augmentation이 **$\sigma$를 연속 매개변수**로 하여 ERM($\sigma = 0$)과 strong augmentation($\sigma$ 큼) 사이를 interpolate. $\sigma$ tuning이 augmentation 강도 조절.

</details>

**문제 2** (심화): VRM의 generalization bound (정리 1.5)에서 "vicinity가 $P$에 가까울수록 bound tighter"를 증명하라. 힌트: triangle inequality on risk.

<details>
<summary>힌트 및 해설</summary>

$L(f)$ = true risk, $\hat{L}_{\text{VRM}}(f)$ = VRM empirical risk.

$|L(f) - \hat{L}_{\text{VRM}}(f)|$에 대한 분해:

$\leq |L(f) - \mathbb{E}_{\hat{P}_{\text{VRM}}}[\ell]| + |\mathbb{E}_{\hat{P}_{\text{VRM}}}[\ell] - \hat{L}_{\text{VRM}}|$

첫 항: true $P$와 vicinal $\hat{P}_{\text{VRM}}$ 사이 거리 (Wasserstein 등) — $\ell$ Lipschitz면 $L_\ell \cdot W_1(P, \hat{P}_{\text{VRM}})$.

둘째 항: concentration inequality로 bounded by $O(1/\sqrt{n})$.

**결론**: $\hat{P}_{\text{VRM}}$이 $P$에 가까우면(첫 항 작음) total bound가 tighter. "Good vicinity choice"의 정량적 의미.

**예**: Rotation augmentation이 자연 이미지 분포에 잘 맞으므로 $W_1(P, \hat{P}_{\text{VRM}})$ 작음 → vision에서 효과적. Text에 rotation은 $P$와 크게 벗어남 → 효과 없음.

</details>

**문제 3** (이론-실전): Contrastive learning (SimCLR 등)에서 "두 augmented view" 접근은 VRM의 어떤 vicinity에 해당하는가? Self-supervised loss가 왜 invariance learning과 연결되는가?

<details>
<summary>힌트 및 해설</summary>

**SimCLR**: 각 $x$에 대해 두 random augmentation $t_1, t_2$ 적용 → $(x^{(1)}, x^{(2)})$ positive pair.

**VRM 관점**: Vicinity $\mathcal{D}_x = \text{Uniform}_{T}$ (augmentation 집합 $T$). Positive pair $(x^{(1)}, x^{(2)}) \in T \times T$.

**InfoNCE loss**:
$$\mathcal{L} = -\mathbb{E}[\log \frac{\exp(\text{sim}(z^{(1)}, z^{(2)})/\tau)}{\sum_k \exp(\text{sim}(z^{(1)}, z^{(k)})/\tau)}]$$

이것이 "두 augmented view의 representation이 가깝도록" 강제 → **augmentation-invariant feature**.

VRM 관점에서 SimCLR의 목적: $f(t_1(x)) \approx f(t_2(x))$ for all $t_1, t_2 \in T$. 즉 **$T$-invariant representation**. Data augmentation이 "invariance specification" 역할.

**Chen 2020의 관찰**: $T$의 선택이 representation quality를 결정. Color jitter + crop + Gaussian blur가 특히 효과적 — 이 invariance들이 downstream classification에 유용한 feature를 선택하게 강제.

이는 VRM의 "vicinity 설계"가 contrastive learning에서도 핵심임을 보임. Ch4-05에서 더 자세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Chapter 3 → 06. RMSNorm](../ch3-normalization/06-rmsnorm-modern.md) | [📚 README로 돌아가기](../README.md) | [02. Invariance Injection ▶](./02-invariance-injection.md) |

</div>
