# 02. Invariance Injection

## 🎯 핵심 질문

- Augmentation이 **group-equivariant network** 설계와 어떻게 **기능적으로 동등**한가?
- Dao et al. 2019: augmentation의 **first-order expansion**이 왜 **feature averaging penalty**로 환원되는가?
- Rademacher complexity 관점에서 augmentation이 bound를 어떻게 tighten하는가?
- 언제 architectural invariance가 augmentation보다 선호되는가?

---

## 🔍 왜 "invariance" 관점이 중요한가

Ch4-01의 VRM framework은 "vicinity를 잘 고르자"는 general principle을 준다. 하지만 **왜 rotation invariance, flip invariance가 vision에서 효과적**인가?

**답**: Vision의 natural image는 rotation/flip/translation의 **group action 하에 invariant한 semantic**을 갖는다. "고양이" 라벨은 이미지를 뒤집어도 동일. Augmentation은 이 invariance를 **data로** 주입.

대안은 **architectural equivariance** — group-equivariant CNN (Cohen 2016), SE(2)-ResNet 등 — 네트워크 구조 자체가 group에 equivariant. 

두 접근이 **기능적으로 동등**하다는 것이 최근 이론 (Chen 2020, Dao 2019). 이 문서는:

1. Group theory의 augmentation 기술.
2. Dao 2019: augmentation = first-order expansion에서 **Jacobian-based regularization**.
3. Rademacher complexity의 감소 증명.
4. 실전 선택 (augmentation vs equivariant architecture).

---

## 📐 수학적 선행 조건

- Ch4-01: VRM framework
- [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive): Rademacher complexity
- 기본 group theory: group $G$, group action $G \times \mathcal{X} \to \mathcal{X}$
- 선형대수: Jacobian, first-order Taylor expansion

---

## 📖 직관적 이해

### Group Action과 Invariance

Group $G$가 $\mathcal{X}$에 act: $g \cdot x \in \mathcal{X}$ for $g \in G, x \in \mathcal{X}$.

- **Invariant function** $f$: $f(g \cdot x) = f(x) \ \forall g, x$.
- **Equivariant function** $f$: $f(g \cdot x) = \rho(g) f(x)$ for some representation $\rho$.

Vision 예:
- $G = SO(2)$ (2D rotation): 이미지 분류 → rotation-invariant.
- $G = \mathbb{Z}^2$ (translation): CNN은 translation-equivariant (convolution의 성질).

### Augmentation Injects Invariance

Vicinity를 group orbit으로 정의: $\mathcal{D}_{x} = \text{Uniform}_{\{g \cdot x : g \in G\}}$.

VRM loss에서 $\mathbb{E}_g[\ell(f(g \cdot x), y)]$을 minimize하면 $f$가 $G$-invariant를 학습하도록 유도.

### Dao 2019의 First-Order Expansion

Augmentation 분포가 identity $e \in G$ 주변에 concentrated일 때 (e.g. 작은 rotation angle), 1차 Taylor:

$f(g \cdot x) \approx f(x) + \epsilon \cdot \nabla_g f \big|_{g=e}$

Expected loss의 variance 항: $\text{Var}_g(f(g \cdot x)) \approx \epsilon^2 \|\nabla_g f\|^2$.

결과: **augmentation = gradient-based smoothing** — feature가 group action 방향으로 **gradient 크기를 작게** 유지하도록 penalty.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — $G$-Invariant Loss

$G$-invariant function class $\mathcal{F}_G = \{f : f(g \cdot x) = f(x) \ \forall g \in G\}$. $G$-invariant ERM:

$$\min_{f \in \mathcal{F}_G} \hat{L}_n(f) = \min_{f \in \mathcal{F}_G} \frac{1}{n}\sum_i \ell(f(x_i), y_i)$$

### 정의 2.2 — Augmentation-Induced Vicinity

Group $G$, measure $\mu_G$ on $G$. Vicinity:

$$\mathcal{D}_{x, y} = (g \cdot x) \sim \mu_G \otimes \delta_y$$

### 정리 2.3 — Dao et al. 2019의 First-Order Regularization

Augmentation $g_\epsilon = \text{Identity} + \epsilon \cdot \text{generator}$ (Lie group, 작은 $\epsilon$). Augmented loss:

$$\hat{L}_{\text{aug}}(f) \approx \hat{L}_n(f) + \frac{\epsilon^2}{2} \mathbb{E}_g\left[\|\nabla_g f(x)\|^2\right]$$

즉 augmentation = **ERM + first-order Jacobian-norm penalty**. 후자는 "feature의 group-direction gradient" 제약.

### 정리 2.4 — Rademacher Complexity Reduction

$G$-invariant class $\mathcal{F}_G$의 Rademacher complexity는 full class $\mathcal{F}$의 적어도 $1/|G|$:

$$\mathcal{R}_n(\mathcal{F}_G) \leq \frac{1}{|G|} \mathcal{R}_n(\mathcal{F})$$

$|G|$는 group의 orbit 크기 ($G$ finite이면 order, Lie group은 volume). **Augmentation으로 effective function class 감소** → generalization bound tighter.

### 정리 2.5 — Augmentation ≈ Equivariant Architecture

(Chen, Dobriban, Lee 2020 "A Group-Theoretic Framework for Data Augmentation") Augmentation over $G$로 ERM 최적화하면 수렴 시 $f$가 $G$-invariant에 가까움:

$$f_{\text{aug}}^* \approx \pi_G(f_{\text{ERM}}^*)$$

$\pi_G$는 $G$-invariant subspace로의 projection. 즉 **augmentation = ERM 해를 invariant subspace로 project**.

---

## 🔬 수학적 유도

### 정리 2.3 증명 (Dao 2019)

$\mathcal{L}_{\text{aug}}(f) = \mathbb{E}_{g \sim \mu_G}[\ell(f(g \cdot x), y)]$. $g = e + \epsilon \xi$ (작은 perturbation, $\xi$는 Lie algebra의 tangent vector):

$f(g \cdot x) = f(e \cdot x + \epsilon \xi \cdot x + O(\epsilon^2))$

$= f(x) + \epsilon (\nabla_x f)(x) \cdot (\xi \cdot x) + \frac{\epsilon^2}{2} [\text{quadratic terms}]$

**Loss expansion**:

$\ell(f(g \cdot x), y) = \ell(f(x), y) + \ell'(f(x), y) \cdot (f(g \cdot x) - f(x)) + \frac{1}{2}\ell''(\cdot)(f(g\cdot x) - f(x))^2 + \ldots$

Expectation over $\mu_G$ (assume mean 0, variance $\sigma_g^2$):

$\mathbb{E}_g[\ell] \approx \ell(f(x), y) + \frac{\sigma_g^2}{2}\left[\ell'' \cdot |\nabla_x f \cdot \xi|^2 + \ell' \cdot \nabla_{\text{quadratic}}\right]$

첫 번째 extra term이 **"$\nabla_x f$의 $\xi$ 방향 크기 제곱에 비례하는 penalty"** — Jacobian-based regularization. $\square$

### Rademacher complexity 감소 — 단순 경우

$G = \{e, g\}$ (이진 group, e.g. horizontal flip). $\mathcal{F}_G = \{f : f(g \cdot x) = f(x)\}$.

$\mathcal{F}$의 Rademacher: $\mathcal{R}_n(\mathcal{F}) = \mathbb{E}[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum \epsilon_i f(x_i)]$.

$\mathcal{F}_G$의 경우 $f(x_i) = f(g x_i)$ 제약으로 $\sup$가 제한됨. 구체적 bound: $\mathcal{F}$의 각 $f$에 대해 $f^+ = (f + f \circ g)/2$ (group averaging)는 invariant. 이 averaging이 complexity를 최대 **$1/|G|$** 로 축소.

### Augmentation이 "noise 주입"과 다른 이유

Gaussian noise augmentation은 group-structured가 아닌 **isotropic noise**. 이는:
- VRM 관점: $\mathcal{N}(x, \sigma^2 I) \otimes \delta_y$로 유효.
- Invariance 관점: $G$가 없음 (no structure).
- Dao의 Jacobian penalty: $\|\nabla_x f\|^2$ (모든 방향).

Group augmentation은 **특정 방향** (group generator)만 penalize → task-informed smoothing. 더 효과적.

---

## 💻 실험으로 효과 검증

### 실험 1 — Rotation augmentation vs equivariant conv

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomRotation

# CIFAR-10 small experiment: rotation augmentation vs "rotated test" without augmentation
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
    def forward(self, x): return self.conv(x)

# (a) 표준 훈련 (no augmentation)
# (b) Rotation augmentation (±30°)
# Test accuracy:
#   - (a) normal test: 높음
#   - (a) rotated test (0°, 15°, 30°): 급락 → 모델이 rotation-invariant하지 않음
#   - (b) normal test: 비슷
#   - (b) rotated test: 안정 → invariance 주입 확인
```

### 실험 2 — Jacobian penalty 구현 (Dao style)

```python
class JacobianRegularizedLoss(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam
    def forward(self, model, x, y):
        base_loss = F.cross_entropy(model(x), y)
        # Compute Jacobian norm via autograd
        x.requires_grad_(True)
        logits = model(x)
        grad_outputs = torch.ones_like(logits)
        grads = torch.autograd.grad(logits, x, grad_outputs=grad_outputs,
                                     create_graph=True)[0]
        jac_penalty = grads.pow(2).sum()
        return base_loss + self.lam * jac_penalty
```

### 실험 3 — Augmentation으로 학습된 feature의 invariance 측정

```python
# Feature extractor f에 대해 pre(x), f(rot(x))의 cosine similarity 측정
# - Augmentation 없이 훈련: similarity 낮음 (non-invariant).
# - Rotation augmentation: similarity 높음 (approx-invariant).

def measure_invariance(model, x_test, num_angles=12):
    model.eval()
    with torch.no_grad():
        f_orig = model.backbone(x_test)
        similarities = []
        for angle in np.linspace(0, 360, num_angles, endpoint=False):
            x_rot = rotate(x_test, angle)
            f_rot = model.backbone(x_rot)
            sim = F.cosine_similarity(f_orig.flatten(1), f_rot.flatten(1)).mean()
            similarities.append(sim.item())
    return similarities

# Without augmentation: similarities drop sharply with angle
# With rotation augmentation: similarities stay near 1.0
```

### 실험 4 — Group Convolution vs Rotation augmentation

```python
# Cohen & Welling 2016 "Group Equivariant CNN"의 core idea
# rotation equivariant conv: filter를 여러 각도로 복사하고 응답 stack

# 실전 결과:
#   - Equivariant arch: rotation-shift robust, parameter 적음, 훈련 빠름
#   - Rotation augmentation: 간단 구현, architecture 자유, 대신 훈련 더 필요

# 작은 dataset (e.g. medical imaging): equivariant가 유리
# 큰 dataset (ImageNet): augmentation + 표준 CNN이 경쟁력
```

---

## 🔗 실전 활용

### Vision에서의 표준 augmentation

**CIFAR-10/100**:
- Random crop (32×32 → 32×32 with padding 4)
- Horizontal flip
- Color jitter
- Random erasing

**ImageNet**:
- Random resize crop (224×224)
- Horizontal flip
- ColorJitter, AutoAugment or RandAugment (Ch4-04)
- Mixup / CutMix (Ch4-03, Ch4-04)

### Task별 invariance 적절성

| Task | 적절 | 부적절 |
|------|------|-------|
| Natural image 분류 | Rotation (small), flip, crop | Large rotation (180°), vertical flip (for upright images) |
| Text | Random word drop, synonym | Rotation, flip |
| Audio | Pitch shift (small), time shift | Reverse, large pitch shift |
| Molecular (SMILES) | Atom ordering 변경 | Random swap |
| Medical imaging | 보수적 (rotation ≤15°, flip) | Aggressive color jitter |

### Architectural invariance vs Augmentation

**Architectural (Group equivariant CNN 등)**:
- **장점**: Exact invariance, parameter 효율, small data.
- **단점**: Architecture 복잡, group 정의 필요.

**Augmentation**:
- **장점**: 구현 쉬움, 어떤 model에도 적용.
- **단점**: Approx invariance, 추가 훈련 cost.

**실전 혼용**: Translation equivariance는 CNN architecture로 이미 내장, rotation은 augmentation으로 추가. Best of both worlds.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Group $G$가 task-relevant | 잘못 선택하면 useful feature 파괴 |
| Augmentation이 label-preserving | 많은 augmentation은 approximate only |
| First-order expansion (Dao 2019) | 큰 perturbation에서는 정확 X |
| Finite-group Rademacher bound | Continuous group (Lie)에서 더 미묘 |
| Data distribution이 $G$-symmetric | 실제로는 bias 있음 (e.g. upright 얼굴이 대부분) |

**주의**: **Augmentation이 invariance를 완전히 학습시키지 않음**. "Approximate invariance" — 훈련 중 보지 못한 $g$에 대해서는 generalize 안 될 수도.

---

## 📌 핵심 정리

$$\boxed{\text{Augmentation = VRM with group orbit vicinity → } G\text{-invariant feature learning}}$$

| 개념 | 의미 |
|------|------|
| **Group action** | $g \cdot x$ semantic preserving |
| **Augmentation** | $\mu_G$-distributed orbit sampling |
| **Dao 2019** | Aug = Jacobian-norm regularization (first-order) |
| **Chen 2020** | Aug ≈ invariant subspace projection |
| **Rademacher** | $\mathcal{R}_n(\mathcal{F}_G) \leq (1/|G|)\mathcal{R}_n(\mathcal{F})$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Horizontal flip augmentation은 어떤 group $G$의 action인가? 이 $G$의 order는?

<details>
<summary>힌트 및 해설</summary>

$G = \{e, \text{flip}\} = \mathbb{Z}_2$ (order 2). $|G| = 2$.

Rademacher reduction factor ≥ $1/|G| = 1/2$. 즉 flip augmentation만으로도 effective function class의 complexity를 **최대 절반**으로 축소.

</details>

**문제 2** (심화): Rotation augmentation의 generator는 무엇인가? Lie algebra 표현으로 infinitesimal rotation을 구하라.

<details>
<summary>힌트 및 해설</summary>

$SO(2)$ (2D rotation group)의 generator는 **skew-symmetric matrix**:

$$J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

Infinitesimal rotation: $g_\epsilon = e^{\epsilon J} = I + \epsilon J + O(\epsilon^2)$.

Image $x$ (2D grid)에 대한 action: pixel $(i, j)$를 $J \cdot (i, j)^T = (-j, i)^T$ 방향으로 이동. 이를 bilinear interpolation으로 resample.

Dao 정리에서 $\xi \cdot x$가 이 infinitesimal flow. Jacobian penalty $\|\nabla_x f \cdot (\xi \cdot x)\|^2$는 **rotation 방향의 feature gradient**를 penalize.

실전 구현: continuous rotation은 texture-destroying interpolation 필요. Discrete rotation ($90°$ multiples)이 정확한 group action.

</details>

**문제 3** (이론-실전): ImageNet의 "upright bias" (대부분 이미지가 "똑바로 서 있음")는 rotation augmentation의 사용을 어떻게 제한하는가? Rotation ±180°가 도움될까 harm될까?

<details>
<summary>힌트 및 해설</summary>

ImageNet은 대부분 natural image → **upright orientation bias**. Test set도 주로 upright.

**$±10-15°$ small rotation**:
- Real distribution variation (사진을 약간 기울여 찍은 경우).
- Data augmentation으로 일관성 있게 효과적.

**$±90°$ or $±180°$ large rotation**:
- Distribution mismatch — train에는 "뒤집힌 사과"가 없지만 test도 없음.
- **Feature 파괴**: gravity-dependent object (예: "upright bottle")는 180° 회전하면 다른 의미.
- **Harmful**: test accuracy 감소 가능.

결론: ImageNet 표준 recipe는 **작은 rotation만**. 단 어떤 domain (medical imaging, satellite, microscopy)에서는 $90°$ / $180°$ / reflection이 natural — domain-specific.

**이는 VRM의 "vicinity가 $P$에 가까워야 한다"** 교훈의 구체적 예 (Ch4-01).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Vicinal Risk Minimization](./01-vicinal-risk.md) | [📚 README로 돌아가기](../README.md) | [03. Mixup ▶](./03-mixup.md) |

</div>
