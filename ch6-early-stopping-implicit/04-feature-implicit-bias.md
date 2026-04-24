# 04. Feature-wise Implicit Bias와 Homogeneous Networks

## 🎯 핵심 질문

- **Homogeneous network** (ReLU, positive homogeneity)의 정의와 특성은?
- Lyu-Li 2019: GD가 **margin-maximizing stationary point**로 수렴하는 조건은?
- Neyshabur 2015의 **path-norm**과 homogeneity가 어떤 관계인가?
- Layer-wise normalization 없이도 NN이 암묵적으로 scale-matching 하는 이유는?

---

## 🔍 왜 homogeneous network bias가 중요한가

Ch6-02는 linear separable logistic에서 GD가 max-margin으로 수렴 (Soudry 2018)을 보였다. 이 결과를 **deep homogeneous network**로 확장한 것이 **Lyu & Li 2019**, **Ji & Telgarsky 2019**.

핵심 insight:
- ReLU NN + linear output layer는 **positive homogeneous** — $f(\alpha x; \theta) = \alpha f(x; \theta)$ (activation에 대한 scale).
- Homogeneity가 implicit bias를 **훨씬 풍부한 function class**로 확장.
- GD가 **"첫 번째 KKT point of margin-maximizing problem"**으로 수렴.

이는 "**over-parameterized NN이 어떤 feature를 배우는가**"라는 질문에 답하는 한 각도.

Ch6의 마지막 문서로서:
- Ch6-01: Early stopping = L2 (spectral filter).
- Ch6-02: SGD direction → max margin (linear).
- Ch6-03: Ridgeless = min-norm (over-parameterized).
- **Ch6-04: Deep network에서의 margin maximization**.

---

## 📐 수학적 선행 조건

- Ch6-02: Soudry 2018 max-margin
- ReLU network의 구조 (Neural Network Theory Deep Dive)
- KKT conditions, optimality
- Path-norm 개념 (Neyshabur 2015)

---

## 📖 직관적 이해

### Homogeneous Network

**Positive $L$-homogeneous function** $f$: $f(\alpha \theta) = \alpha^L f(\theta)$ for $\alpha > 0$.

**ReLU NN with linear output**: $f(x; \theta)$가 $\theta$에 대해 **$L$-homogeneous** ($L$ = depth, if weight biases 없음).

**Why**: ReLU$(\alpha z) = \alpha \text{ReLU}(z)$ for $\alpha > 0$. 네트워크의 모든 weight를 $\alpha$배 하면 output도 $\alpha^L$배.

### Consequence: Scale Freedom

$\theta$와 $\alpha\theta$가 같은 prediction (up to scale). 따라서 **훈련 loss가 scale-invariant** — 모델이 **$\theta$ direction만 학습**, magnitude는 logistic loss 때문에 $\|\theta\| \to \infty$.

같은 구조가 Soudry 2018 (linear의 $L = 1$ homogeneity)의 **general version**.

### Lyu-Li 2019의 결과

Separable data + logistic loss + ReLU NN에서:

$$\frac{\theta_t}{\|\theta_t\|} \to \text{KKT point of } \min_\theta \tfrac{1}{2}\|\theta\|^2 \text{ s.t. margin} \geq 1$$

즉 GD가 "**NN-space의 max-margin**"으로 수렴. 선형 SVM의 NN generalization.

### Margin의 재정의

Deep NN의 "margin"이란? Output $y_i f(x_i; \theta) \geq 1$ (logistic loss가 0에 가까워지는 region). "Signed distance from decision boundary"의 nonlinear version.

### Neyshabur Path-Norm

Alternative capacity measure (Neyshabur 2015):

$$\|f\|_{\text{path}}^2 = \sum_{\text{paths in NN}} \prod_{\text{edges on path}} w_e^2$$

ReLU NN이 positive homogeneous이기 때문에 path-norm이 **scale-invariant** — reparameterization $(W_l, W_{l+1}) \to (cW_l, W_{l+1}/c)$에 불변.

**Claim**: Path-norm이 NN generalization의 **더 좋은 capacity measure**. Norm-based PAC-Bayes bounds의 기초.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Positive Homogeneous Function

$f: \mathbb{R}^d \to \mathbb{R}^k$ is **$L$-homogeneous** if $f(\alpha x) = \alpha^L f(x)$ for all $\alpha > 0$.

### 정의 4.2 — Homogeneous Neural Network

ReLU NN $f(x; \theta)$ with bias terms = 0 is $L$-homogeneous in $\theta$:

$$f(x; c\theta) = c^L f(x; \theta) \quad \forall c > 0$$

$L$ = depth (number of weight layers).

### 정리 4.3 — Scale Equivalence

Logistic loss $L(\theta) = \sum \log(1 + e^{-y_i f(x_i; \theta)})$에서, $\theta$와 $c\theta$의 prediction이 **same direction** ($\text{sign}$):

$$y_i f(x_i; c\theta) = c^L y_i f(x_i; \theta)$$

Loss value $\neq$ same, but margin proportional: $y_i f/\|\theta\|^L$는 $c$-invariant.

### 정의 4.4 — Normalized Margin

$$\bar{\gamma}_i(\theta) := \frac{y_i f(x_i; \theta)}{\|\theta\|_2^L}$$

Scale-invariant margin of point $i$. Worst-case margin:

$$\bar{\gamma}(\theta) = \min_i \bar\gamma_i(\theta)$$

### 정리 4.5 — Lyu-Li 2019 (주 정리)

Separable data, logistic loss, homogeneous ReLU NN, 충분한 훈련 시간에서:

$$\frac{\theta_t}{\|\theta_t\|} \to \arg\min_{\theta: \|\theta\| = 1} \|\theta\| \text{ s.t. } \min_i y_i f(x_i; \theta) \geq 1$$

Specifically, 이는 **1st-order KKT point**. 전역 max-margin 보장은 없음 (NN이 non-convex이므로).

### 정의 4.6 — Path-Norm (Neyshabur et al. 2015)

$L$-layer ReLU NN의 path $p$ = $(e_1, e_2, \ldots, e_L)$ (각 layer에서 한 edge). Path weight: $\prod_{e \in p} w_e$.

$$\|\theta\|_{\text{path}} := \sqrt{\sum_{\text{paths } p} \left(\prod_{e \in p} w_e\right)^2}$$

### 정리 4.7 — Path-Norm의 Scale Invariance

Weight rescaling $(W_l, W_{l+1}) \to (cW_l, W_{l+1}/c)$: Path의 weight product 불변 → path-norm 불변.

반면 **$L_2$-norm** $\|\theta\|_2^2 = \sum w_e^2$는 이 rescaling에 **변화**.

**함의**: Path-norm이 "reparameterization-invariant" capacity measure.

---

## 🔬 수학적 유도

### Lyu-Li 2019 증명 Idea

**Step 1**: Homogeneous NN에서 GD trajectory의 asymptotic behavior — $\theta_t$가 $\log t$ scale로 증가.

**Step 2**: Direction $\hat{\theta}_t = \theta_t / \|\theta_t\|$의 limit 존재 (smooth loss + positive homogeneity 덕분).

**Step 3**: Limit $\hat{\theta}^*$가 margin-maximization KKT 조건 만족:
- Stationarity: $\nabla_\theta L(\theta^*) = \lambda \theta^*$ (proportionality with Lagrange multiplier).
- Primal feasibility: margin $\geq 1$.
- Complementary slackness: support vectors에만 constraint active.

**Key insight** (Nacson 2019, Ji-Telgarsky 2019): Logistic loss의 "**heavy tail**" (slow decay)가 **strict maximum** 선호 유도.

### Linear SVM과의 Connection

$L = 1$ (linear) special case: Lyu-Li의 결과 = Soudry 2018의 Linear SVM 수렴.

$L > 1$ (deep NN)에서는:
- Margin 정의가 nonlinear.
- 여러 KKT points 존재 가능 (non-convex).
- 전역 최적 보장 없음, 하지만 "lazy" KKT point로도 **good generalization** 경험적.

### Path-Norm vs L2-Norm

Homogeneous NN에서 $\|\theta\|_2$는 **not invariant** to $(W_l, W_{l+1}) \to (cW_l, W_{l+1}/c)$. Path-norm은 invariant.

**결과**:
- Bartlett-Foster-Telgarsky 2017의 **spectrally-normalized margin bound**는 $\prod \|W_l\|_\sigma / \gamma \sqrt{n}$ — 일부 scale-invariance.
- **Full invariance** 얻으려면 path-norm 사용 — Generalization Theory Deep Dive Ch2-03.

---

## 💻 실험으로 효과 검증

### 실험 1 — 2-layer ReLU NN에서 margin convergence

```python
import torch
import torch.nn as nn
import numpy as np

# Separable 2D data
torch.manual_seed(0)
X_a = torch.randn(30, 2) + torch.tensor([3., 0.])
X_b = torch.randn(30, 2) + torch.tensor([-3., 0.])
X = torch.cat([X_a, X_b])
y = torch.cat([torch.ones(30), -torch.ones(30)])

class ReLU_NN(nn.Module):
    def __init__(self, h=100):
        super().__init__()
        self.fc1 = nn.Linear(2, h, bias=False)   # no bias for homogeneity
        self.fc2 = nn.Linear(h, 1, bias=False)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)

net = ReLU_NN()
opt = torch.optim.SGD(net.parameters(), lr=0.01)

# Track margin over training
margins = []
for step in range(50000):
    opt.zero_grad()
    logits = net(X)
    loss = torch.log(1 + torch.exp(-y * logits)).mean()
    loss.backward(); opt.step()
    
    if step % 100 == 0:
        with torch.no_grad():
            theta_norm = sum((p**2).sum().sqrt() for p in net.parameters()).item()
            min_margin = (y * net(X)).min().item() / theta_norm**2
            margins.append(min_margin)

import matplotlib.pyplot as plt
plt.plot(margins)
plt.xlabel('training step'); plt.ylabel(r'normalized margin $\bar\gamma$')
plt.title('Lyu-Li 2019: margin → max-margin KKT')
plt.grid(alpha=0.3); plt.show()
# → margin이 증가해 일정 값에 수렴
```

### 실험 2 — Path-norm의 scale invariance 확인

```python
def path_norm(net):
    """For 2-layer NN."""
    W1 = net.fc1.weight    # (h, d)
    W2 = net.fc2.weight    # (1, h)
    # Path: input_i → hidden_j → output. Weight: W1[j, i] * W2[0, j]
    paths = (W2.squeeze() ** 2) * (W1 ** 2).sum(dim=1)
    return paths.sum().sqrt()

# 훈련된 net의 path-norm
pn_orig = path_norm(net).item()
l2_orig = sum((p**2).sum() for p in net.parameters()).sqrt().item()

# Rescale: W1 *= 10, W2 /= 10
with torch.no_grad():
    net.fc1.weight.mul_(10)
    net.fc2.weight.div_(10)

pn_rescaled = path_norm(net).item()
l2_rescaled = sum((p**2).sum() for p in net.parameters()).sqrt().item()

print(f"Before rescale: path-norm = {pn_orig:.4f}, L2 = {l2_orig:.4f}")
print(f"After  rescale: path-norm = {pn_rescaled:.4f}, L2 = {l2_rescaled:.4f}")
# → path-norm 불변, L2는 크게 변화
```

### 실험 3 — Homogeneity 확인

```python
# net 복사본
net_copy = ReLU_NN()
net_copy.load_state_dict(net.state_dict())

# 입력 $x$에서 prediction
x_test = torch.tensor([[1.5, 0.5]])
y1 = net_copy(x_test).item()

# 모든 weight를 2배로
with torch.no_grad():
    for p in net_copy.parameters():
        p.mul_(2)

y2 = net_copy(x_test).item()
print(f"Original prediction: {y1:.4f}")
print(f"After *2 on all weights: {y2:.4f}")
print(f"Ratio: {y2/y1:.4f} (expected 2^L = 2^2 = 4)")
# → y2/y1 ≈ 4 (2-layer NN → L = 2)
```

### 실험 4 — 학습된 NN의 normalized margin 비교

```python
# 다른 initialization, 다른 step count에서 훈련된 모델들의 normalized margin
margins_list = []
for seed in range(5):
    torch.manual_seed(seed)
    net = ReLU_NN()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(30000):
        opt.zero_grad()
        loss = torch.log(1 + torch.exp(-y * net(X))).mean()
        loss.backward(); opt.step()
    
    theta_norm_sq = sum((p**2).sum() for p in net.parameters()).item()
    with torch.no_grad():
        margin = (y * net(X)).min().item() / theta_norm_sq
    margins_list.append(margin)

print("Normalized margins from 5 random seeds:", margins_list)
# → 비슷한 값으로 수렴 (KKT point는 unique하지 않지만 비슷한 margin)
```

---

## 🔗 실전 활용

### NN Generalization 이해의 구성 요소

1. **Implicit regularization from GD**: Max-margin selection.
2. **Homogeneity**: Prediction이 weight direction에 의존.
3. **Over-parameterization + min-norm**: Feature selection bias.
4. **Flat minima**: SGD의 landscape exploration.

이 모두가 "**왜 big model이 generalize하는가**"의 다른 측면.

### Path-Norm의 실용적 사용

- **PAC-Bayes bound**: Path-norm 기반 bound가 spectral norm보다 **tighter** (Neyshabur 2017).
- **Pruning**: Path-norm 작은 edge를 먼저 prune → lottery ticket.
- **Implicit bias measurement**: 훈련된 NN의 path-norm이 test accuracy와 correlation.

### 경험적 관찰

- **Deep network**의 학습된 solution이 **near-max-margin** — Lyu-Li 이론 뒷받침.
- **Wide network**에서 margin이 더 명확 (NTK regime).
- **Feature learning**에서 margin은 정의 어려움 — 이론 아직 미완.

### 한계

Homogeneity 가정이 **실제 NN에서 깨짐**:
- Bias terms: $f(x; \theta + b)$는 homogeneous 아님.
- BatchNorm: batch statistics scaling effect.
- Nonlinear normalization: LayerNorm, RMSNorm.

**그럼에도 bias 없는 linear output layer만** 있어도 homogeneity 상당 부분 유지 → 실전 relevant.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| No biases in NN | 실제 NN은 bias 포함 (homogeneity partial) |
| Separable data | Non-separable에서는 margin undefined |
| Logistic / exponential loss | 다른 loss에서 결과 다름 |
| ReLU activation | Sigmoid, tanh 등에서 non-homogeneous |
| 전역 최적 아닌 KKT | Multiple KKT points, non-convex |
| 무한 훈련 시간 | Finite training에서 approximation |

**주의**: Lyu-Li 결과는 "**asymptotic**" — 실전 finite training에서 exact max-margin 달성은 어려움. 그러나 **direction bias**가 early training부터 나타남.

---

## 📌 핵심 정리

$$\boxed{\text{Homogeneous NN + GD} \to \text{KKT of } \min \|\theta\|^2 \text{ s.t. margin} \geq 1}$$

| 개념 | 의미 |
|------|------|
| **Homogeneity** | $f(c\theta) = c^L f(\theta)$ for ReLU NN |
| **Normalized margin** | $y_i f(x_i)/\|\theta\|^L$ — scale-invariant |
| **Lyu-Li 2019** | Deep NN version of Soudry 2018 |
| **Path-norm** | Scale-invariant capacity measure |
| **Ch6 마무리** | Implicit regularization 3대 요소 통합 (early stop, SGD bias, overparam, homogeneity) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-layer ReLU NN $f(x) = W_3 \text{ReLU}(W_2 \text{ReLU}(W_1 x))$에서 모든 weight를 5배로 증가시키면 output은 몇 배?

<details>
<summary>힌트 및 해설</summary>

$L = 3$ (3 weight layers, no bias). $f(5\theta) = 5^3 f(\theta) = 125 f(\theta)$.

각 ReLU layer가 1-homogeneous이고 composition으로 degree가 더해진다.

Linear output ($L = 1$)과 비교: 같은 $c = 5$이면 $5 f$만. Depth가 커질수록 homogeneity의 exponent 증가 → scale sensitivity 증가.

</details>

**문제 2** (심화): Bias term이 있는 NN에서 homogeneity는 어떻게 깨지는가? 실용적 함의는?

<details>
<summary>힌트 및 해설</summary>

$f(x) = W_2 \text{ReLU}(W_1 x + b_1)$.

$c\theta = (cW_1, cb_1, cW_2, cb_2)$에 대해:

$f(x; c\theta) = c W_2 \text{ReLU}(c W_1 x + c b_1) = c^2 W_2 \text{ReLU}(W_1 x + b_1)$ (if ReLU는 positive homogeneous).

따라서 $f(x; c\theta) = c^L f(x; \theta)$ **여전히** 성립 — bias가 weight처럼 scaled된다면.

**하지만**: Bias가 "모든 sample에 shift" 역할이라, 훈련 중 bias 업데이트 dynamics가 weight와 달라 **effective homogeneity 깨짐**.

**실용적 함의**:
1. Lyu-Li 이론이 **approximately** 성립.
2. Path-norm 정의 수정 필요 (bias path 별도 처리).
3. 실전 NN에서 "almost homogeneous" — 결론 robust.

**현대 Transformer**: Attention에 bias 없는 경우 많음, LayerNorm에만 bias. 어떤 layer는 homogeneous, 어떤 layer는 not — hybrid structure.

</details>

**문제 3** (이론-실전): Path-norm이 spectral norm보다 tighter PAC-Bayes bound를 주지만 실전에서 덜 쓰이는 이유는?

<details>
<summary>힌트 및 해설</summary>

**Path-norm의 이론적 장점**:
- Scale invariance → parameterization-agnostic.
- Bartlett-Telgarsky의 기하적 complexity에 더 가까움.
- 학습된 NN에서 spectral norm보다 tighter empirical bound.

**실전에서 덜 쓰이는 이유**:

1. **Computational cost**: Path-norm $\sum_{\text{paths}} \prod w_e^2$는 **exponentially many paths** ($\sim h^L$). 직접 계산 infeasible.
   - 근사 알고리즘이 필요 (sampling).

2. **Gradient 복잡**: Path-norm regularization을 training loss에 추가하려면 efficient backprop 구현 필요.

3. **Spectral norm의 simplicity**: 
   - 쉽게 계산 (power iteration).
   - PyTorch `torch.nn.utils.spectral_norm`으로 내장.
   - Generative model (SNGAN) 같은 응용.

4. **경험적 성능**: 실전 성능 면에서 path-norm-regularized가 spectral-norm-regularized와 크게 차이 안 남.

**현대 실용적 접근**:
- 명시적 path-norm regularization 대신 **implicit homogeneity** 활용.
- Weight decay + BN/LN 조합으로 "path-norm-like" behavior 간접 유도.
- 이론 side에서는 path-norm이 중요, 실전 training에서는 다른 도구.

이는 "이론적 최적 vs 실용적 simplicity"의 전형적 trade-off.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Ridgeless Regression](./03-ridgeless-regression.md) | [📚 README로 돌아가기](../README.md) | [Chapter 7 → 01. SWA ▶](../ch7-modern-synthesis/01-swa.md) |

</div>
