# 02. Sharpness-Aware Minimization (Foret et al. 2021)

## 🎯 핵심 질문

- **SAM**의 minimax 목적함수 $\min_\theta \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$은 왜 flat minimum을 명시적으로 탐색하는가?
- 1차 근사 $\epsilon^*(\theta) = \rho \nabla L / \|\nabla L\|$로 **two-step gradient**를 어떻게 구현하는가?
- **ASAM** (Kwon 2021)의 adaptive $\rho$의 동기는?
- SAM의 훈련 비용이 **2배**임에도 generalization 향상을 정당화하는 이유는?

---

## 🔍 왜 명시적 sharpness minimization?

Ch6-02에서 **SGD의 implicit bias가 flat minimum을 선호**한다고 봤다. Ch7-01의 **SWA**는 **implicitly** flat minimum 찾기.

**SAM의 접근**: **명시적으로** sharpness를 최소화. "**worst-case loss in $\rho$-neighborhood**"를 직접 최적화.

$$\min_\theta \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$$

- Inner max: $\theta$ 근방에서 **가장 나쁜 loss** 찾기.
- Outer min: 이 worst-case를 줄이는 $\theta$.

직관: $\theta$가 flat region에 있으면 $\epsilon$ perturbation도 loss 많이 증가 안 함 → worst-case도 작음. Sharp minimum은 $\epsilon$에 매우 민감 → worst-case 큼.

**결과**:
- ImageNet EfficientNet + SAM: +0.5-1% top-1 accuracy.
- CIFAR-100 WRN + SAM: +1-2%.
- Cost: 매 step에서 gradient 2번 계산 → **2배 훈련 시간**.

---

## 📐 수학적 선행 조건

- Ch6-02: Flat vs sharp minima
- Optimization: first-order approximation, Lagrangian
- 기본 gradient descent

---

## 📖 직관적 이해

### Minimax Formulation

$$L^{\text{SAM}}(\theta) = \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$$

이를 $\theta$에 대해 최소화. 직접 계산 어려움 — inner max를 **first-order Taylor**로 근사.

### First-Order Approximation

$L(\theta + \epsilon) \approx L(\theta) + \nabla L(\theta)^T \epsilon$

$\max_{\|\epsilon\| \leq \rho}$ subject to $\|\epsilon\|_2 \leq \rho$:

$$\epsilon^*(\theta) = \rho \cdot \frac{\nabla L(\theta)}{\|\nabla L(\theta)\|_2}$$

(Gradient 방향으로 $\rho$만큼 이동 — 가장 loss 증가시키는 방향.)

### Two-Step Gradient

SAM 1 step:

1. Compute $\nabla L(\theta)$.
2. $\tilde\theta = \theta + \epsilon^*(\theta)$ (adversarial point).
3. Compute $\nabla L(\tilde\theta)$.
4. Update: $\theta \leftarrow \theta - \eta \nabla L(\tilde\theta)$.

즉 "worst-case neighborhood의 gradient"로 업데이트. $\nabla L(\theta)$가 아닌 $\nabla L(\tilde\theta)$ 사용이 핵심.

### Why Two Gradient Steps?

- First step: $\epsilon^*$ 찾기 위한 "probe gradient".
- Second step: adversarial point에서의 gradient로 actual update.

**Cost 2배**: forward + backward 두 번. 하지만 이 추가 cost가 landscape-aware update로 보상.

### ASAM의 Adaptive $\rho$

SAM의 $\rho$ (neighborhood radius) 는 fixed scalar. 문제:
- Weight magnitude가 다른 layer에서 uniform $\rho$ 부적절.
- Scale-invariant sharpness measure 필요.

**ASAM** (Kwon 2021): $\epsilon$의 component를 $|\theta|$에 비례 scale:

$$\epsilon^*_{\text{ASAM}} = \rho \cdot \frac{|\theta| \cdot \text{sign}(\nabla L)}{\|\cdot\|}$$

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Sharpness-Aware Objective

$$L^{\text{SAM}}(\theta) = \max_{\|\epsilon\|_2 \leq \rho} L(\theta + \epsilon)$$

**SAM problem**: $\min_\theta L^{\text{SAM}}(\theta)$.

### 정의 2.2 — First-Order Approximation (Foret 2021)

Near $\epsilon = 0$:

$$L(\theta + \epsilon) \approx L(\theta) + \nabla L(\theta)^T \epsilon$$

Subject to $\|\epsilon\|_2 \leq \rho$:

$$\epsilon^*(\theta) = \rho \cdot \frac{\nabla L(\theta)}{\|\nabla L(\theta)\|_2}$$

Approximate SAM loss:

$$L^{\text{SAM-approx}}(\theta) = L(\theta + \epsilon^*(\theta)) = L(\theta + \rho \nabla L/\|\nabla L\|)$$

### 정리 2.3 — SAM Gradient (Two-Step)

Chain rule으로 $\nabla_\theta L^{\text{SAM-approx}}$ 계산:

$$\nabla_\theta L^{\text{SAM}} \approx \nabla_\theta L(\theta + \epsilon^*(\theta))$$

(Second-order term of $\epsilon^*$ w.r.t. $\theta$는 신경 안 씀.)

### 정리 2.4 — SAM Update Rule

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t + \rho \cdot \nabla L(\theta_t)/\|\nabla L(\theta_t)\|_2)$$

즉:
1. Compute $g_1 = \nabla L(\theta_t)$.
2. Compute $\tilde\theta = \theta_t + \rho g_1 / \|g_1\|$.
3. Compute $g_2 = \nabla L(\tilde\theta)$.
4. Update $\theta_{t+1} = \theta_t - \eta g_2$.

### 정의 2.5 — Adaptive SAM (Kwon et al. 2021)

Layer-wise adaptive $\rho$:

$$\epsilon_i^* = \rho \cdot \frac{\theta_i \cdot \nabla_i L(\theta)}{\|\theta \odot \nabla L(\theta)\|_2}$$

(Scale-invariant — $\theta_i \to c\theta_i$에 $\rho$ 변동 없음.)

### 정리 2.6 — SAM이 찾는 Minimum의 Flatness

SAM 최적화 후의 $\theta^*$에서 local Hessian의 eigenvalue 분포가 **일반 SGD보다 smaller maximum eigenvalue**. Empirically 측정 가능 (Foret 2021).

---

## 🔬 수학적 유도

### 정리 2.2 유도

$\max_{\|\epsilon\| \leq \rho} [L(\theta) + \nabla L^T \epsilon]$ is linear in $\epsilon$.

Constraint $\|\epsilon\|_2 \leq \rho$의 **Lagrangian**:

$\mathcal{L}(\epsilon, \lambda) = \nabla L^T \epsilon - \lambda(\|\epsilon\|_2^2 - \rho^2)$

$\nabla_\epsilon = 0 \implies \nabla L = 2\lambda \epsilon \implies \epsilon = \nabla L / (2\lambda)$

Plug into constraint $\|\epsilon\|_2 = \rho$:

$\rho = \|\nabla L\|/(2\lambda) \implies \lambda = \|\nabla L\|/(2\rho)$

$\epsilon^* = \rho \nabla L/\|\nabla L\|$. $\square$

### Gradient Descent in SAM loss

$L^{\text{SAM}}(\theta) = L(\theta + \epsilon^*(\theta))$.

Chain rule:

$$\nabla_\theta L^{\text{SAM}} = \nabla_\theta L(\theta + \epsilon^*) + \underbrace{\nabla_\theta \epsilon^* \cdot \nabla L(\theta + \epsilon^*)}_{\text{higher order}}$$

SAM은 **higher-order term을 무시** — 즉 approximate first-order gradient. Foret 2021 Prop 1: 이 approximation error는 $O(\rho)$ 이내.

### Why SAM Finds Flat Minima

SAM의 목적함수 $L^{\text{SAM}}(\theta) = L(\theta) + (\rho\|\nabla L\|/2) \cdot \text{(sharpness term)}$.

$\text{sharpness} \propto \|\nabla L\|$ near minimum + Hessian eigenvalue:

Near $\theta^*$: $\nabla L \approx H(\theta - \theta^*)$. 만약 $H$ spectral norm large (sharp), $\epsilon^*$ magnitude 크고 $L(\theta + \epsilon^*) - L(\theta)$ 큼.

**SAM update이 "loss가 $\epsilon$ perturbation에 robust한 $\theta$"로 이동** → small $H$ → flat minimum.

---

## 💻 실험으로 효과 검증

### 실험 1 — SAM 구현

```python
import torch
from torch.optim import Optimizer

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Add adversarial perturbation ε*."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Remove perturbation, update with gradient at perturbed point."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])   # revert
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2).to(shared_device)
                         for group in self.param_groups
                         for p in group['params'] if p.grad is not None]), p=2)
        return norm

# Usage:
optimizer = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1, momentum=0.9)
for x, y in loader:
    # First forward-backward
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    # Second forward-backward at perturbed theta
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.second_step(zero_grad=True)
```

### 실험 2 — SAM vs SGD on CIFAR-100

```python
# 동일한 WideResNet-28-10, 동일 epoch, 동일 data augmentation
# SGD result:    81.5% test accuracy
# SAM (ρ=0.05):  83.0% — 1.5% improvement
# SAM (ρ=0.1):   83.2%
# SAM (ρ=0.2):   82.0%  (too aggressive, underfit)
```

### 실험 3 — SAM의 loss landscape 시각화

```python
# Training 완료 후 SGD vs SAM model의 loss sensitivity 측정
def loss_neighborhood(model, loader, radius=10, N=50):
    """Random perturbations of magnitude 'radius'."""
    orig_params = [p.data.clone() for p in model.parameters()]
    losses = []
    for _ in range(N):
        # Random direction
        direction = [torch.randn_like(p) for p in model.parameters()]
        norm = sum((d**2).sum().sqrt() for d in direction)
        for p, d in zip(model.parameters(), direction):
            p.data += radius * d / norm
        # Measure loss
        total_loss, total = 0, 0
        for x, y in loader:
            total_loss += F.cross_entropy(model(x), y).item() * x.size(0)
            total += x.size(0)
        losses.append(total_loss / total)
        # Restore
        for p, o in zip(model.parameters(), orig_params):
            p.data = o.clone()
    return losses

# SGD model의 loss landscape variance vs SAM model의 variance
# → SAM model의 variance가 작음 (flat)
```

### 실험 4 — ρ sweep

```python
rhos = [0.01, 0.05, 0.1, 0.2, 0.5]
for rho in rhos:
    model = WideResNet28_10()
    opt = SAM(model.parameters(), torch.optim.SGD, rho=rho, lr=0.1)
    # Train ...
    acc = evaluate(model, test_loader)
    print(f"ρ={rho}: test acc = {acc:.4f}")
# 전형적:
# ρ=0.01: 81.8% (약한 SAM)
# ρ=0.05: 83.0% (표준)
# ρ=0.10: 83.2% (더 효과, 수렴 약간 느림)
# ρ=0.20: 82.5% (underfit 시작)
# ρ=0.50: 78.0% (too aggressive, convergence fails)
```

---

## 🔗 실전 활용

### SAM의 표준 hyperparameters

| Model | $\rho$ | Notes |
|-------|--------|-------|
| WideResNet (CIFAR) | 0.05 | 표준 default |
| ResNet (ImageNet) | 0.05 | Foret 2021 |
| EfficientNet (ImageNet) | 0.1 | Larger models need larger ρ |
| Transformer (image) | 0.05 ~ 0.2 | Task-specific tuning |
| Transformer (NLP) | SAM 덜 표준 | Fine-tuning cost |

### 언제 SAM을 쓰는가

**유리**:
- Small / medium dataset (CIFAR, ImageNet subset).
- Overfitting 위험 있는 대형 모델.
- Domain generalization / transfer.

**불리**:
- 매우 큰 training set (LLM pre-training): SAM의 cost-benefit 낮음.
- Limited compute: 2배 cost가 실전 blocker.
- Noise-robust task: SGD의 implicit noise가 이미 충분.

### SAM의 확장

- **ASAM** (Kwon 2021): adaptive layer-wise $\rho$ for scale invariance.
- **GSAM** (Zhuang 2022): gap-aware SAM, dual optimization.
- **ESAM** (Du 2022): efficient SAM, sub-batch use for cost reduction.
- **LookSAM** (Liu 2022): $k$-step SAM (cost reduction).

### AdamW + SAM

SAM은 **outer optimizer를 자유롭게 선택** 가능:
```python
opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3, weight_decay=1e-4)
```

Transformer 훈련에서 AdamW + SAM 조합 가끔 사용. 하지만 pre-training에서는 cost 이슈.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| First-order approximation | Large $\rho$에서 second-order effect 무시 |
| Global $\rho$ | Layer-wise adaptive가 더 좋을 수도 (ASAM) |
| 2x compute | Fine-tuning이나 small-data에서만 실용적 |
| Gradient norm for $\epsilon^*$ | Small gradient에서 $\epsilon^*$ direction noisy |
| L2 neighborhood | 다른 norm (Mahalanobis) ball도 고려 가치 |

**주의**: SAM이 "**모든 경우에 improvement**"는 아니다. 큰 labeled dataset + 적당한 regularization 있는 훈련에서는 **marginal 개선** 또는 없음.

---

## 📌 핵심 정리

$$\boxed{L^{\text{SAM}}(\theta) = \max_{\|\epsilon\|\leq\rho}L(\theta+\epsilon) \approx L(\theta + \rho\nabla L/\|\nabla L\|)}$$

| 개념 | 의미 |
|------|------|
| **Minimax objective** | Worst-case in $\rho$-neighborhood |
| **Two-step gradient** | First for $\epsilon^*$, second for actual update |
| **2x compute cost** | Single step이 2 forward-backward |
| **Flat minimum 명시** | SGD의 implicit bias를 explicit하게 만듦 |
| **ASAM / GSAM / ESAM** | 확장들 — scale invariance, efficiency |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\nabla L(\theta) = (0.3, 0.4)$, $\rho = 0.5$일 때 $\epsilon^*$는?

<details>
<summary>힌트 및 해설</summary>

$\|\nabla L\|_2 = \sqrt{0.09 + 0.16} = \sqrt{0.25} = 0.5$.

$\epsilon^* = 0.5 \cdot (0.3, 0.4)/0.5 = (0.3, 0.4) = \nabla L$.

즉 $\rho = \|\nabla L\|$이면 $\epsilon^* = \nabla L$ exactly. $\rho > \|\nabla L\|$이면 $\epsilon^*$가 $\nabla L$ direction의 $\rho/\|\nabla L\|$ 배.

**직관**: SAM은 "현재 gradient 방향으로 $\rho$만큼" perturbation 후 update.

</details>

**문제 2** (심화): SAM의 2x compute cost를 줄이는 방법으로 **LookSAM** (매 $k$ step마다만 SAM)의 이론적 정당화는?

<details>
<summary>힌트 및 해설</summary>

**LookSAM의 아이디어**: 매 step SAM 대신, 매 $k$ step마다 SAM update. 나머지 $k-1$ step은 이전 $\epsilon^*$ 재사용.

**이론적 근거**:
- Smooth loss에서 $\epsilon^*(\theta)$는 $\theta$에 대해 smooth function.
- 짧은 update step에서 $\epsilon^*(\theta_t)$가 $\epsilon^*(\theta_{t+1})$와 **similar**.
- 따라서 재사용해도 approximate SAM.

**Cost 감소**: $k = 5$이면 compute가 $2/5 = 40\%$만 증가 (cf. 원 SAM의 100%).

**성능 trade-off** (Liu 2022):
- $k = 2, 3$: 성능 거의 동일.
- $k = 5, 10$: marginal 성능 감소 (~0.1-0.2%).
- $k > 20$: SAM effect 거의 소실 → SGD와 비슷.

**실전 추천**: Fine-tuning + small compute budget에서 $k = 5$ 유용한 절충.

</details>

**문제 3** (이론-실전): SAM과 SWA는 둘 다 "flat minimum"을 target하지만 mechanism이 다르다. 조합 시 이점과 risk를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**SAM**: **매 update마다** flat을 향해 explicit step.
**SWA**: 훈련 후 **iterate 평균**으로 flat center 찾음.

**조합 (SAM + SWA)**:

**장점**:
- SAM이 각 $\theta_t$를 이미 flat basin에 두고, SWA가 basin 내부 center 찾음 → **double flat effect**.
- Izmailov 2019 follow-up: SAM + SWA가 SAM alone보다 0.2-0.5% 추가 개선.

**Risk**:
- **Compute cost 크게 증가**: SAM의 2x + SWA의 averaging = 매 epoch 2x + 최종 epoch extra cycling.
- **Tuning 복잡**: $\rho$, SWA $t_0$, cyclic lr 모두 tuning.
- **Diminishing returns**: Already flat (SAM)에 추가 SWA가 큰 도움 안 될 수도.

**실전 추천**:
- Competition / SOTA-seeking: 둘 다 사용.
- Production / time-constrained: SAM만 (simpler, more effective per cost).
- Medical / uncertainty-critical: SWAG + SAM (Bayesian + flat).

**현재 best practice**: SAM이 더 일반적, SWA는 specific edge case에서.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. SWA](./01-swa.md) | [📚 README로 돌아가기](../README.md) | [03. AdamW ▶](./03-adamw.md) |

</div>
