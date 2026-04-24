# 03. Weight Decay vs L2 in Adaptive Methods — AdamW

## 🎯 핵심 질문

- Adam에서 L2 regularization $\lambda \|w\|^2$을 loss에 더하면 왜 **원래 L2 의미를 잃는가**?
- **AdamW** (Loshchilov & Hutter 2019)의 **decoupled weight decay**는 무엇을 분리하는가?
- 수식 대조: Adam + L2 vs AdamW의 update rule 차이는?
- ImageNet에서 AdamW가 Adam + L2보다 왜 **1-2% 우수**한가?

---

## 🔍 왜 이 주제가 중요한가

Ch1-01에서 "L2 regularization = Gaussian prior MAP"로 시작했다. 실전에서는 **weight decay**로 구현 — PyTorch `weight_decay=1e-4`.

**그러나**: Adam 같은 adaptive optimizer에서 "**L2 추가**"와 "**weight decay**"는 **다른 것**이 됐다. Loshchilov & Hutter 2019의 발견:

- Adam + L2 loss: gradient가 $v_t$ (moment)로 divide되어 weight decay가 **parameter별로 왜곡**.
- AdamW: weight decay를 update 단계에서 **분리**하여 순수 L2 shrinkage 유지.

**결과**:
- Adam + L2: 경험적으로 **generalization 나쁨**.
- AdamW: Adam의 빠른 수렴 + 진짜 L2 regularization → **더 나은 generalization**.

이 발견 후 (2019+) **모든 주요 Transformer 훈련이 AdamW 사용** — GPT, BERT, Llama, Mistral 등.

---

## 📐 수학적 선행 조건

- Ch1-01: L2 = Gaussian prior MAP
- Adam optimizer 수식 (Kingma 2014)
- Momentum, adaptive learning rate

---

## 📖 직관적 이해

### Adam Recap

Adam update:

$$g_t = \nabla L(\theta_t), \quad m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Key**: Update step이 $\sqrt{v}$로 scaled — 각 parameter별로 **adaptive lr**. 자주 큰 gradient 받은 param은 update 작게, 드물게 큰 gradient는 update 크게.

### Adam + L2 의 문제

L2 loss 추가: $\tilde{L}(\theta) = L(\theta) + \tfrac{\lambda}{2}\|\theta\|^2$.

Gradient: $\tilde{g} = g + \lambda \theta$.

Adam update:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{\tilde{m}}_t}{\sqrt{\hat{\tilde{v}}_t} + \epsilon}$$

여기서 $\tilde{m}, \tilde{v}$는 $g + \lambda\theta$의 running moments.

**문제**: $\sqrt{v_t}$가 **parameter별로** 다름. $\theta$가 큰 방향은 $v$도 커서 $\sqrt{v}$로 divide하면 **weight decay의 effective strength가 약해짐**.

즉 "**큰 weight을 더 shrink해야 하는데 오히려 덜 shrink됨**" — L2 정신에 반함.

### AdamW의 해결

Weight decay를 **gradient 이전에 분리**:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

$\hat{m}, \hat{v}$는 **원래 loss gradient** $g$만 사용. Weight decay $\eta\lambda\theta_t$는 **직접 subtract**.

**효과**: Weight decay가 $\sqrt{v}$로 scaling되지 않음 → 모든 parameter에 **uniform L2 shrinkage**.

### 수식 차이 명확히

| Method | Update rule |
|--------|-------------|
| SGD + L2 | $\theta \leftarrow \theta - \eta (g + \lambda\theta) = \theta - \eta g - \eta\lambda\theta$ |
| Adam + L2 | $\theta \leftarrow \theta - \eta \frac{\hat{m}(g + \lambda\theta)}{\sqrt{\hat{v}(g + \lambda\theta)}}$ |
| **AdamW** | $\theta \leftarrow \theta - \eta \frac{\hat{m}(g)}{\sqrt{\hat{v}(g)}} - \eta\lambda\theta$ |

**SGD + L2와 AdamW의 parallel**: 둘 다 weight decay term이 learning rate로만 곱해짐. Adam + L2는 $\sqrt{v}$로 왜곡.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Adam with L2 Regularization

$$g_t = \nabla L(\theta_t) + \lambda \theta_t$$

Adam's $m_t, v_t$ computed from $g_t$ (L2 gradient included):

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 정의 3.2 — AdamW (Loshchilov & Hutter 2019)

$$g_t = \nabla L(\theta_t) \quad \text{(L2 제외)}$$

Adam's $m_t, v_t$ computed from pure gradient:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

Weight decay $-\eta\lambda\theta_t$ **update 단계에서 분리**.

### 정리 3.3 — Adam + L2의 Implicit Per-Parameter Weight Decay

Adam + L2 update의 effective weight decay per parameter:

$$\lambda_{\text{eff}, i} = \lambda \cdot \frac{\theta_i}{\sqrt{\hat{v}_{t, i}}}$$

즉 **parameter별로 다른 strength**. 큰 $\hat{v}$ (historically large gradient) parameter는 $\lambda_{\text{eff}}$ **작음** — L2의 의도 위반.

### 정리 3.4 — AdamW의 Uniform Weight Decay

AdamW에서 weight decay term은 모든 parameter에 **uniform $\lambda$**:

$$\theta_i \leftarrow (1 - \eta\lambda)\theta_i + \text{(gradient update)}$$

### 정리 3.5 — Loshchilov & Hutter 2019 Empirical Result

ImageNet ResNet + Adam 훈련:
- Adam + L2: 75.2% (weight_decay=1e-4).
- AdamW: **76.4%** (+1.2%).

CIFAR-10, Language modeling 등 다양한 task에서 consistent improvement.

### 정의 3.6 — Cosine Annealing + AdamW

전형적 현대 recipe:

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$$

AdamW + cosine schedule + warmup — Transformer 훈련 표준.

---

## 🔬 수학적 유도

### Adam + L2의 "distortion" 정량화

$\nabla(L + \tfrac{\lambda}{2}\|\theta\|^2) = g + \lambda\theta$.

$v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t + \lambda\theta_t)^2 \approx \beta_2 v_{t-1} + (1-\beta_2) g_t^2 + 2(1-\beta_2)\lambda \theta_t g_t$

Cross term $\lambda \theta_t g_t$: problem-dependent. Small $\lambda$에서는 negligible, large $\lambda$에서는 significant.

**Major effect**: $v_t$가 $\lambda^2 \theta_t^2$ term 포함 → $\sqrt{v_t}$가 $|\theta_t|$에 영향.

Update: $-\eta \hat{m}_t / \sqrt{\hat{v}_t}$. Weight decay component:

$-\eta \cdot \lambda\theta_t / \sqrt{\hat{v}_t} \neq -\eta\lambda\theta_t$ (AdamW)

**Diffference**: $\sqrt{\hat{v}_t}$ factor. Magnitude depends on gradient history.

### AdamW가 Pure L2와 Equivalent

$\theta_t$에 $(1 - \eta\lambda)$ 곱 = $\theta_t - \eta\lambda\theta_t$. 정확히 Gaussian prior MAP의 gradient update (Ch1-01 Gradient Descent form):

$-\eta \lambda\theta = -\eta \nabla(\tfrac{\lambda}{2}\|\theta\|^2)$

Adam의 adaptive gradient update는 $g$에만 작용 → separation of concerns.

### 왜 Adam + L2가 Underperforming

- Adam + L2: 큰 weight이 덜 shrink (by $1/\sqrt{v}$ factor) → effective regularization 약해짐 → overfitting.
- AdamW: Uniform shrinkage → 모든 weight에 동일한 regularization pressure.

특히 **Transformer의 extreme parameters** (embedding, attention weight)에서 이 차이 중요.

---

## 💻 실험으로 효과 검증

### 실험 1 — PyTorch AdamW 사용

```python
import torch

# Adam + L2 (incorrect way)
opt_bad = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# PyTorch의 Adam with weight_decay > 0은 "L2 loss 추가"로 구현 (Adam + L2 문제)

# AdamW (correct way)
opt_good = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Decoupled weight decay
```

### 실험 2 — Adam vs AdamW 수동 구현

```python
class AdamWithL2:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, lam=1e-4):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.lam = lr, beta1, beta2, eps, lam
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None: continue
            g = p.grad + self.lam * p.data    # L2 added to gradient!
            m.mul_(self.b1).add_(g, alpha=1-self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1-self.b2)
            m_hat = m / (1 - self.b1**self.t)
            v_hat = v / (1 - self.b2**self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

class AdamW_manual:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, lam=1e-4):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.lam = lr, beta1, beta2, eps, lam
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None: continue
            g = p.grad    # Pure gradient, no L2
            m.mul_(self.b1).add_(g, alpha=1-self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1-self.b2)
            m_hat = m / (1 - self.b1**self.t)
            v_hat = v / (1 - self.b2**self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
            p.data -= self.lr * self.lam * p.data    # Decoupled weight decay
```

### 실험 3 — ImageNet ResNet 재현 (Loshchilov 2019 style)

```python
# 동일 model, 동일 hyperparam 제외 optimizer:
# (a) Adam, weight_decay=1e-4 (Adam + L2)
# (b) AdamW, weight_decay=1e-4

# 전형적 결과:
# (a) Adam + L2:  75.0% top-1
# (b) AdamW:      76.3% top-1
# → ~1.3% 차이
```

### 실험 4 — Weight norm tracking

```python
import matplotlib.pyplot as plt

# 훈련 중 weight norm 변화 측정
def track_weight_norms(optimizer_name, epochs=50):
    model = YourModel()
    opt = AdamWithL2(...) if optimizer_name == 'Adam+L2' else AdamW_manual(...)
    norms_per_layer = {n: [] for n, _ in model.named_parameters()}
    
    for epoch in range(epochs):
        # train one epoch...
        for n, p in model.named_parameters():
            norms_per_layer[n].append(p.norm().item())
    return norms_per_layer

norms_adam = track_weight_norms('Adam+L2')
norms_adamw = track_weight_norms('AdamW')

# Plot: 각 layer의 norm trajectory
# → Adam+L2: 큰 weight은 상대적으로 덜 shrink (distorted)
# → AdamW: 모든 layer에서 uniform shrinkage rate
```

### 실험 5 — Hyperparameter transfer (Adam+L2 vs AdamW)

```python
# Adam+L2에서 최적 weight_decay는 task-dependent, hard to tune
# AdamW의 weight_decay는 SGD처럼 interpretable

# Recommended AdamW recipes:
# Transformer (small): weight_decay=0.1
# Transformer (large LLM): weight_decay=0.1
# CNN (ImageNet): weight_decay=0.05
# Vision Transformer: weight_decay=0.05

# Adam+L2에서는 보통 weight_decay=1e-4 (much smaller) — distortion 때문
```

---

## 🔗 실전 활용

### 현대 LLM 훈련 Recipe

**표준 setup**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # peak LR
    betas=(0.9, 0.95), # β2=0.95 for stability
    weight_decay=0.1,  # strong decoupled WD
    eps=1e-8
)

# Cosine schedule + warmup
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=2000, num_training_steps=total_steps
)
```

**거의 모든 SOTA LLM이 이 recipe 변형**:
- GPT-3, GPT-4
- Llama 1, 2, 3
- Mistral, Mixtral
- PaLM, Gemini

### AdamW의 확장

- **NAdamW**: Nesterov momentum + decoupled decay.
- **RAdam** (Liu 2020): Variance-rectified adaptive lr + decoupled decay.
- **Lion** (Chen 2023): Sign-based + decoupled decay. AdamW의 더 efficient 대체.
- **LAMB** (You 2019): Layer-wise adaptive + decoupled decay. Large-batch training.

이 모든 변형이 **decoupled weight decay를 핵심 digestive로** 채택.

### 실전 hyperparameter 가이드

| Model size | lr | weight_decay | betas |
|-----------|-----|--------------|-------|
| < 100M | 3e-4 ~ 6e-4 | 0.1 | (0.9, 0.98) |
| 100M - 1B | 3e-4 | 0.1 | (0.9, 0.95) |
| 1B - 10B | 1e-4 ~ 3e-4 | 0.1 | (0.9, 0.95) |
| > 10B | 6e-5 ~ 1.5e-4 | 0.1 | (0.9, 0.95) |

### Weight Decay를 제외할 Parameter

현대 recipe에서는 **다음 parameter에 wd 제외**:
- LayerNorm / RMSNorm의 $\gamma, \beta$.
- Bias terms.
- Embedding layer (논란, 일부는 wd 적용).

```python
decay_params, no_decay_params = [], []
for n, p in model.named_parameters():
    if 'norm' in n or 'bias' in n:
        no_decay_params.append(p)
    else:
        decay_params.append(p)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=3e-4)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Adaptive lr이 필요 | Pure SGD로 충분한 task에서는 효과 적음 |
| Weight decay가 Gaussian prior | Non-Gaussian prior (Laplace, etc.)는 decoupled 구현 달라짐 |
| 모든 param에 같은 wd | LayerNorm 등은 제외해야 — 기계적 적용 불가 |
| 수치 안정성 | Very small $\epsilon$ + small $v$에서 update 폭발 |
| AdamW의 generalization 우위 | 일부 task (e.g. Adam이 이미 잘 되는)에서는 marginal |

**주의**: AdamW도 **perfect 아님**. Lion (Chen 2023) 같은 더 나은 optimizer 연구 진행 중. Decoupled weight decay 원리는 **robust** 하지만 Adam 자체의 한계는 남음.

---

## 📌 핵심 정리

$$\boxed{\text{AdamW: } \theta \leftarrow \theta - \eta \frac{\hat m}{\sqrt{\hat v} + \epsilon} - \eta\lambda\theta \quad (\text{weight decay 분리})}$$

| 개념 | 의미 |
|------|------|
| **Adam + L2** | Gradient에 $\lambda\theta$ 추가 → $\sqrt{v}$로 왜곡 |
| **AdamW** | Weight decay를 update 단계에서 분리 |
| **차이** | Adam+L2는 per-param wd, AdamW는 uniform wd |
| **결과** | ImageNet +1.2%, Transformer에 표준 |
| **Modern LLM** | 거의 모두 AdamW 사용 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Adam의 $\hat{v}_t = 0.5$, $\theta_t = 2$, $\lambda = 1e-4$, $\eta = 1e-3$일 때 Adam + L2와 AdamW의 weight decay update는 각각 얼마?

<details>
<summary>힌트 및 해설</summary>

**Adam + L2**: Weight decay via gradient addition.
- $g_{\text{total}}$에 $\lambda\theta = 2 \cdot 10^{-4}$ 추가.
- Update scaled by $\sqrt{\hat{v}} = \sqrt{0.5} \approx 0.707$.
- Effective WD update: $-\eta \cdot \lambda\theta / \sqrt{\hat{v}} = -10^{-3} \cdot 2 \cdot 10^{-4} / 0.707 \approx -2.83 \cdot 10^{-7}$.

**AdamW**: Direct WD update.
- $-\eta\lambda\theta = -10^{-3} \cdot 10^{-4} \cdot 2 = -2 \cdot 10^{-7}$.

**비교**: Adam+L2가 약 1.4배 크게 shrink **이 parameter에 대해서만**. 다른 parameter ($\hat{v}$가 다른)는 또 다른 ratio.

**함의**: Adam+L2의 effective WD가 **parameter마다 다름** — 튜닝 어렵고 non-uniform regularization.

</details>

**문제 2** (심화): Loshchilov 2019 Table에서 Adam vs AdamW의 **lr과 wd의 optimal pair**는 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

**Adam + L2**: 
- Optimal lr과 wd가 **coupled** — lr을 변경하면 optimal wd도 변경.
- Grid search가 2D (lr × wd) 필요.
- 예: lr=1e-3 → wd=1e-4 optimal. lr=3e-4 → wd=3e-5 optimal (proportional shift).

**AdamW**:
- Optimal lr과 wd가 **decoupled** — 각각 independently tune 가능.
- Grid search가 효율적 (1D for each).
- 예: lr를 어떻게 바꿔도 wd=1e-2 혹은 0.1이 optimal (robust).

**Practical benefit**: AdamW는 hyperparameter search가 **훨씬 간단**. 이것이 large-scale experiments에서 중요한 이유.

**수식적 근거**: AdamW의 weight decay term $-\eta\lambda\theta$에서 $\eta$와 $\lambda$가 **multiplicatively 결합**하지만 meaning 관점에서 **effective WD strength = $\eta\lambda$**. 따라서 $(\eta, \lambda)$의 비율만 중요 → 1-dim optimization.

</details>

**문제 3** (이론-실전): 현대 LLM에서 **weight decay=0.1** (매우 큼)을 쓰는 이유는? SGD에서 보통 weight_decay=1e-4였다.

<details>
<summary>힌트 및 해설</summary>

**Scale difference의 원인**:

1. **Adam의 adaptive lr**: Adam이 effective lr을 작게 만듦 (normalize by $\sqrt{v}$). WD가 "같은 effect"를 주려면 $\lambda$를 **훨씬 크게** 설정.
   - SGD: $-\eta\lambda\theta \approx -\eta \cdot 10^{-4} \theta$.
   - AdamW: $-\eta\lambda\theta$에서 $\lambda = 0.1$이지만 Adam의 effective lr이 SGD의 1/1000 정도 → similar WD effect.

2. **Language model의 large parameter**: 각 parameter가 rich semantic 담음 → 약간의 shrinkage로 performance 크게 영향.

3. **Pre-training의 data abundance**: 과적합 위험이 크다 → 강한 regularization 필요.

**수치적 분석**:
- Llama 2 (7B): $\lambda = 0.1$, $\eta = 3 \times 10^{-4}$. Effective WD per step: $3 \times 10^{-5}$.
- ResNet SGD: $\lambda = 10^{-4}$, $\eta = 0.1$. Effective WD: $10^{-5}$.

**비슷한 magnitude** — 다른 notation이 같은 regularization strength.

**교훈**: $\lambda$ 수치를 비교할 때 **optimizer와 lr을 고려해야 함**. AdamW의 wd=0.1은 SGD의 wd=1e-4와 "**비슷한 regularization**".

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. SAM](./02-sam.md) | [📚 README로 돌아가기](../README.md) | [04. 4축 통합 Recipe ▶](./04-unified-recipe.md) |

</div>
