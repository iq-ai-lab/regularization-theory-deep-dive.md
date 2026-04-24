# 01. Stochastic Weight Averaging (Izmailov et al. 2018)

## 🎯 핵심 질문

- **SWA** $\bar{\theta} = \frac{1}{T}\sum_{t=t_0}^{t_0+T} \theta_t$는 왜 flat region에 있는 해를 주는가?
- Loss surface 곡률과 iterate averaging의 수학적 관계는?
- **SWAG** (Maddox 2019)의 SWA 위에 **Gaussian posterior**는 어떻게 Bayesian uncertainty를 주는가?
- CIFAR-100에서 SWA vs SGD의 실전 비교는?

---

## 🔍 왜 SWA가 필요한가

Ch6-02에서 SGD가 flat minimum을 선호한다는 것을 봤다. **하지만**:

- SGD가 수렴한 후에도 $\theta_t$는 **minimum 주변에서 oscillate** (mini-batch noise 때문).
- Single $\theta_t$는 **정확한 minimum이 아닌 근처 noisy point**.
- 여러 $\theta_t$를 **평균**하면 noise 상쇄 + 더 flat region.

Izmailov et al. 2018의 단순하지만 강력한 아이디어: 훈련 후반부의 iterate를 **평균**한다.

**Benefits**:
1. Test accuracy 개선 (CIFAR-100에서 ~1-2%).
2. Generalization gap 감소.
3. **Implementation trivial** — 1-2 줄 코드 추가.

**Extension - SWAG** (Maddox 2019): SWA mean + sample covariance로 **Gaussian posterior** 구축 → Bayesian uncertainty 거의 무료로.

---

## 📐 수학적 선행 조건

- Ch6-02: Flat vs sharp minima, SGD bias
- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): Gaussian posterior, BNN
- 기본 loss landscape 이해

---

## 📖 직관적 이해

### SWA의 정의

Standard SGD + iterate averaging:

$$\bar{\theta}_T = \frac{1}{T}\sum_{t=t_0}^{t_0+T} \theta_t$$

- $t_0$: "averaging 시작 시점" (보통 75-90% epoch 후).
- $T$: averaging duration.

**구현**: Running average 유지:

$$\bar{\theta} \leftarrow \bar{\theta} + \frac{1}{n+1}(\theta_t - \bar{\theta})$$

($n$ is the number of iterates averaged so far.)

### Cyclical Learning Rate

SWA는 보통 **cyclical lr**와 함께:
- High lr 주기적으로 사용 → loss surface 탐험.
- 각 "cycle"의 끝에서 $\theta_t$ sampling → **여러 flat region 방문**.
- 평균으로 robust center 찾기.

### Geometric Intuition

Loss surface가 **asymmetric minimum** 주변에 있을 때:

- SGD iterate는 minimum "뾰족한 쪽"으로 치우침.
- 평균 iterate는 **valley의 center**로 이동.
- Valley center가 일반적으로 **더 flat** → better generalization.

### SWAG — Bayesian Extension

SWA가 mean만 준다. Covariance도 수집하면:

$$p(\theta | D) \approx \mathcal{N}(\bar\theta, \Sigma_{\text{SWA}})$$

$\Sigma_{\text{SWA}} = \tfrac{1}{2}[\Sigma_{\text{diag}} + \Sigma_{\text{low-rank}}]$ (Maddox의 approximation).

이 Gaussian posterior에서 sample → **Deep Ensemble과 유사한 uncertainty**. Ch2-02의 MC Dropout과 유사.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Stochastic Weight Averaging

SGD trajectory $\{\theta_t\}_{t=1}^T$. SWA weight after $T$ iterates (from $t_0$):

$$\bar{\theta}_{\text{SWA}} = \frac{1}{T - t_0 + 1}\sum_{t=t_0}^T \theta_t$$

$t_0$ = averaging start (hyperparameter).

### 정의 1.2 — Running Average Implementation

$$n \leftarrow n + 1, \quad \bar{\theta} \leftarrow \bar{\theta} + \frac{1}{n}(\theta_t - \bar{\theta})$$

Memory cost: $2 \times$ parameter count (current $\theta$ + running average).

### 정리 1.3 — SWA Moves Toward Valley Center

Asymmetric quadratic loss $L(\theta) = \theta^T H \theta / 2$ with anisotropic Hessian $H$. SGD iterates oscillate in $H$의 large-eigenvalue directions. **Average** reduces variance in those directions:

$$\text{Var}(\bar{\theta}_k) \approx \text{Var}(\theta_t)_k / T$$

More averaging → less variance → closer to "mean position".

### 정리 1.4 — Izmailov 2018 Empirical Results

CIFAR-100 ResNet-164:
- SGD: 74.8% test accuracy.
- SGD + SWA: **76.3%** (+1.5%).

ImageNet ResNet-50:
- SGD: 76.2%.
- SGD + SWA: 76.6%.

Generalization gap (train - test) 감소 특히 overfitting 큰 setting.

### 정의 1.5 — SWAG (Maddox et al. 2019)

SWA mean + low-rank + diagonal Gaussian:

$$p_{\text{SWAG}}(\theta | D) = \mathcal{N}\left(\bar{\theta}, \frac{1}{2}\Sigma_{\text{diag}} + \frac{1}{2}\Sigma_{\text{low-rank}}\right)$$

Where:
- $\Sigma_{\text{diag}} = \text{diag}(\overline{\theta_t^2} - \bar{\theta}^2)$ (running 2nd moment).
- $\Sigma_{\text{low-rank}} = \tfrac{1}{K-1}\sum_k (\theta_{t_k} - \bar{\theta})(\theta_{t_k} - \bar{\theta})^T$ (rank-$K$).

**Uncertainty**: Sample $\theta^{(s)} \sim p_{\text{SWAG}}$, forward pass, aggregate.

### 정리 1.6 — SWA와 Flat Minimum

Averaged weight의 effective sharpness:

$$\text{sharpness}(\bar{\theta}) \leq \text{sharpness}(\theta_t^*)$$

$\theta_t^*$는 단일 converged iterate. 평균이 **valley center**에 더 가까워 Hessian eigenvalues 작음.

---

## 🔬 수학적 유도

### SWA가 Variance 감소시키는 이유

SGD가 SDE로 모델링: $d\theta = -\nabla L dt + \sqrt{2T_{\text{eff}}}dB$.

Stationary distribution $p^*(\theta) \propto e^{-L/T_{\text{eff}}}$. Average $\bar{\theta} = \int \theta \cdot p^*(\theta) d\theta$는 local minimum $\theta^*$.

**하지만** single iterate $\theta_t$는 **sample**이고, SWA는 **empirical mean**.

$$\bar{\theta}_T = \frac{1}{T}\sum_t \theta_t \to \mathbb{E}_{p^*}[\theta] \text{ as } T \to \infty$$

By law of large numbers:

$$\text{Var}(\bar{\theta}_T) \approx \text{Var}(\theta_t) / T$$

(Independent case. SGD iterate는 correlated하지만 효과 유사.)

### Why Flat Minimum

Wide valley + SGD mixing → iterates가 valley 전체를 cover. 평균 위치 = **valley의 중심**.

Narrow valley + SGD → iterates가 고정된 위치에 집중. 평균이 큰 차이 없음.

**결론**: SWA가 **wide valley (flat minimum)에서 더 큰 이점** — 이게 generalization 개선의 메커니즘.

### Cyclic LR의 역할

Constant lr: 한 minimum 주변에만 머묾.
Cyclic lr (high lr로 주기적 복귀): 여러 minimum 탐험 → diverse averaging.

Izmailov 2018: cyclic lr + SWA가 standard SGD보다 정확히 1-2% 개선.

---

## 💻 실험으로 효과 검증

### 실험 1 — SWA 구현

```python
import torch
import torch.nn as nn
from copy import deepcopy

class SWAWrapper:
    def __init__(self, model):
        self.swa_model = deepcopy(model)
        self.n = 0
    
    def update(self, model):
        """Running average update."""
        self.n += 1
        for swa_p, p in zip(self.swa_model.parameters(), model.parameters()):
            swa_p.data += (p.data - swa_p.data) / self.n
    
    def update_bn_stats(self, loader, device):
        """Recompute BN running stats with SWA weights."""
        self.swa_model.train()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                self.swa_model(x)
        self.swa_model.eval()

# 훈련 루프
model = nn.Sequential(...)   # ResNet 등
swa = SWAWrapper(model)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# 처음 75% epoch: standard SGD
for epoch in range(75):
    train_standard(model, opt)

# 나머지 25% epoch: SGD + SWA
for epoch in range(25):
    train_standard(model, opt)
    swa.update(model)  # update SWA weight

# Test with SWA weights
swa.update_bn_stats(train_loader, 'cuda')
test_acc = evaluate(swa.swa_model, test_loader)
```

### 실험 2 — PyTorch 내장 SWA 사용

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# Setup
model = ResNet18()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

swa_start = 75
for epoch in range(100):
    for x, y in train_loader:
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    if epoch >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()

# Update BN statistics
update_bn(train_loader, swa_model, device='cuda')

# Evaluate
swa_model.eval()
test_acc = evaluate(swa_model, test_loader)
```

### 실험 3 — Loss landscape 시각화 (SGD vs SWA)

```python
import numpy as np
import matplotlib.pyplot as plt

def loss_in_neighborhood(model, loss_fn, batch, radius=5, N=30):
    """Sample random directions, compute loss."""
    # Save original weights
    orig = {n: p.data.clone() for n, p in model.named_parameters()}
    
    losses = []
    for _ in range(N):
        # Random perturbation of magnitude 'radius'
        for p in model.parameters():
            p.data += radius * torch.randn_like(p) * 0.01
        losses.append(loss_fn(model(batch[0]), batch[1]).item())
        # Restore
        for n, p in model.named_parameters():
            p.data = orig[n].clone()
    return losses

# SGD model의 neighborhood vs SWA model의 neighborhood
# → SWA neighborhood의 loss variance가 더 작음 (flat)
```

### 실험 4 — CIFAR-100 실험 재현

```python
# Ablation: epoch별 SGD vs SWA test accuracy
# 전형적 결과 (ResNet-56 CIFAR-100):
# SGD at epoch 200:     72.5%
# SGD + SWA (75-200):   74.0%
# Difference:           +1.5%
```

### 실험 5 — SWAG 구현 sketch

```python
class SWAGWrapper:
    def __init__(self, model, max_rank=20):
        self.mean = deepcopy(model)
        self.sq_mean = deepcopy(model)  # for variance
        self.deviations = []            # low-rank devs
        self.n = 0
        self.max_rank = max_rank
    
    def update(self, model):
        self.n += 1
        # Update mean
        for m_p, p in zip(self.mean.parameters(), model.parameters()):
            m_p.data += (p.data - m_p.data) / self.n
        # Update 2nd moment
        for sq_p, p in zip(self.sq_mean.parameters(), model.parameters()):
            sq_p.data += (p.data**2 - sq_p.data) / self.n
        # Store deviation (low-rank)
        dev = [p.data - m_p.data for p, m_p in zip(model.parameters(), self.mean.parameters())]
        self.deviations.append(dev)
        if len(self.deviations) > self.max_rank:
            self.deviations.pop(0)
    
    def sample(self):
        """Sample from SWAG posterior."""
        new_params = []
        for i, (m_p, sq_p) in enumerate(zip(self.mean.parameters(), self.sq_mean.parameters())):
            var_diag = (sq_p.data - m_p.data**2).clamp_min(0)
            z_diag = torch.randn_like(m_p.data) * var_diag.sqrt()
            # Low-rank contribution
            z_lr = 0
            for dev in self.deviations:
                z_lr += dev[i] * torch.randn(1).item() / (len(self.deviations) - 1)**0.5
            new_params.append(m_p.data + 0.5**0.5 * z_diag + 0.5**0.5 * z_lr)
        return new_params

# Usage: MC samples로 predictive uncertainty 추정
```

---

## 🔗 실전 활용

### SWA의 표준 setup

1. **Warmup phase**: 처음 75% epoch은 standard SGD with momentum.
2. **SWA phase**: 나머지 25%에서 cyclical lr + iterate averaging.
3. **BN recompute**: 최종 SWA weight로 BN stats 다시 계산 (**critical!**).

### Hyperparameters

- **$t_0$ (averaging start)**: 전체 epoch의 75~80%.
- **Cyclic lr range**: $[0.01, 0.05]$ (base lr $0.1$의 10~50%).
- **Averaging frequency**: 매 epoch (가장 간단), 또는 매 cycle 끝.

### BatchNorm Trick

SWA weight은 SGD와 다른 분포에서 샘플 → BN running stats가 **mismatch**. 해결:
- 훈련 끝나고 SWA weights를 model에 로드.
- Training mode로 1 epoch 돌려 BN stats 업데이트 (**no gradient step**).
- 이것만으로 test accuracy 크게 개선.

### 언제 SWA를 쓰는가

- **Efficient ensemble alternative**: Deep Ensembles의 $K$배 cost 없이 비슷한 효과.
- **Production deployment**: Single model로 inference.
- **Any SGD-trained model**: CNN, Transformer 모두 적용 가능.

### SWA의 대안 / 관련 기법

- **EMA (Exponential Moving Average)**: $\bar\theta_{t+1} = \alpha\bar\theta_t + (1-\alpha)\theta_{t+1}$. Softer moving window. Transformer 훈련에 종종.
- **Lookahead** (Zhang 2019): Fast + slow weights interleaved. SWA와 유사 philosophy.
- **Polyak averaging**: SWA의 전신, 전체 training에 걸친 averaging (1990년대 optimization).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Wide valley 존재 | Sharp minimum에서는 SWA 효과 미미 |
| BN recompute | 실수로 skip하면 성능 급락 |
| SGD with momentum | Adam 등 adaptive optimizer와 결합 시 trial 필요 |
| Averaging 기간 충분 | Short averaging window는 효과 작음 |
| Cyclical lr | 복잡한 lr scheduling 필요 |

---

## 📌 핵심 정리

$$\boxed{\bar{\theta}_{\text{SWA}} = \frac{1}{T}\sum_t \theta_t \to \text{flat valley center}}$$

| 개념 | 의미 |
|------|------|
| **SWA** | SGD iterate의 평균 weights |
| **Flat minimum** | 평균이 valley center로 이동 → generalization ↑ |
| **SWAG** | SWA + Gaussian posterior → Bayesian uncertainty |
| **BN recompute** | SWA weight으로 BN stats 재계산 필수 |
| **Ch7 시작** | Modern regularization의 첫 번째 — landscape tool |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 100 epoch 훈련에서 SWA는 보통 어떤 schedule로 시작하는가?

<details>
<summary>힌트 및 해설</summary>

**표준**: 75 epoch까지 standard SGD, 75-100 epoch에서 SWA (cyclical lr + averaging).

- **Why 75%**: 이 시점이 model이 "거의 수렴"한 상태. 더 일찍 시작하면 "convergence되지 않은 iterate까지 평균" → noise 많음.
- **Cyclic lr range**: warmup lr의 1/10 ~ 1/2.

실전 tip: 짧은 훈련에서는 $t_0$ 비율을 더 높게 (e.g. 90%). 긴 훈련에서는 70% 가능.

</details>

**문제 2** (심화): SWA가 "flat minimum 찾는다"는 주장은 **loss의 Hessian** 관점에서 어떻게 나타나는가?

<details>
<summary>힌트 및 해설</summary>

Loss $L(\theta) \approx L(\theta^*) + \tfrac{1}{2}(\theta - \theta^*)^T H(\theta - \theta^*)$.

SGD iterate $\theta_t$는 $\theta^*$ 주변에서 noisy sampling — covariance related to $H^{-1}$ (Fokker-Planck).

SWA mean $\bar\theta \approx \theta^*$ (LLN).

**Hessian of SWA's effective loss**:
- Static $\theta_t$: $H$ evaluated at single point.
- SWA: averaging smoothed Hessian over basin.

"Smoothed Hessian의 eigenvalues가 작다" → flat loss curve. Isolation of single $\theta_t$에서는 sharper.

**empirical measurement** (Izmailov 2018 Fig 6): SWA model의 Hessian spectral norm이 SGD보다 **훨씬 작음** (2-10배).

**함의**: SWA가 effectively "flat regularization" — explicit regularization 아니지만 landscape property 얻음.

</details>

**문제 3** (이론-실전): SWAG의 Gaussian posterior는 정확히 Bayesian posterior인가? 왜 effective uncertainty를 주는가?

<details>
<summary>힌트 및 해설</summary>

**정확한 Bayesian은 아님**. SWAG의 Gaussian은 **SGD iterate의 sample covariance**로 posterior approximation.

**근사 수준**:
- SGD iterate가 posterior stationary distribution에서 샘플된다고 가정 (Mandt 2017).
- 실제로는 SGD가 Bayesian MCMC가 아님 — iterate 간 correlation, boundary effects.

**그래도 uncertainty가 effective인 이유**:
1. **Local posterior shape**: Loss surface 주변의 Gaussian 근사. Most BNN 근사 기법과 비슷.
2. **Multiple flat modes**: Cyclical lr로 여러 mode 방문 → posterior multimodality 일부 캡처.
3. **Computational free**: 추가 훈련 없이 이미 수집한 iterate 사용.

**SWAG vs Deep Ensembles**:
- Deep Ensemble: $K$개 독립 훈련 → **different local minima** 포착.
- SWAG: 한 훈련에서 여러 iterate → **one local basin 내부 structure**.
- Uncertainty quality: Deep Ensemble > SWAG > MC Dropout (Maddox 2019 실험).

**실전 위치**: SWAG는 "낮은 cost Bayesian approximation" — medical, safety-critical 아닌 일반 applications에 적합.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Chapter 6 → 04. Homogeneous Networks](../ch6-early-stopping-implicit/04-feature-implicit-bias.md) | [📚 README로 돌아가기](../README.md) | [02. Sharpness-Aware Minimization ▶](./02-sam.md) |

</div>
