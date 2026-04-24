# 02. Dropout = Variational Inference (Gal & Ghahramani 2016)

## 🎯 핵심 질문

- 왜 Dropout이 **Bayesian 추론의 근사**로 해석되는가?
- **Bernoulli variational posterior** $q(W) = \prod \text{Bernoulli}$의 ELBO 최적화가 어떻게 Dropout + L2 loss와 같아지는가?
- **MC Dropout**은 왜 predictive uncertainty를 준다고 주장할 수 있는가?
- Ch2-01의 앙상블 해석과 이 VI 해석은 어떻게 공존·대립하는가?

---

## 🔍 왜 이 해석이 중요한가

Ch2-01의 앙상블 해석은 **deterministic** 앙상블을 말한다. Gal의 해석은 훨씬 더 **Bayesian**:

1. Dropout으로 훈련된 NN = **approximate posterior over weights**.
2. Test time에 Dropout을 **계속 활성** 유지 → 여러 샘플 → predictive distribution.
3. 이 분포의 **variance**가 예측 uncertainty.

이 관점의 힘은 네 가지.

- **Free uncertainty**: Dropout 있는 어느 네트워크에도 MC Dropout 적용 가능. BNN처럼 복잡한 posterior 추론 없이.
- **Deep Gaussian Process와의 연결**: Dropout이 GP의 approximate posterior로 유도됨.
- **weight decay ↔ prior**: L2 regularization strength가 prior precision에 대응 → Ch1-01과 직접 연결.
- **VI framework으로 확장 가능**: Concrete Dropout(Ch2-04), variational RNN dropout 등으로 일반화.

이 문서는 **ELBO를 loss로 전개**해 dropout + weight decay와의 동치를 보인다. 이 유도가 없으면 MC Dropout은 "heuristic"에 머물고 Bayesian 정당화를 받지 못한다.

---

## 📐 수학적 선행 조건

- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): **VI** — posterior $p(W|D) \approx q(W)$, ELBO $= \mathbb{E}_q[\log p(y|W)] - \text{KL}(q\|p)$
- Ch2-01: Dropout의 기본 메커니즘, Bernoulli mask
- Ch1-01: Gaussian prior의 negative log = L2 regularization
- 확률: Reparameterization trick, MC 추정자

---

## 📖 직관적 이해

### BNN에서 Dropout까지 거꾸로

**Bayesian NN (BNN)**: weight에 prior $p(W) = \mathcal{N}(0, I/\tau)$, posterior $p(W|D) \propto p(D|W) p(W)$. 예측:

$$p(y|x, D) = \int p(y|x, W) p(W|D) dW$$

정확한 $p(W|D)$는 intractable. **VI**는 parametric family $\{q_\phi(W)\}$에서 $p(W|D)$와 가장 가까운 $q^*$ 선택:

$$q^* = \arg\min_\phi \text{KL}(q_\phi(W) \| p(W|D)) = \arg\max_\phi \text{ELBO}(\phi)$$

### Gal의 아이디어

**Variational family로 "Bernoulli-scaled deterministic matrix"** 를 선택한다:

$$q(W_l) = \text{diag}(z_l) M_l, \quad z_{l,i} \sim \text{Bernoulli}(1-p)$$

여기서 $M_l$은 학습 가능한 **deterministic** 행렬, $z_l$은 Bernoulli random vector.

**관찰**: $W_l$을 샘플링한다는 것은 **$M_l$의 행 일부를 랜덤하게 0으로 설정**하는 것 — 이것이 정확히 **Dropout**.

ELBO를 전개하면 **Dropout loss + L2 regularization**이 튀어나온다 (정리 2.3). 결론: **Dropout된 NN의 훈련 = approximate BNN의 VI**.

### MC Dropout

Test 때도 Dropout을 활성 → weight $W^{(t)}$를 매번 샘플 → predictive distribution을 Monte Carlo로 근사:

$$p(y|x, D) \approx \frac{1}{T} \sum_{t=1}^T p(y|x, W^{(t)}), \quad W^{(t)} \sim q(W)$$

이 예측들의 **분산**이 epistemic uncertainty.

### 앙상블 vs VI의 대조

| 관점 | Train 해석 | Test 해석 |
|------|------|------|
| Ensemble (Ch2-01) | $2^N$ thinned nets | weight scaling으로 geometric mean 근사 |
| VI (이 문서) | Bernoulli $q(W)$의 ELBO 최대화 | MC samples로 predictive distribution |

두 관점은 **같은 알고리즘**을 다른 언어로 기술하지만 **다른 추론** 을 제안한다: 앙상블은 single prediction, VI는 predictive distribution.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — ELBO (Evidence Lower Bound)

관측 $D = \{(x_i, y_i)\}_{i=1}^n$, model parameter $W$에 대해:

$$\text{ELBO}(q) := \mathbb{E}_{W \sim q}[\log p(D|W)] - \text{KL}(q(W) \| p(W))$$

$\log p(D) \geq \text{ELBO}(q)$ 하한. ELBO 최대화 = KL 최소화.

### 정의 2.2 — Gal의 Variational Family

$L$-layer NN, layer $l$의 weight matrix $W_l \in \mathbb{R}^{K_l \times K_{l-1}}$. Variational posterior:

$$q(W_l) = M_l \text{diag}(z_l), \quad z_{l,i} \stackrel{\text{iid}}{\sim} \text{Bernoulli}(1-p_l)$$

여기서 $M_l$은 deterministic variational parameter, $p_l$은 layer별 dropout rate. Equivalent form: $W_l$의 $i$번째 **열**이 $m_{l,i} = z_{l,i}$로 랜덤 drop.

Prior: $p(W_l) = \mathcal{N}(0, I/\tau_l)$로 설정 ($\tau_l$은 prior precision).

### 정리 2.3 — ELBO = Dropout Loss + L2 (주 정리)

Regression 문제에서 Gaussian likelihood $y|x, W \sim \mathcal{N}(f_W(x), \tau^{-1})$를 가정하면, ELBO의 negative (loss)는 상수를 제외하고:

$$-\text{ELBO} \propto \frac{1}{n}\sum_{i=1}^n \underbrace{\frac{\tau}{2}\|y_i - f_{\hat{W}_i}(x_i)\|^2}_{\text{dropout loss}} + \sum_l \underbrace{\frac{p_l}{2\tau_l} \|M_l\|_F^2}_{\text{L2 regularization}} + \text{const}$$

여기서 $\hat{W}_i$는 sample $i$의 dropout-masked weight. 이는 정확히 **Dropout + weight decay**.

### 정리 2.4 — Predictive Distribution

Test 입력 $x^*$에 대해 predictive distribution:

$$p(y^*|x^*, D) \approx \int p(y^*|x^*, W) q(W) dW \approx \frac{1}{T}\sum_{t=1}^T p(y^*|x^*, W^{(t)})$$

여기서 $W^{(t)} \sim q(W)$ — 단순히 Dropout 활성 상태로 $T$번 forward pass.

**Predictive mean**: $\bar{f}(x^*) = \frac{1}{T}\sum_t f_{W^{(t)}}(x^*)$.

**Predictive variance**: $\text{Var}(y^*|x^*) \approx \tau^{-1} + \frac{1}{T}\sum_t \|f_{W^{(t)}}(x^*)\|^2 - \|\bar{f}(x^*)\|^2$.

첫 항은 **aleatoric (noise)** uncertainty, 둘째가 **epistemic (model)** uncertainty.

### 정리 2.5 — KL Approximation for Bernoulli $q$

$q(W_l) = M_l \text{diag}(z_l)$ ($z_l \sim \text{Bern}(1-p)$), $p(W_l) = \mathcal{N}(0, I/\tau_l)$:

$$\text{KL}(q \| p) \approx \frac{(1-p) \tau_l}{2} \|M_l\|_F^2 + \text{const}$$

(Gal의 논문 Appendix A에서 $M_l$이 각 row별 Gaussian-Bernoulli 혼합으로 근사될 때 유도.) 이 근사가 **L2 regularization term의 정체**.

---

## 🔬 수학적 유도

### 정리 2.3 증명 스케치

**ELBO 전개**:

$$\text{ELBO} = \sum_i \mathbb{E}_{q}[\log p(y_i|x_i, W)] - \text{KL}(q\|p)$$

**Gaussian likelihood**:

$$\log p(y_i|x_i, W) = -\frac{\tau}{2}\|y_i - f_W(x_i)\|^2 + \text{const}$$

**Monte Carlo 추정** (sample per data point):

$$\mathbb{E}_q[\log p(y_i|x_i, W)] \approx \log p(y_i|x_i, W_i), \quad W_i \sim q$$

즉 **한 번의 forward pass** (각 data point마다 다른 dropout mask).

**KL term** (정리 2.5):

$$\text{KL}(q\|p) \approx \sum_l \frac{(1-p_l) \tau_l}{2} \|M_l\|_F^2$$

합치면:

$$-\text{ELBO} \approx \frac{\tau}{2}\sum_i \|y_i - f_{W_i}(x_i)\|^2 + \sum_l \frac{(1-p_l) \tau_l}{2} \|M_l\|_F^2$$

$\tau, \tau_l$를 재조정하면 **Dropout loss + weight decay**로 정확히 같은 형태. $\square$

**대응**:
- Weight decay $\lambda_l = (1-p_l)\tau_l / (2\tau)$.
- Dropout rate $p_l$과 prior precision $\tau_l$이 함께 $\lambda_l$을 결정.

### MC Dropout 정당화 (정리 2.4)

Train 시 최적화된 $q^*(W)$ ≈ $p(W|D)$. Test:

$$p(y^*|x^*, D) = \int p(y^*|x^*, W) p(W|D) dW \approx \int p(y^*|x^*, W) q^*(W) dW$$

$q^*$에서 샘플 = dropout mask 샘플 = **test time dropout 활성 상태의 forward pass**. $T$ 샘플 → Monte Carlo 근사.

Classification task면 $p(y^*|x^*) = \text{softmax}(f_W(x^*))$, predictive distribution은 $T$개 softmax의 평균.

---

## 💻 실험으로 효과 검증

### 실험 1 — MC Dropout로 regression uncertainty

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
# Toy data: y = sin(2πx) + noise, x ∈ [-0.5, 0.5]
n = 30
X = torch.linspace(-0.5, 0.5, n).unsqueeze(1)
y = torch.sin(2*np.pi*X) + 0.05*torch.randn_like(X)

class MCDropoutNet(nn.Module):
    def __init__(self, p=0.1, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 1))
    def forward(self, x): return self.net(x)

net = MCDropoutNet(p=0.1)
opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)

for epoch in range(2000):
    opt.zero_grad()
    loss = ((net(X) - y)**2).mean()
    loss.backward(); opt.step()

# MC Dropout predict
x_eval = torch.linspace(-1.5, 1.5, 400).unsqueeze(1)
net.train()   # dropout ON
T = 200
preds = torch.stack([net(x_eval) for _ in range(T)]).squeeze(-1)
mean, std = preds.mean(0), preds.std(0)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c='k', label='train data', s=30)
plt.plot(x_eval, mean.detach(), 'b-', label='MC mean')
plt.fill_between(x_eval.squeeze(), (mean-2*std).detach(), (mean+2*std).detach(),
                 alpha=0.3, label='±2σ')
plt.plot(x_eval, torch.sin(2*np.pi*x_eval), 'g--', alpha=0.5, label='true function')
plt.axvspan(-0.5, 0.5, alpha=0.1, color='yellow', label='train region')
plt.xlabel('x'); plt.ylabel('y')
plt.title('MC Dropout — train region 밖에서 uncertainty 확장')
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

**관찰**: Train region ($[-0.5, 0.5]$) 내부에서는 uncertainty 작지만 **extrapolation region에서 uncertainty 크게 벌어짐** — Bayesian predictive distribution의 전형적 성질.

### 실험 2 — 분류에서 MC Dropout의 uncertainty

```python
from torch.utils.data import DataLoader, TensorDataset

# MNIST 일부 + OOD (Fashion-MNIST)를 이용 — 여기서는 대체 구현
class MCMLP(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 10))
    def forward(self, x): return self.net(x)

# 가상: net이 MNIST에 훈련되었다고 가정
# Test 시 in-distribution vs OOD 입력의 predictive entropy 비교

def mc_predict_with_uncertainty(net, x, T=50):
    net.train()
    with torch.no_grad():
        probs = torch.stack([torch.softmax(net(x), -1) for _ in range(T)])
    mean = probs.mean(0)
    # Entropy of predictive distribution (uncertainty)
    H = -(mean * mean.clamp_min(1e-10).log()).sum(-1)
    # Mutual information ~ epistemic uncertainty
    H_samples = -(probs * probs.clamp_min(1e-10).log()).sum(-1)
    MI = H - H_samples.mean(0)
    return mean, H, MI

# In-distribution (MNIST style) vs OOD에서 MI 히스토그램 비교
# → OOD에서 MI (epistemic)가 크게 증가
```

### 실험 3 — $\tau$ (prior precision)와 weight decay의 대응

```python
# ELBO-based loss
def elbo_loss(model, x, y, p_drop, tau_likelihood, tau_prior):
    n = x.shape[0]
    # Likelihood term (negative log)
    nll = 0.5 * tau_likelihood * ((model(x) - y)**2).sum()
    # KL approximation — sum over all Linear layers
    kl = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            kl += 0.5 * (1 - p_drop) * tau_prior * (m.weight**2).sum()
    return (nll + kl) / n

# 이 loss로 훈련하면 Dropout + weight decay 훈련과 동치임을 확인 가능
# weight decay = (1-p) * tau_prior / tau_likelihood
p_drop, tau_lik, tau_pri = 0.3, 1.0, 0.1
wd_equiv = (1 - p_drop) * tau_pri / tau_lik
print(f"weight_decay_equivalent = {wd_equiv}")
```

---

## 🔗 실전 활용

### MC Dropout을 쓰는 situations

1. **Active learning**: MI가 큰 sample부터 라벨링 (EPIG, BALD 등 acquisition function).
2. **Medical AI**: 진단 불확실성의 정량화 — "모델이 confident 한가?"
3. **Autonomous driving**: object detection의 uncertainty-aware downstream.
4. **OOD detection**: In-distribution vs OOD를 predictive entropy로 구별.

### 실전 요령

- $T$ (MC samples): 10~100 충분. 더 많이는 diminishing returns.
- Train·test 모두 같은 dropout rate 사용 (inconsistency를 피하기 위해).
- BatchNorm + Dropout은 권장하지 않음 — BN의 running stats 교란. LayerNorm + Dropout이 더 일관적.
- MC Dropout의 uncertainty는 **under-estimate** 경향 — Deep Ensembles(Lakshminarayanan 2017)가 더 보수적 uncertainty.

### 비판과 대안

- Osband 2016: MC Dropout uncertainty는 **prior에 독립적**인 성질이 있어 "proper Bayesian"에서 벗어남.
- Myshkov 2016: Dropout rate를 학습하지 않으면 uncertainty 규모가 임의적.
- **Concrete Dropout** (Gal 2017, Ch2-04): rate $p$도 학습 가능하게.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Bernoulli variational family | 복잡한 posterior는 더 풍부한 family 필요 (e.g. flow-based) |
| 모든 sample에 같은 mask 독립 | 실제로 BNN posterior는 correlated 가능성 |
| KL approximation (정리 2.5) | Gal 논문의 Gaussian-Bernoulli 근사는 거친 근사 |
| Gaussian likelihood (regression) | Classification에서는 cross-entropy → KL이 softmax와 상호작용 |
| Test time Dropout | Inference 비용 $T$배 증가 (50~100 samples 필요) |
| Posterior = $q$ 자체 | $q$와 true posterior의 KL은 유한하지 않을 수 있음 |

**주의**: "Dropout = exact Bayesian"은 **과장**이다. 실제로는 **heuristic approximation**이며, 여러 rigorous Bayesian 방법(HMC, variational with richer family)과 비교하면 uncertainty quality가 낮을 수 있다.

---

## 📌 핵심 정리

$$\boxed{\text{Dropout + L2 training} = \text{ELBO maximization for } q(W) = M \text{diag}(z), z \sim \text{Bern}(1-p)}$$

| 개념 | 의미 |
|------|------|
| **Variational posterior** | $q(W)$ = scaled Bernoulli-masked weight |
| **ELBO term 1** | Dropout forward-pass loss (Gaussian likelihood) |
| **ELBO term 2** | L2 regularization ≈ KL to Gaussian prior |
| **MC Dropout** | Test time $T$ stochastic passes → predictive distribution |
| **Weight decay ↔ Prior** | $\lambda = (1-p)\tau_l/\tau$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Dropout rate $p = 0.3$, prior precision $\tau_l = 1$, likelihood precision $\tau = 10$일 때, 해당하는 weight decay $\lambda$는?

<details>
<summary>힌트 및 해설</summary>

정리 2.3의 대응: $\lambda = (1-p)\tau_l / (2\tau)$. 계산: $(0.7)(1) / (20) = 0.035$.

해석: Prior를 **더 좁게** ($\tau_l$ ↑) 두거나 likelihood noise를 **더 크게** ($\tau$ ↓) 두면 $\lambda$가 증가 → 더 강한 regularization. 이것이 Bayesian의 "prior 믿음 ↔ regularization" 직접적 대응.

</details>

**문제 2** (심화): Gal의 variational family $q(W) = M \text{diag}(z)$는 **rank-1 Bernoulli** 구조. 왜 이 특수한 형태에서 KL이 L2 norm처럼 간단히 근사되는가? 더 일반적 variational family라면 어떤 어려움이 있는가?

<details>
<summary>힌트 및 해설</summary>

$q(W)$에서 $z$가 **iid Bernoulli**이고 $M$이 deterministic이므로 $q$는 mixture of Diracs at $M_{:, i}$ (각 column이 "on" 또는 "off"). 이 discrete structure에서 Gaussian prior $p(W)$와의 KL은 각 column의 $L^2$ norm의 가중합으로 단순화된다.

**더 복잡한 family** (e.g. continuous Gaussian posterior $q(W) = \mathcal{N}(\mu, \Sigma)$): KL = $\tfrac{1}{2}[\text{tr}(\tau_l\Sigma) + \tau_l\|\mu\|^2 - \log\det\Sigma - d]$ — 이 경우 $\log\det\Sigma$ term이 추가로 정보 기하를 반영. Loss가 더 복잡하지만 더 풍부한 posterior를 표현.

Gal의 선택은 **"표현력은 제한적이지만 구현이 단순"** — L2 + Dropout이라는 매우 익숙한 형태로 환원되기 위한 최소 구조. Practical success의 이유.

</details>

**문제 3** (이론-실전): Deep Ensembles (Lakshminarayanan 2017): $K$개의 서로 다른 initialization으로 훈련된 NN의 앙상블. 이는 Bayesian posterior의 근사인가? MC Dropout과 비교해 uncertainty quality는 어떤 차이가 있는가?

<details>
<summary>힌트 및 해설</summary>

**Deep Ensembles는 정확히 Bayesian이 아니지만 "더 나은 uncertainty"로 종종 더 성공**.

- MC Dropout: 하나의 네트워크에서 mask로만 variation. **Mode 하나**의 주변에 restricted.
- Deep Ensembles: 독립적 훈련으로 **다른 mode** 탐색 가능. Posterior multimodality를 더 잘 포착.

Fort 2019 "Deep Ensembles: A Loss Landscape Perspective" — Deep Ensembles가 서로 다른 loss landscape mode로 수렴하는 반면 MC Dropout은 같은 mode 근처. Multimodal posterior에서는 전자가 훨씬 정확.

**실전 trade-off**:
- MC Dropout: 훈련 1회, inference 비용 $T$배 (10~50).
- Deep Ensembles: 훈련 $K$배, inference $K$배 (5 정도로도 충분).

Snapshot Ensemble (Huang 2017)처럼 "cyclical LR로 여러 mode 방문"하는 중간 기법도 존재. Ch7-01의 SWAG는 또 다른 절충.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Dropout = 앙상블](./01-dropout-ensemble.md) | [📚 README로 돌아가기](../README.md) | [03. Dropout = Adaptive L2 ▶](./03-dropout-adaptive-l2.md) |

</div>
