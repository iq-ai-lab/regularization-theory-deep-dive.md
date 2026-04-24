# 01. Dropout = 앙상블 근사 (Srivastava et al. 2014)

## 🎯 핵심 질문

- Dropout의 원래 동기는 무엇이었는가? 왜 "뉴런을 랜덤하게 끔"이 효과적인가?
- 훈련 때 $2^N$개의 **thinned subnetwork**가 생기는 것을 inference 때 어떻게 평균내는가?
- Test-time **weight scaling** $\times(1-p)$는 정확히 어떤 양의 근사인가?
- PyTorch의 `nn.Dropout`(inverted dropout)과 원 논문의 convention은 어떻게 다른가?

---

## 🔍 왜 이 해석이 중요한가

Dropout은 가장 널리 쓰이는 regularization 중 하나지만 **"왜 되는지"** 에 대한 답은 하나가 아니다. 이 레포는 세 가지 해석을 나란히 제시한다.

1. **앙상블 (이 문서)**: 각 forward pass가 다른 subnetwork → $2^N$ 앙상블의 근사
2. **Variational Inference** (Ch2-02): Bernoulli variational posterior의 ELBO 최적화
3. **Adaptive L2** (Ch2-03): linear model에서 feature별 L2로 환원

이 문서는 Srivastava 2014의 **원래 직관**을 엄밀화한다. 핵심은 두 가지:

- 훈련 중 **noise 주입 → generalization 개선**이라는 일반 원리
- Test time의 weight scaling이 **앙상블 출력의 geometric mean을 근사**한다는 구체적 정리

이것이 없으면 "왜 Dropout rate 0.5가 표준인가", "왜 inference에서 Dropout을 꺼야 하는가"라는 기본 질문에 답할 수 없다.

---

## 📐 수학적 선행 조건

- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): forward pass, activation function
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Bernoulli 분포, 기댓값 계산
- 정보이론 기초: geometric mean $\text{GM}(x_1, \ldots, x_K) = (\prod_k x_k)^{1/K}$, Jensen 부등식
- Ch1-01: L2 regularization의 concept — 나중에 대조

---

## 📖 직관적 이해

### Dropout의 메커니즘

훈련 중 각 뉴런의 activation을 확률 $p$로 0으로 만든다. 수식으로:

$$\tilde{h}_i = m_i \cdot h_i, \quad m_i \sim \text{Bernoulli}(1-p) \quad \text{iid}$$

매 mini-batch (혹은 매 sample)마다 다른 mask $m$을 샘플.

### 왜 이게 regularization인가 (Srivastava의 원 해석)

1. **Co-adaptation 방지**: 뉴런이 특정 조합에 의존하지 못함. 각 뉴런이 **혼자 힘으로** 유용해야 한다.
2. **앙상블**: 각 mask는 다른 architecture. 총 $2^N$개의 thinned network. Test 때 모두 평균 내면 variance 감소.
3. **Noise robustness**: 훈련 중 input의 일부가 사라져도 예측 안정 → feature에 robustness 주입.

이 세 해석 중 **(2) 앙상블**이 가장 엄밀화 가능. 나머지는 직관적이지만 정량적 이론은 없다.

### Test-time weight scaling

$N$개 뉴런 전체에 대해 모든 mask를 나열하려면 $2^N$번의 forward pass 필요 (계산 불가능). Srivastava는 하나의 forward pass로 근사:

$$\text{테스트 때: 뉴런 출력에 } (1-p) \text{를 곱한다}$$

혹은 동치적으로 **inverted dropout** (PyTorch 표준): 훈련 때 $1/(1-p)$로 미리 scale하고 test 때는 그대로.

**왜 $(1-p)$?** 훈련 때 뉴런이 평균 $(1-p) \cdot h$로 나타나므로, test 때 그 "평균 규모"로 맞춰준다. 더 엄밀히는 **geometric mean approximation** (정리 1.3).

### 핵심 대응표

| 상황 | Output |
|------|--------|
| 훈련 (mask $m$) | $f(x; m \odot W)$ |
| 완전 앙상블 | $\mathbb{E}_m[f(x; m \odot W)]$ — $2^N$ forward pass 평균 |
| Weight scaling | $f(x; (1-p) W)$ — 단일 forward, geometric mean 근사 |

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Dropout Operator

Layer의 hidden vector $h \in \mathbb{R}^N$에 대해:

$$\text{Dropout}_p(h) = m \odot h, \quad m_i \stackrel{\text{iid}}{\sim} \text{Bernoulli}(1-p)$$

"drop rate" $p$, "keep rate" $(1-p)$. Srivastava 2014 표준은 $p = 0.5$ (hidden), $p = 0.2$ (input).

### 정의 1.2 — Full Ensemble Prediction

$$\hat{y}_{\text{ens}}(x) = \mathbb{E}_m\left[f(x; m \odot W)\right] = \frac{1}{2^N} \sum_{m \in \{0,1\}^N} f(x; m \odot W)$$

(각 $m$을 keep probability로 가중하면 $\sum_m (1-p)^{|m|} p^{N-|m|} f(x; m\odot W)$.)

### 정리 1.3 — Weight Scaling = Geometric Mean (Softmax Network)

**Single-layer softmax 네트워크** $f(x; W) = \text{softmax}(W^T x)$에서, class $c$에 대한 log-prediction의 weight scaling:

$$\log f_c(x; (1-p) W) = \log \text{softmax}_c((1-p) W^T x)$$

는 geometric mean 앙상블의 log-prediction:

$$\log \text{GM}(f_c) = \mathbb{E}_m[\log f_c(x; m \odot W)]$$

에 **근사**. 등호는 일반적으로 성립하지 않지만 **많은 경우에 좋은 근사** (Baldi & Sadowski 2013).

### 정리 1.4 — Linear Output의 경우 정확한 동치

Linear output $f(x; W) = W^T x$ (no nonlinearity, no softmax)에서:

$$\mathbb{E}_m[f(x; m \odot W)] = (1-p) W^T x = f(x; (1-p)W)$$

따라서 linear model에서는 **weight scaling이 정확히 앙상블 평균**.

### 정리 1.5 — Single ReLU의 Weight Scaling Error

$f(x; W) = \max(0, W^T x)$의 경우 weight scaling은 geometric mean의 정확한 근사가 아니지만, 오차는 $O(p^2)$로 작음 (Baldi-Sadowski 2013의 Taylor 전개).

### 정리 1.6 — MC Dropout Monte Carlo Approximation

$T$번 stochastic forward pass로 앙상블 평균 근사:

$$\hat{y}_{\text{MC}}(x) = \frac{1}{T}\sum_{t=1}^T f(x; m^{(t)} \odot W), \quad m^{(t)} \stackrel{\text{iid}}{\sim} \text{Bernoulli}(1-p)$$

$T \to \infty$에서 true ensemble mean으로 수렴 (대수의 법칙). Ch2-02와 연결.

---

## 🔬 수학적 유도

### 정리 1.4 증명 (linear model의 정확한 동치)

$f(x; m \odot W) = \sum_i m_i w_i x_i$. 기댓값:

$$\mathbb{E}_m[\sum_i m_i w_i x_i] = \sum_i \mathbb{E}[m_i] w_i x_i = (1-p) \sum_i w_i x_i = f(x; (1-p)W) \quad \square$$

Linear에서는 기댓값이 weight에 대한 linear 연산이므로 **정확히** weight scaling과 같다.

### 정리 1.3 유도 (softmax geometric mean)

$f_c(x; m\odot W) = \exp((m\odot W)_c^T x) / Z(x; m\odot W)$, $Z$는 softmax denominator.

$\log f_c(x; m\odot W) = (m\odot W)_c^T x - \log Z$.

**Geometric mean 정의**:

$$\log \text{GM}(f_c) = \mathbb{E}_m[\log f_c(x; m\odot W)] = (1-p) W_c^T x - \mathbb{E}_m[\log Z]$$

첫 항은 $(1-p) W_c^T x$. 둘째 항은 $\mathbb{E}_m[\log Z]$로 일반적으로 $\log \mathbb{E}_m[Z]$와 다름 (Jensen).

**Weight scaling** $f_c(x; (1-p)W) = \exp((1-p)W_c^T x)/Z'$, $Z' = \sum_k \exp((1-p)W_k^T x)$.

$\log f_c(x; (1-p)W) = (1-p) W_c^T x - \log Z'$.

두 식의 차이는 $\log Z'$ vs $\mathbb{E}_m[\log Z]$. Baldi-Sadowski는 $\exp$의 Taylor 전개와 분산 항 분석으로 $|\mathbb{E}_m[\log Z] - \log Z'| = O(p^2 \|W\|^2)$임을 보인다. 작은 $p$나 compact output 분포에서 좋은 근사. $\square$

### 왜 geometric mean인가

각 mask의 prediction $p_m = f_c(x; m\odot W)$. 완전 앙상블의 arithmetic mean $\bar{p} = \mathbb{E}_m[p_m]$ vs geometric mean $\tilde{p} = \exp(\mathbb{E}_m[\log p_m])$.

- Arithmetic mean → 계산 위해 모든 $p_m$ 필요 ($2^N$).
- Geometric mean → log 공간에서 평균이므로 softmax의 **logit 평균**과 일치 → 하나의 forward pass로 근사 가능.

따라서 weight scaling은 "**log space 앙상블**"의 근사.

---

## 💻 실험으로 효과 검증

### 실험 1 — Linear model에서 weight scaling이 정확함을 확인

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
N, p = 50, 0.3
W = torch.randn(10, N)              # (output=10, input=N)
x = torch.randn(N)

# Full ensemble (2^N 불가능 → Monte Carlo large T)
T = 10000
preds_mc = torch.stack([W @ (torch.bernoulli(torch.full((N,), 1-p)) * x) for _ in range(T)])
mean_mc = preds_mc.mean(0)

# Weight scaling
pred_ws = (1 - p) * W @ x

print("MC ensemble mean  :", mean_mc[:3].tolist())
print("Weight scaling    :", pred_ws[:3].tolist())
print("max |diff|        :", (mean_mc - pred_ws).abs().max().item())
# → 0에 가까운 값 (MC 오차 수준)
```

### 실험 2 — Softmax의 approximation error 측정

```python
def softmax_logits(W, x):  # returns logits
    return W @ x

def ensemble_softmax(W, x, p, T=5000):
    probs = []
    for _ in range(T):
        m = torch.bernoulli(torch.full((W.shape[1],), 1-p))
        logits = W @ (m * x)
        probs.append(torch.softmax(logits, dim=0))
    return torch.stack(probs).mean(0)

def weight_scaling_softmax(W, x, p):
    return torch.softmax((1-p) * W @ x, dim=0)

W = torch.randn(5, 20)
x = torch.randn(20)
for p in [0.1, 0.3, 0.5, 0.7]:
    arith = ensemble_softmax(W, x, p)
    ws = weight_scaling_softmax(W, x, p)
    err = (arith - ws).abs().max().item()
    print(f"p={p:.1f}: max |arithmetic mean - weight scaling| = {err:.4f}")
# → p가 클수록 오차 증가, 하지만 여전히 작은 크기
```

### 실험 3 — Inverted Dropout (PyTorch 표준) 확인

```python
class ExplicitDropout(nn.Module):
    """원 논문 convention: 훈련 때 mask, test 때 scale."""
    def __init__(self, p): super().__init__(); self.p = p
    def forward(self, x):
        if self.training:
            m = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return m * x
        else:
            return (1 - self.p) * x

class InvertedDropout(nn.Module):
    """PyTorch nn.Dropout: 훈련 때 scale, test 때 그대로."""
    def __init__(self, p): super().__init__(); self.p = p
    def forward(self, x):
        if self.training:
            m = torch.bernoulli(torch.full_like(x, 1 - self.p)) / (1 - self.p)
            return m * x
        else:
            return x

# 두 구현의 test-time 출력이 같은지 확인
x = torch.randn(10)
e = ExplicitDropout(0.3); e.eval()
i = InvertedDropout(0.3); i.eval()
print("Explicit test output  :", e(x)[:3])
print("Inverted test output  :", i(x)[:3])
# → Explicit은 (1-p)*x, Inverted는 x — 기대값은 같지만 scale 다름

# 훈련 시 기댓값 비교
e.train(); i.train()
E_e = torch.stack([e(x) for _ in range(2000)]).mean(0)
E_i = torch.stack([i(x) for _ in range(2000)]).mean(0)
print("E[e_train(x)] ≈", E_e[:3])   # ≈ (1-p) * x
print("E[i_train(x)] ≈", E_i[:3])   # ≈ x (inverted가 추가 1/(1-p) 곱하므로)
```

**결론**: PyTorch `nn.Dropout`은 inverted dropout. 훈련·테스트 모두 기댓값이 $x$로 일치하도록 설계.

### 실험 4 — 작은 네트워크에서 앙상블 vs weight scaling vs MC

```python
class MLP(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(20, 50)
        self.dropout = ExplicitDropout(p)
        self.fc2 = nn.Linear(50, 3)
    def forward(self, x):
        return self.fc2(torch.relu(self.dropout(self.fc1(x))))

net = MLP(0.3)
x = torch.randn(20)

# 1. True ensemble (large MC)
net.train()
T = 5000
preds = torch.stack([torch.softmax(net(x), -1) for _ in range(T)]).mean(0)

# 2. Weight scaling
net.eval()
ws = torch.softmax(net(x), -1)

print("Ensemble (T=5000):", preds)
print("Weight scaling :", ws)
print("KL divergence  :", torch.sum(preds * (preds.log() - ws.log())).item())
# → 작은 KL (좋은 근사), 하지만 0은 아님 (nonlinearity 때문)
```

---

## 🔗 실전 활용

### Dropout rate 선택

- **Hidden layer**: $p = 0.5$ — 원 논문 권장. 매우 공격적 regularization.
- **Input layer**: $p = 0.1 \sim 0.2$ — input 정보 손실 최소화.
- **Attention / FFN (Transformer)**: $p = 0.1$ — 많은 regularization 이미 존재 (LN, weight decay).
- **Overfitting 심할 때**: $p \uparrow$, underfitting이면 $p \downarrow$.

### 언제 Dropout을 안 쓰는가

- **BatchNorm 있는 CNN**: BN이 이미 regularization 역할, Dropout 중복 효과 제한적 (Ioffe 2015).
- **작은 데이터 + RNN**: variational dropout (Ch2-04) 권장.
- **Scale이 매우 큰 모델 (GPT-3)**: data와 compute scale이 dropout 효과를 대체.

### MC Dropout for Uncertainty

Test 때도 `model.train()`으로 Dropout 활성 → 여러 forward pass → predictive mean/variance로 **epistemic uncertainty** 추정. 이는 Ch2-02(VI 해석)의 근거.

```python
def mc_dropout_predict(model, x, T=100):
    model.train()  # keep dropout active
    with torch.no_grad():
        preds = torch.stack([torch.softmax(model(x), -1) for _ in range(T)])
    return preds.mean(0), preds.std(0)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Weight scaling = ensemble mean | 비선형 + multi-layer에서 근사에 불과, 큰 $p$에서는 오차 커짐 |
| Bernoulli mask 독립 | Spatial Dropout, Variational RNN Dropout은 이 가정 수정 (Ch2-04) |
| Test time Dropout 비활성 | MC Dropout에서는 의도적으로 계속 활성 (다른 목적) |
| 단일 $p$ | 실전에서는 layer별 다른 $p$가 좋을 수 있음 (Concrete Dropout, Gal 2017) |
| 앙상블 해석 | Gal 2016 VI 해석과 공존 — 두 해석이 각각 다른 유효 범위 (Ch2-02) |

**주의**: Dropout이 "그냥 noise 주입 → regularization"이라는 서술은 정확하지 않다. Noise를 같은 크기로 Gaussian 주입하는 것은 Dropout과 **다른** 효과 (Wager 2013의 adaptive L2와도 다름). Dropout의 **multiplicative Bernoulli** 성질이 특별한 이유는 Ch2-03에서 본격 논의.

---

## 📌 핵심 정리

$$\boxed{\hat{y}_{\text{ens}} = \mathbb{E}_m[f(x; m \odot W)] \approx f(x; (1-p)W) \quad (\text{linear 정확, nonlinear 근사})}$$

| 개념 | 의미 |
|------|------|
| **Dropout** | 훈련 중 Bernoulli mask로 뉴런 랜덤 drop |
| **Thinned network** | 한 mask가 정의하는 서브네트워크, 총 $2^N$개 |
| **Weight scaling** | test 때 $(1-p) W$ — geometric mean 앙상블 근사 |
| **Inverted dropout** | 훈련 때 $1/(1-p)$로 scale (PyTorch 표준) |
| **다음 질문** | 앙상블 말고 VI로 해석할 수 있는가? → Ch2-02 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Dropout $p = 0.5$에서 $N = 100$개 뉴런이면 총 몇 개의 thinned subnetwork가 있는가? $N = 1000$이면?

<details>
<summary>힌트 및 해설</summary>

$2^{100} \approx 1.27 \times 10^{30}$. $N = 1000$이면 $2^{1000} \approx 10^{301}$. 우주의 원자 수($10^{80}$)를 초과. **모두를 명시적으로 평균내는 것은 불가능** — 그래서 weight scaling 근사가 필수. 앙상블 해석의 이론적 의미와 실용적 근사 둘 다 중요.

</details>

**문제 2** (심화): Linear model에서 Dropout + L2 regularization이 linear Dropout 하나만 쓸 때와 왜 다른지, 최종 loss function을 비교해 서술하라. Ch2-03의 **Wager 2013** 결과를 미리 예견해보라.

<details>
<summary>힌트 및 해설</summary>

$y = X w + \varepsilon$ (noise 없음으로 단순화). Dropout loss $\mathbb{E}_m[(y - X(m\odot w))^2]$:

$= \mathbb{E}_m[y^T y - 2 y^T X (m\odot w) + (m\odot w)^T X^T X (m\odot w)]$

각 항:
- 2nd: $\mathbb{E}[m_i] = (1-p)$, 따라서 $-2(1-p) y^T X w$.
- 3rd: $\mathbb{E}[m_i m_j] = (1-p)^2$ for $i \neq j$, $(1-p)$ for $i = j$. 분해:

$\sum_{i,j} \mathbb{E}[m_i m_j] w_i w_j (X^T X)_{ij} = (1-p)^2 w^T X^T X w + p(1-p) \sum_i w_i^2 (X^T X)_{ii}$

결국 Dropout 기댓값 loss는 **scaled squared loss + feature별 L2** 형태:

$\mathbb{E}_m[\text{loss}] = \|y - (1-p) X w\|^2 + p(1-p) w^T \text{diag}(X^TX) w$

두 번째 항이 **adaptive L2** — feature scale에 의존하는 weight별 regularization. 일반 L2($\lambda\|w\|^2$)와 다름. 이것이 Ch2-03의 핵심 결과를 미리 보는 것. 여기 L2를 **추가**로 더하면 해당 항이 커지지만 본질적으로 같은 구조 유지.

</details>

**문제 3** (이론-실전): Dropout의 "weight scaling $(1-p)$"과 BN의 "test time running statistics"는 모두 "training/test mode 차이"를 갖는 기법이다. 두 기법의 train/test 일관성을 수학적으로 비교하라.

<details>
<summary>힌트 및 해설</summary>

- **Dropout**: train 때 $m\odot h$ 샘플링 → test에는 $(1-p)h$. 근사적 기댓값 = 앙상블 geometric mean.
- **BN**: train 때 mini-batch $\mu_B, \sigma_B$ → test에는 running stats $\hat{\mu}, \hat{\sigma}$. 근사적 기댓값 = population 정규화.

**공통점**: "**stochastic train → deterministic test**". 훈련 중 noise로 regularize하지만 inference는 deterministic.

**차이**:
- Dropout은 **뉴런 independent**, BN은 **batch dependent** (sample 간 정보 공유).
- Dropout은 scale 조정이 간단(mul by $(1-p)$), BN은 EMA 수집 필요.
- Dropout은 train/test의 output이 오차 있지만 BN은 population stats 정확하면 무오차.

두 기법이 함께 쓰일 때 서로의 통계량을 교란할 수 있어 (Li 2019: "Disharmony between Dropout and BN") Transformer에서 LN + Dropout 조합이 더 자주 쓰이는 이유.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Chapter 1 → 05. Elastic Net](../ch1-l1-l2/05-elastic-net-group-lasso.md) | [📚 README로 돌아가기](../README.md) | [02. Dropout = VI ▶](./02-dropout-as-vi.md) |

</div>
