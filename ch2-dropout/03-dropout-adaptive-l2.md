# 03. Dropout = Adaptive L2 (Wager et al. 2013)

## 🎯 핵심 질문

- Linear regression + Bernoulli dropout의 **기댓값 loss**는 정확히 어떤 형태로 분해되는가?
- 이 분해에서 등장하는 **$\Gamma = p(1-p) \text{diag}(X^T X)$** 항의 의미는?
- 왜 이것이 **feature-scale에 적응적인 L2**인가?
- Feature가 표준화된 경우에만 단일 $\lambda$로 환원되는 이유는?

---

## 🔍 왜 이 세 번째 해석이 필요한가

Ch2-01(앙상블)과 Ch2-02(VI)는 Dropout의 **추상적** 해석을 주지만, "이 regularization이 파라미터 수준에서 **정확히 무엇을 하는가**"는 여전히 불투명했다.

Wager, Wang, Liang 2013은 **선형 모델에서만** Dropout의 기댓값 loss를 **closed form**으로 계산한다. 결과:

$$\mathbb{E}_m[\text{loss}] = \underbrace{\|y - (1-p) X w\|^2}_{\text{rescaled MSE}} + \underbrace{p(1-p) \cdot w^T \text{diag}(X^T X) w}_{\text{adaptive L2}}$$

두 번째 항이 **feature $j$별로 $\lambda_j \propto \|X_j\|^2$의 L2 regularization**. 이는 표준 L2($\lambda \|w\|^2$)와 **정량적으로 다르다**:

- 표준 L2: 모든 feature에 같은 penalty.
- Dropout L2: feature가 **크게 나타날수록** 더 강한 penalty — feature 간 스케일 불균형을 **자동 보정**.

이 결과의 세 가지 함의:

1. Dropout을 쓸 때 **feature 정규화가 덜 중요** — L2와 달리.
2. Adam + dropout과 SGD + dropout이 다르게 행동하는 이유의 일부 (L2 유효성이 다름).
3. 이 이론은 **linear**에서만 엄밀 — nonlinear NN에서는 정성적 직관에 그침.

---

## 📐 수학적 선행 조건

- Ch2-01: Bernoulli mask, $\mathbb{E}[m_i] = 1-p, \mathbb{E}[m_i m_j] = (1-p)^2 (i \neq j), (1-p) (i = j)$
- Ch1-01: L2 regularization의 MAP 해석
- 선형대수: $X^T X$의 diagonal, trace 성질
- 통계: Squared loss의 기댓값 계산

---

## 📖 직관적 이해

### 왜 "적응적"인가

Feature $X_j$의 scale이 크다는 것은 "그 feature가 예측에 큰 기여" → 해당 weight가 작아도 큰 효과. Dropout은 이런 강력한 feature를 랜덤하게 제거하므로 모델이 그 feature에 **지나치게 의존하면 안 된다**. 결과적으로 해당 weight에 **더 큰 penalty**.

수식으로: $\lambda_j = p(1-p) \|X_j\|^2$. Feature scale $\|X_j\|$ ↑ → $\lambda_j$ ↑.

### L2와의 차이

| 기법 | Penalty | Feature scale 효과 |
|------|---------|----------|
| L2 | $\lambda \sum_j w_j^2$ | 모든 weight 동일 |
| Dropout (기댓값) | $\sum_j p(1-p)\|X_j\|^2 w_j^2$ | Feature scale에 비례 |

**같은 효과 얻으려면**: L2를 쓸 때 feature를 먼저 표준화 ($\|X_j\|^2 = n$ 공통). 그러면 Dropout과 L2가 단일 $\lambda = np(1-p)$로 환원.

### "Feature drop의 확률적 회귀"라는 또 다른 비유

Dropout feature $X_j$를 확률 $p$로 삭제 → model이 이 feature를 **optionally** 사용하는 것을 학습. 이는 feature-level **bagging** 혹은 "random subspace 방법"과 유사. L2의 "모든 방향을 약간 축소"와는 정성적으로 다른 **특정 방향의 랜덤 drop**.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Linear Regression with Input Dropout

데이터 $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$. Input feature에 Dropout:

$$\tilde{X}_{ij} = m_{ij} X_{ij}, \quad m_{ij} \stackrel{\text{iid}}{\sim} \text{Bernoulli}(1-p)$$

(각 sample, 각 feature별 독립 mask — 혹은 sample별로 공유된 column-wise mask.) Inverted dropout으로 scale을 유지한다면 $\tilde{X}_{ij} = m_{ij} X_{ij} / (1-p)$.

### 정리 3.2 — Expected Loss Decomposition (Wager 2013, 주 정리)

Training sample별 Bernoulli mask $m_i \in \{0,1\}^d$ 하에 loss $L_m(w) = \|y - \tilde{X}w\|^2$의 **기댓값**:

$$\boxed{\mathbb{E}_m[L_m(w)] = \|y - (1-p)Xw\|^2 + p(1-p) w^T \, \text{diag}(X^TX) \, w}$$

(Mask가 $(1-p)$로 scale되지 않은, 즉 standard dropout의 경우.)

**Inverted dropout** ($\tilde{X}/(1-p)$)의 경우:

$$\mathbb{E}_m[L_m(w)] = \|y - Xw\|^2 + \frac{p}{1-p} w^T \text{diag}(X^TX) w$$

### 정리 3.3 — Equivalent Adaptive L2

정리 3.2는 목적함수가 **Ridge 변형**임을 보인다:

$$\min_w \|y - Xw\|^2 + w^T \Gamma w, \quad \Gamma = \frac{p}{1-p} \text{diag}(X^T X)$$

Closed-form 해:

$$\hat{w}_{\text{drop}} = (X^TX + \Gamma)^{-1} X^T y = \left(X^TX + \frac{p}{1-p}\text{diag}(X^TX)\right)^{-1}X^T y$$

### 정리 3.4 — Standardized Features Case

Feature를 표준화하여 $\|X_j\|^2 = n$ ($\forall j$)이면:

$$\Gamma = \frac{np}{1-p} I$$

즉 **일반 Ridge**와 정확히 동치, $\lambda_{\text{equiv}} = np/(1-p)$. Feature normalization이 없으면 각 feature별로 다른 $\lambda_j$가 작용.

### 정리 3.5 — Taylor 확장으로 일반 loss 확장

Wager 2013은 **smooth convex loss** $\ell(y, f(x; w))$ 일반화:

$$\mathbb{E}_m[\ell] \approx \ell(y, f) + \frac{p(1-p)}{2} \nabla_w^2 \ell \cdot \text{diag}(x x^T) + O(p^2)$$

$\nabla_w^2 \ell$이 Hessian. Adaptive L2 구조가 **generalized linear model (GLM)**에도 확장됨을 보인다.

---

## 🔬 수학적 유도

### 정리 3.2 증명

Loss를 전개:

$$L_m(w) = \sum_i (y_i - \tilde{X}_i^T w)^2 = \sum_i y_i^2 - 2\sum_i y_i \tilde{X}_i^T w + \sum_i (\tilde{X}_i^T w)^2$$

**기댓값** 각 항별로:

1st: $\sum y_i^2$ ($w$ 독립, const).

2nd: $\mathbb{E}_m[\sum y_i \tilde{X}_i^T w] = \sum_i y_i \sum_j w_j \mathbb{E}[m_{ij}] X_{ij} = (1-p)\sum_i y_i X_i^T w = (1-p) y^T X w$.

3rd: $\mathbb{E}_m[(\tilde{X}_i^T w)^2] = \mathbb{E}_m[\sum_{j,k} m_{ij} m_{ik} X_{ij} X_{ik} w_j w_k]$.

$\mathbb{E}[m_{ij} m_{ik}] = \begin{cases} (1-p)^2 & j \neq k \\ (1-p) & j = k \end{cases}$. 분해:

$$= (1-p)^2 \sum_{j\neq k} X_{ij} X_{ik} w_j w_k + (1-p) \sum_j X_{ij}^2 w_j^2$$

$= (1-p)^2 (X_i^T w)^2 + (1-p)[1 - (1-p)] \sum_j X_{ij}^2 w_j^2$

$= (1-p)^2 (X_i^T w)^2 + p(1-p) \sum_j X_{ij}^2 w_j^2$

$\sum_i$하면 3rd term:

$$(1-p)^2 \|Xw\|^2 + p(1-p) w^T \text{diag}(X^TX) w$$

합치면:

$$\mathbb{E}[L_m] = \|y\|^2 - 2(1-p) y^T X w + (1-p)^2 \|Xw\|^2 + p(1-p) w^T \text{diag}(X^TX) w$$

첫 세 항은 $\|y - (1-p) Xw\|^2$. $\square$

### Inverted dropout case

$\tilde X/(1-p)$로 scale. 2nd 항: $(1-p) / (1-p) = 1$ → $-2 y^T X w$. 3rd 항: $1/(1-p)^2$로 나누면 $\|Xw\|^2 + p(1-p)/(1-p)^2 \cdot w^T\text{diag}(X^TX)w = \|Xw\|^2 + \frac{p}{1-p} w^T \text{diag}(X^TX) w$. 따라서:

$$\mathbb{E}[L_m^{\text{inv}}] = \|y - Xw\|^2 + \frac{p}{1-p} w^T \text{diag}(X^TX) w \quad \square$$

### 정리 3.4 증명

$\|X_j\|^2 = \sum_i X_{ij}^2 = n$ (표준화). 그러면 $\text{diag}(X^TX) = n \cdot I$, $\Gamma = \frac{p}{1-p} n I$. $\hat{w}_{\text{drop}} = (X^T X + \frac{np}{1-p} I)^{-1}X^T y$ = Ridge $\lambda = np/(1-p)$. $\square$

---

## 💻 실험으로 효과 검증

### 실험 1 — 기댓값 동치 확인 (Monte Carlo)

```python
import numpy as np

np.random.seed(0)
n, d = 200, 10
X = np.random.randn(n, d)
w_true = np.random.randn(d)
y = X @ w_true + 0.1 * np.random.randn(n)

p = 0.2
w = np.random.randn(d)  # 테스트용 weight

# Empirical E[loss] via MC
T = 5000
losses = []
for _ in range(T):
    m = (np.random.rand(n, d) > p).astype(float)   # Bernoulli(1-p)
    X_tilde = m * X
    losses.append(np.sum((y - X_tilde @ w)**2))
mc_mean = np.mean(losses)

# Analytical prediction (정리 3.2)
analytic = np.sum((y - (1-p) * X @ w)**2) + p*(1-p)* w @ np.diag(X.T @ X) @ w

print(f"MC E[loss]        : {mc_mean:.3f}")
print(f"Analytical value  : {analytic:.3f}")
print(f"Relative error    : {abs(mc_mean - analytic)/analytic:.4f}")
# → 0.1% 이내 일치
```

### 실험 2 — Dropout 훈련 = Adaptive Ridge 훈련 확인

```python
from sklearn.linear_model import Ridge

def dropout_train(X, y, p, epochs=500, lr=0.02):
    d = X.shape[1]
    w = np.zeros(d)
    for _ in range(epochs):
        m = (np.random.rand(*X.shape) > p).astype(float)
        X_t = m * X / (1 - p)    # inverted dropout
        grad = -2 * X_t.T @ (y - X_t @ w) / X.shape[0]
        w -= lr * grad
    return w

# Dropout 훈련
np.random.seed(1)
w_drop = dropout_train(X, y, p=0.3, epochs=3000)

# Adaptive Ridge 직접 풀기 (Γ = p/(1-p) · diag(X^TX))
Gamma = 0.3/0.7 * np.diag(np.diag(X.T @ X))
w_adaptive = np.linalg.solve(X.T @ X + Gamma, X.T @ y)

# 표준 Ridge (scaled λ)
lam_equiv = 0.3/0.7 * np.diag(X.T @ X).mean()
w_ridge_eq = np.linalg.solve(X.T @ X + lam_equiv * np.eye(X.shape[1]), X.T @ y)

print("w_true      :", np.round(w_true, 3))
print("w_dropout   :", np.round(w_drop, 3))
print("w_adaptive  :", np.round(w_adaptive, 3))
print("w_ridge_eq  :", np.round(w_ridge_eq, 3))
print()
print("cos(w_drop, w_adaptive):", w_drop @ w_adaptive / (np.linalg.norm(w_drop)*np.linalg.norm(w_adaptive)))
# → w_drop ≈ w_adaptive (Wager 정리 검증), w_ridge_eq는 살짝 다름
```

### 실험 3 — Feature scale 불균형 시나리오

```python
# feature 스케일이 극단적으로 다른 경우 → adaptive L2가 자동 보정
X_imbal = X.copy()
X_imbal[:, 0] *= 10      # feature 0만 10배 크게
X_imbal[:, 5] *= 0.1     # feature 5는 1/10 크기

y_imbal = X_imbal @ w_true + 0.1 * np.random.randn(n)

w_drop_imb = dropout_train(X_imbal, y_imbal, p=0.3, epochs=3000)

# 표준 Ridge — 단일 λ는 큰 feature를 과잉 억제
lam = 1.0
w_ridge_imb = np.linalg.solve(X_imbal.T @ X_imbal + lam * np.eye(X_imbal.shape[1]),
                              X_imbal.T @ y_imbal)

print("Dropout        :", np.round(w_drop_imb, 3))
print("Standard Ridge :", np.round(w_ridge_imb, 3))
print("True w         :", np.round(w_true, 3))
# → Dropout은 스케일 변동에 거의 영향 없이 복원, Ridge는 왜곡됨
```

### 실험 4 — $p$에 따른 effective $\lambda$

```python
import matplotlib.pyplot as plt

ps = np.linspace(0.01, 0.7, 50)
lambda_effective = ps / (1 - ps) * np.diag(X.T @ X).mean()
plt.plot(ps, lambda_effective)
plt.xlabel('dropout p'); plt.ylabel('equivalent Ridge λ')
plt.title(r'Dropout 등가 Ridge 강도 $\lambda = \frac{p}{1-p} \bar{\|X_j\|^2}$')
plt.grid(alpha=0.3); plt.show()
# → p=0.5에서 λ = diag(X^TX) 평균 (가장 흔한 설정)
```

---

## 🔗 실전 활용

### 실전적 함의

1. **Feature normalization이 덜 critical**: Dropout 있는 네트워크는 weight decay만 있는 경우보다 feature scale에 덜 민감. 그래서 Transformer나 modern NN에서 Dropout이 추가 안전판.
2. **Inverted Dropout의 $\lambda_{\text{eff}} = p/(1-p)$ 의미**: $p = 0.5$에서 $p/(1-p) = 1$ 즉 "feature별 L2 penalty = feature variance". 매우 강한 regularization. 그래서 $p = 0.5$는 hidden layer 공격적 설정.
3. **Non-linear NN에서의 확장**: Wager의 결과는 정리 3.5로 generalized linear model까지. 깊은 NN에서는 **여전히 intuition**으로만: "Dropout = 각 layer의 feature별 adaptive L2".

### 표준 L2와의 선택

- **거의 표준화된 feature**: 두 기법 거의 동치. 선호에 따라.
- **크게 scaled feature**: Dropout이 자동 보정.
- **희귀 feature**: 둘 다 힘들 수 있음 — Group Lasso 고려.
- **Adam + L2**: Ch7-03에서 본 AdamW 이슈 — Adam에 단순 weight decay 주면 Wager 효과도 왜곡.

### 딥러닝 실전

현대 Transformer나 CNN에서:
- Dropout ≈ 0.1 (Attention, FFN에서)
- Weight decay ≈ 1e-4 (Adam과 AdamW)

두 가지가 **독립**적으로 작용하는 것이 아니라 **보완**. 정리 3.4의 "Dropout이 adaptive L2"이기 때문에 명시적 L2는 표준화된 weight의 "평균적인 penalty"를 더한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear model | Nonlinear NN에서는 정리 3.2가 엄밀히 성립하지 않음 |
| Bernoulli iid mask | RNN/sequence에서는 mask의 구조 다름 (Ch2-04 variational) |
| Input에 dropout | Hidden에 drop 시 해석은 훨씬 복잡 (feature가 학습된 것이므로) |
| Gaussian loss | 다른 loss에서는 Taylor 근사 (정리 3.5)로만 가능 |
| Standard MC estimator | Cross-data 구조 있으면 empirical loss와 기댓값 차이 생김 |

**주의**: "Dropout = adaptive L2"가 **Ch2-01의 앙상블 해석과 Ch2-02의 VI 해석을 replace**하지 않는다. 세 해석은 **같은 알고리즘의 다른 측면**. 각각 linear(Wager), deterministic ensemble(Srivastava), Bayesian approximate posterior(Gal).

---

## 📌 핵심 정리

$$\boxed{\mathbb{E}_m[\|y - \tilde X w\|^2] = \|y - (1-p)Xw\|^2 + p(1-p) \cdot w^T \text{diag}(X^TX) w}$$

| 개념 | 의미 |
|------|------|
| **Adaptive L2** | feature별 $\lambda_j = p(1-p)\|X_j\|^2$ |
| **Standardized features** | $\|X_j\|^2 = n$일 때 통상 Ridge로 환원 |
| **Feature-scale robustness** | Dropout은 scale 불균형에 자동 적응 |
| **Nonlinear 확장** | Taylor 근사로만 (정리 3.5) |
| **세 해석 공존** | Ensemble (Ch2-01) + VI (Ch2-02) + Adaptive L2 (이 문서) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $p = 0.5$, $\|X_j\|^2 = n = 100$인 feature에서 adaptive L2 penalty의 계수는?

<details>
<summary>힌트 및 해설</summary>

Standard dropout: $\lambda_j = p(1-p)\|X_j\|^2 = 0.25 \times 100 = 25$.

Inverted dropout: $\lambda_j = p/(1-p) \cdot \|X_j\|^2 = 1 \times 100 = 100$.

큰 coefficient에 놀라지 말 것. 목적함수는 $\|y - Xw\|^2 + 100 \cdot w_j^2$로 평균 loss 대비 큰 값이다 — 이게 $p = 0.5$ dropout의 공격성의 원인. 평균 loss 사용하면 $\lambda_j = 1$로 normal scale.

</details>

**문제 2** (심화): Input에 독립 Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$를 주면 어떤 regularization이 되는가? Dropout의 adaptive L2와 비교하라.

<details>
<summary>힌트 및 해설</summary>

$\tilde{X} = X + \varepsilon$ ($\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$). $\mathbb{E}_\varepsilon[\|y - \tilde X w\|^2]$:

- $\mathbb{E}[\varepsilon] = 0$이므로 2nd term은 $-2y^T X w$ 그대로.
- $\mathbb{E}[\|\tilde X w\|^2] = \|Xw\|^2 + n\sigma^2 \|w\|^2$ (각 sample별 독립 noise).

따라서:

$$\mathbb{E}[L] = \|y - Xw\|^2 + n\sigma^2 \|w\|^2$$

즉 **표준 Ridge** $\lambda = n\sigma^2$ — feature scale에 독립!

**Dropout과의 차이**: 
- Gaussian noise input = **uniform L2** (모든 feature 같은 penalty).
- Dropout input = **feature-scale 적응 L2** (큰 feature일수록 강한 penalty).

이것이 왜 Dropout이 Gaussian noise injection과 **다른** 효과를 내는지의 이유. Feature scale이 중요한 real data에서 Dropout이 더 적절한 bias.

</details>

**문제 3** (이론-실전): 깊은 NN에서 "Wager의 adaptive L2 해석"을 직접 적용하려면 어떤 복잡성이 추가되는가? Hidden dropout의 경우 각 layer의 adaptive L2는 무엇에 적응하는가?

<details>
<summary>힌트 및 해설</summary>

Hidden layer $h = \sigma(W_1 x + b_1)$에 dropout: $\tilde{h} = m \odot h$. 다음 layer $y = W_2 \tilde{h}$. 기댓값 loss는 $W_2$의 각 column에 대한 adaptive L2 구조를 주지만:

1. **$\|h_j\|^2$ 자체가 학습된 값** — training 중 변함. 따라서 $\lambda_j(\text{layer 2})$가 **layer 1의 학습과 상호작용**.
2. **Hidden representation의 correlation**도 중요. 정리 3.2의 diagonal $\text{diag}(X^TX)$ 대신 **full $X^T X$**의 일부가 들어옴.
3. 여러 layer에 dropout이 있으면 **compositional** effect가 Taylor 전개로만 근사.

실전 교훈:
- Hidden dropout은 "learned feature에 대한 adaptive L2"로 정성적 이해.
- 각 layer의 activation magnitude가 dropout의 effective strength를 결정.
- **LayerNorm + Dropout**이 안정적인 이유: LN이 activation scale을 통일하면 dropout이 근사적으로 uniform L2처럼 작용.

Rigorous 확장 논문: Mianjy & Arora 2019 "On Dropout and Nuclear Norm Regularization" — dropout on 2-layer linear NN이 nuclear norm의 변형을 주는 것을 보임.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Dropout = VI](./02-dropout-as-vi.md) | [📚 README로 돌아가기](../README.md) | [04. Dropout 변종 ▶](./04-dropout-variants.md) |

</div>
