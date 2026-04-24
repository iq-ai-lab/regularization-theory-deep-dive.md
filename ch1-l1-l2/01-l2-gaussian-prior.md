# 01. L2 Regularization = Gaussian Prior MAP

## 🎯 핵심 질문

- $\min \|y - Xw\|^2 + \lambda \|w\|^2$의 $\lambda$는 Bayesian 관점에서 정확히 무엇에 대응하는가?
- 왜 Gaussian prior의 MAP이 L2 regularization과 "동치"가 되는가?
- $\lambda = \sigma^2/\sigma_w^2$ 대응 관계의 실전적 함의는 무엇인가?
- Bayesian linear regression의 **posterior mean**은 Ridge의 해와 어떤 관계인가?

---

## 🔍 왜 이 관점이 regularization 이해에 중요한가

실전에서는 **"$\lambda$를 grid search로 튜닝"** 한다. 그러나 이 관점은 $\lambda$를 **임의의 hyperparameter**로 취급해 왜 정확히 $\|w\|^2$ 형태인지, 왜 이차형인지, 왜 L1·L3·L4가 아닌지를 **우연**으로 남긴다.

**Bayesian 관점의 힘**은 이것이다: **regularization term은 prior의 negative log-density**이다. Gaussian prior를 쓰면 정확히 $\lambda \|w\|^2$가, Laplace prior를 쓰면 $\lambda \|w\|_1$이 나온다(다음 문서). 이 관점이 없으면:

- L2의 $\lambda$가 왜 **noise variance**와 연결되는지 설명 불가
- **weight decay**를 너무 크게 주면 왜 underfit하는지 정량화 불가
- Bayesian linear regression의 predictive uncertainty를 regularization 설정에서 유도 불가
- 나중에 **Dropout = VI** (Ch2-02), **SWA = SWAG** (Ch7-01)로 가는 Bayesian 다리가 끊어짐

이 문서는 이 레포 전체의 **"모든 regularization을 가능한 경우 prior로 해석"** 이라는 스타일 가이드의 출발점이다.

---

## 📐 수학적 선행 조건

- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): prior $p(w)$, likelihood $p(y|w)$, posterior $p(w|y) \propto p(y|w)p(w)$, MAP
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 다변수 Gaussian의 pdf $\mathcal{N}(\mu, \Sigma)$, log-density, Woodbury identity
- 선형대수: $X^T X$의 positive semidefiniteness, 역행렬 $(A + \lambda I)^{-1}$의 존재성 ($\lambda > 0$에서)
- 미적분 기초: 이차형의 gradient, 최적성 조건 $\nabla = 0$

---

## 📖 직관적 이해

### "Regularization은 사전 지식"

$\lambda \|w\|^2$을 추가하는 것은 "데이터가 강하게 반박하지 않는 한, 나는 $w$가 **0 근처**에 있다고 **미리** 믿는다"는 진술. 이 믿음을 확률분포로 쓰면:

$$p(w) = \mathcal{N}(0, \sigma_w^2 I)$$

즉 **평균 0, 분산 $\sigma_w^2 I$의 Gaussian**. $\sigma_w^2$이 작을수록 믿음이 강하다 (0 근처에 더 강한 mass).

### Likelihood = 데이터에 대한 모델

회귀 모델 $y = Xw + \varepsilon, \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$이면:

$$p(y \mid w) = \mathcal{N}(Xw, \sigma^2 I)$$

### Bayes rule로 posterior

$$p(w \mid y) \propto p(y \mid w) \, p(w)$$

**MAP (Maximum A Posteriori)** 추정자는 posterior의 mode(최대값):

$$\hat{w}_{\text{MAP}} = \arg\max_w p(w \mid y) = \arg\min_w \big[ -\log p(y \mid w) - \log p(w) \big]$$

두 negative log term을 각각 계산하면 **정확히** L2 regularized loss가 나온다.

### 핵심 대응표

| Regularization 언어 | Bayesian 언어 |
|-------|------|
| $\lambda \|w\|^2$ | Gaussian prior $\mathcal{N}(0, \sigma_w^2 I)$의 negative log |
| $\lambda$ 크다 | prior가 강하다 ($\sigma_w^2$ 작다) |
| $\lambda = 0$ | flat prior (= MLE) |
| $\lambda \to \infty$ | $w = 0$ 고집 (degenerate) |
| Hyperparameter 튜닝 | Prior hyperparameter 튜닝 (empirical Bayes) |

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Bayesian Linear Regression Model

확률 모델 $\mathcal{M}_\text{BLR}$:

$$w \sim \mathcal{N}(0, \sigma_w^2 I_d), \quad y \mid w, X \sim \mathcal{N}(Xw, \sigma^2 I_n)$$

여기서 $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, $w \in \mathbb{R}^d$는 모델 파라미터이며 $\sigma^2, \sigma_w^2 > 0$은 hyperparameter.

### 정의 1.2 — MAP Estimator

Posterior $p(w \mid y, X) = p(y \mid w, X) p(w) / p(y \mid X)$에 대해:

$$\hat{w}_{\text{MAP}} := \arg\max_w p(w \mid y, X) = \arg\max_w \big[\log p(y \mid w, X) + \log p(w)\big]$$

Normalizing constant $p(y|X)$는 $w$ 독립이므로 argmax에서 사라진다.

### 정리 1.3 — L2 = Gaussian Prior MAP (주 정리)

정의 1.1의 $\mathcal{M}_\text{BLR}$에서:

$$\boxed{\hat{w}_{\text{MAP}} = \arg\min_w \Big[ \tfrac{1}{2\sigma^2} \|y - Xw\|^2 + \tfrac{1}{2\sigma_w^2} \|w\|^2 \Big]}$$

이는 **Ridge regression의 목적함수**와 정확히 동치이며, 대응 관계는:

$$\boxed{\lambda_{\text{Ridge}} = \frac{\sigma^2}{\sigma_w^2}}$$

단, 정의에 따라 $\sigma^2$로 나눠 표준화하면 $\lambda \|w\|^2$ 형태에서 $\lambda = \sigma^2/\sigma_w^2$. $n$으로 나누는 통상적 평균 loss $\frac{1}{n}\|y-Xw\|^2 + \lambda\|w\|^2$ convention에서는 $\lambda = \sigma^2/(n \sigma_w^2)$.

### 정리 1.4 — Closed-form Solution

$$\hat{w}_{\text{MAP}} = (X^T X + \lambda I)^{-1} X^T y$$

여기서 $\lambda = \sigma^2/\sigma_w^2$. 이는 Gaussian posterior의 **mean과도 같다** (Gaussian에서 mode = mean).

### 정리 1.5 — Posterior의 정확한 형태

$$p(w \mid y, X) = \mathcal{N}(\mu_w, \Sigma_w), \quad \mu_w = \hat{w}_{\text{MAP}}, \quad \Sigma_w = \sigma^2 (X^T X + \lambda I)^{-1}$$

즉 Bayesian linear regression은 Gaussian posterior를 갖고, 그 **mean = Ridge 해**이다.

---

## 🔬 수학적 유도

### Step 1 — Likelihood의 negative log

$$p(y \mid w) = \frac{1}{(2\pi\sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} \|y - Xw\|^2\right)$$

$$-\log p(y \mid w) = \frac{1}{2\sigma^2} \|y - Xw\|^2 + \underbrace{\frac{n}{2}\log(2\pi\sigma^2)}_{\text{w-독립}}$$

### Step 2 — Prior의 negative log

$$p(w) = \frac{1}{(2\pi\sigma_w^2)^{d/2}} \exp\left(-\frac{1}{2\sigma_w^2} \|w\|^2\right)$$

$$-\log p(w) = \frac{1}{2\sigma_w^2} \|w\|^2 + \underbrace{\frac{d}{2}\log(2\pi\sigma_w^2)}_{\text{w-독립}}$$

### Step 3 — MAP 목적함수 합성

$w$-독립 상수 제거 후:

$$\hat{w}_{\text{MAP}} = \arg\min_w \frac{1}{2\sigma^2} \|y - Xw\|^2 + \frac{1}{2\sigma_w^2} \|w\|^2$$

양변에 $2\sigma^2$ 곱:

$$= \arg\min_w \|y - Xw\|^2 + \underbrace{\frac{\sigma^2}{\sigma_w^2}}_{= \lambda} \|w\|^2 \quad \square$$

### Step 4 — Closed-form Solution 유도

목적함수를 $J(w)$로 쓰면:

$$J(w) = \frac{1}{2\sigma^2}(y - Xw)^T(y - Xw) + \frac{1}{2\sigma_w^2} w^T w$$

$w$에 대한 gradient:

$$\nabla_w J = -\frac{1}{\sigma^2} X^T(y - Xw) + \frac{1}{\sigma_w^2} w = 0$$

$\sigma^2$ 곱:

$$-X^T y + X^T X w + \frac{\sigma^2}{\sigma_w^2} w = 0 \implies (X^T X + \lambda I) w = X^T y$$

$$\hat{w}_{\text{MAP}} = (X^T X + \lambda I)^{-1} X^T y \quad \square$$

**$\lambda > 0$이면** $X^T X + \lambda I \succ 0$이므로 역행렬 존재 (feature 수 $d > n$이어도 무방).

### Step 5 — Posterior Covariance 유도

Posterior $p(w|y) \propto \exp\big(-\frac{1}{2\sigma^2}\|y-Xw\|^2 - \frac{1}{2\sigma_w^2}\|w\|^2\big)$의 지수부를 $w$에 대한 이차형으로 완성하면:

$$-\frac{1}{2}\big(w^T \underbrace{(\tfrac{1}{\sigma^2}X^T X + \tfrac{1}{\sigma_w^2} I)}_{\Sigma_w^{-1}} w - 2 w^T \tfrac{1}{\sigma^2}X^T y\big) + \text{const}$$

이는 $w$가 평균 $\mu_w = \Sigma_w \cdot \tfrac{1}{\sigma^2} X^T y$, 공분산 $\Sigma_w = \sigma^2(X^T X + \lambda I)^{-1}$인 Gaussian임을 의미. $\mu_w = \hat{w}_{\text{MAP}}$임을 직접 확인 가능. $\square$

---

## 💻 실험으로 효과 검증

### 실험 1 — Ridge 해 vs MAP의 수치적 동일성

```python
import numpy as np

np.random.seed(0)
n, d = 50, 20
X = np.random.randn(n, d)
w_true = np.random.randn(d)
sigma = 0.3
y = X @ w_true + sigma * np.random.randn(n)

sigma_w = 1.0
lam = sigma**2 / sigma_w**2       # Bayesian 대응
print(f"lambda from Bayes: {lam:.4f}")

# (1) Ridge closed-form
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)

# (2) Bayesian linear regression posterior mean
Sigma_post = sigma**2 * np.linalg.inv(X.T @ X + lam * np.eye(d))
mu_post = Sigma_post @ X.T @ y / sigma**2

print("max |w_ridge - mu_post| :", np.max(np.abs(w_ridge - mu_post)))
# → 0 (수치 오차 수준)
```

### 실험 2 — $\lambda$-$\sigma_w$ 쌍곡선 시각화

```python
import matplotlib.pyplot as plt

sigma2 = 0.09
sigma_w_list = np.logspace(-1, 1.5, 40)
lam_list = sigma2 / sigma_w_list**2

plt.loglog(sigma_w_list, lam_list)
plt.xlabel(r'$\sigma_w$  (prior strength, 작을수록 강함)')
plt.ylabel(r'$\lambda = \sigma^2/\sigma_w^2$')
plt.title('Bayesian prior 강도와 Ridge λ의 쌍곡선 관계')
plt.grid(True, which='both', alpha=0.3)
plt.show()
# → σ_w가 0에 가까울수록 λ가 폭증(prior 믿음 강함 → 강한 regularization)
```

### 실험 3 — $\lambda$ 스캔에서 coefficient path

```python
lams = np.logspace(-3, 2, 50)
paths = np.array([np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ y)
                  for lam in lams])

plt.figure(figsize=(9, 4))
for j in range(d):
    plt.semilogx(lams, paths[:, j], alpha=0.6)
plt.xlabel(r'$\lambda$ (log)'); plt.ylabel('Ridge coefficient')
plt.title('Ridge coefficient path — λ↑일수록 0에 smooth shrink (L1과 대비)')
plt.axhline(0, c='k', lw=0.5)
plt.grid(True, alpha=0.3); plt.show()
# → L1과 달리 "정확히 0"은 결코 달성되지 않음 (Ch1-02와의 대비)
```

---

## 🔗 실전 활용

### 언제 이 해석이 도움되는가

1. **$\lambda$ 초기값을 원리적으로 고르기**: noise 수준 $\sigma$를 cross-validation으로 추정하고, 사전 믿음으로 $\sigma_w$를 고르면 $\lambda$가 **자동**으로 결정된다.
2. **Weight decay의 scale 결정**: PyTorch `weight_decay=1e-4`는 batch 평균 loss에 대한 Ridge $\lambda$. 이를 $\sigma^2/\sigma_w^2$로 해석하면 noise가 작은 task(깨끗한 라벨)에서는 작은 $\lambda$가 자연스럽다.
3. **Empirical Bayes**: $\sigma^2, \sigma_w^2$을 marginal likelihood $p(y|X, \sigma, \sigma_w)$ 최대화로 추정 (automatic regularization).
4. **Predictive uncertainty**: Posterior covariance $\Sigma_w$로부터 predictive variance $\mathbb{V}[y^*|x^*] = \sigma^2 + x^{*T}\Sigma_w x^*$ 즉시 계산.

### 딥러닝에서의 한계와 확장

딥러닝에서 "$\lambda \|w\|^2 \Leftrightarrow$ Gaussian prior"는 **유효**하지만 MAP은 **posterior mode 하나만** 준다. Posterior의 **전체 분포**를 쓰려면 **Dropout = VI**(Ch2-02) 또는 **SWAG**(Ch7-01)로 확장.

또한 Adam 같은 adaptive optimizer에서 "L2 loss 추가 = weight decay"는 **틀리다** (Ch7-03 AdamW 참고).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ | heavy-tailed noise에서는 Student-t likelihood로 확장 필요 |
| 등방성 prior $\mathcal{N}(0, \sigma_w^2 I)$ | feature scale이 다르면 feature별 $\sigma_{w,j}^2$ 필요 (ARD) |
| MAP = posterior mode = mean | Gaussian에서만 동치. 비정규 posterior에서는 별도. |
| Linear model | NN에서는 $p(w)$ Gaussian이어도 $p(y|w)$가 nonlinear → posterior 복잡 |
| $\lambda$ 한 개 scalar | 실전에서는 layer별·param별 weight decay 필요할 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\hat{w}_{\text{Ridge}} = \arg\min \|y - Xw\|^2 + \lambda\|w\|^2 = \hat{w}_{\text{MAP}} \text{ under } \mathcal{N}(0, \sigma_w^2 I) \text{ prior, with } \lambda = \sigma^2/\sigma_w^2}$$

| 개념 | 의미 |
|------|------|
| **L2 regularization** | Gaussian prior의 negative log |
| **$\lambda$** | $\sigma^2/\sigma_w^2$ — noise-to-prior 분산비 |
| **MAP** | Posterior mode. Gaussian에서는 mean과 같음 |
| **Ridge solution** | $(X^TX + \lambda I)^{-1}X^T y$ — Bayesian posterior mean |
| **다음 질문** | L1은? Laplace prior로 확장 → Ch1-02 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\sigma = 0.1$ (noise 표준편차), $\sigma_w = 2$ (prior 표준편차)일 때 $\lambda$는? $\sigma_w$가 두 배가 되면 $\lambda$는 어떻게 변하는가?

<details>
<summary>힌트 및 해설</summary>

$\lambda = \sigma^2/\sigma_w^2 = 0.01/4 = 0.0025$. $\sigma_w$가 2배면 $\sigma_w^2$는 4배, 따라서 $\lambda$는 **1/4배**로 감소. 직관: prior가 약해지면(더 넓게 퍼짐) regularization도 약해진다.

</details>

**문제 2** (심화): $\lambda = 0$일 때와 $\lambda \to \infty$일 때 $\hat{w}_{\text{MAP}}$은 각각 무엇이 되는가? Gaussian prior의 극한 해석을 함께 서술하라.

<details>
<summary>힌트 및 해설</summary>

- $\lambda \to 0$: $\sigma_w^2 \to \infty$ (flat prior). MAP = MLE = $(X^TX)^{-1}X^T y$ (단, $X^T X$ invertible일 때). Prior가 "아무 정보 없다"고 말함.
- $\lambda \to \infty$: $\sigma_w^2 \to 0$ (delta prior at 0). MAP $\to 0$. Prior가 "무조건 $w=0$"이라 단언.

이 두 극한이 **Ridge path**의 양끝이다. 실전 $\lambda$는 그 사이에서 cross-validation으로 선택.

</details>

**문제 3** (이론-실전): Gaussian likelihood를 Laplace로, Gaussian prior를 유지하면 어떤 regularization이 나오는가? 또 반대로 likelihood는 Gaussian, prior는 Laplace이면?

<details>
<summary>힌트 및 해설</summary>

- Laplace likelihood $p(y|w) \propto \exp(-\|y - Xw\|_1/b)$ + Gaussian prior → 목적함수는 **$\|y - Xw\|_1 + \lambda\|w\|^2$** (robust regression). noise가 heavy-tail일 때 유용.
- Gaussian likelihood + Laplace prior → **$\|y - Xw\|^2 + \lambda\|w\|_1$** (Lasso, 다음 문서).

핵심 교훈: **loss 함수 ↔ likelihood**, **regularizer ↔ prior**. 데이터 noise 특성과 weight 사전 믿음을 **독립적으로** 고를 수 있다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [📚 README로 돌아가기](../README.md) | | [02. L1 = Laplace Prior MAP ▶](./02-l1-laplace-prior.md) |

</div>
