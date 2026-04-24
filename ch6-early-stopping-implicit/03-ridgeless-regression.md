# 03. Ridgeless Regression의 일반화 (Hastie 2019)

## 🎯 핵심 질문

- $p > n$ overparameterized linear model에서 OLS가 **unique**하지 않을 때, 어떤 해가 선택되는가?
- **Minimum-norm solution** $\hat{\beta} = X^+ y$의 asymptotic risk는?
- Ridgeless regression ($\lambda \to 0^+$)이 왜 도움이 되는가?
- Double Descent의 **peak at $p = n$** 와 어떻게 연결되는가?

---

## 🔍 왜 "Ridgeless" 가 중요한가

고전 통계학의 관점: $p > n$ (변수가 sample보다 많음) = "ill-posed problem" — 해가 무한. Ridge regularization ($\lambda > 0$) 필수.

**현대 딥러닝**: $p \gg n$에서도 잘 generalize. 왜?

Hastie, Montanari, Rosset, Tibshirani 2019 **"Surprises in High-Dimensional Ridgeless Least Squares Interpolation"**:

1. **$p > n$**에서 OLS는 $\infty$개 해 → **minimum-norm solution** $\hat{\beta} = X^+ y$ 자동 선택 (GD에서).
2. 이 min-norm solution의 asymptotic risk를 **정확히 계산** — 놀랍게도 finite.
3. Ridge의 $\lambda \to 0^+$ 극한과 동등 — "implicit regularization from initialization".
4. **Double Descent**의 수학적 기반 — peak at $p = n$의 원인.

이는 "**over-parameterization이 왜 작동하는가**"의 중요한 조각. Ch6-01 (Early Stopping = L2), Ch6-02 (SGD bias)에 이어 Implicit regularization 삼위일체 완성.

---

## 📐 수학적 선행 조건

- Ch1-04: Ridge의 SVD 관점
- Ch6-01: Early stopping = Ridge path
- [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive): bias-variance decomposition
- 선형대수: Moore-Penrose pseudoinverse, rank-deficient matrix
- 확률: Marchenko-Pastur distribution (random matrix theory)

---

## 📖 직관적 이해

### Over-parameterized OLS

$y = X\beta^* + \varepsilon$, $X \in \mathbb{R}^{n \times p}$, $p > n$.

OLS: $\arg\min \|y - X\beta\|^2$. $X$가 row rank $n$이면 **infinitely many $\beta$**가 $X\beta = y$ perfectly interpolate.

**Question**: 어떤 해를 선택?

### Minimum-Norm Solution

$$\hat{\beta}_{\min} = \arg\min \|\beta\|^2 \quad \text{s.t.} \quad X\beta = y$$

Closed form: $\hat{\beta}_{\min} = X^+ y = X^T(XX^T)^{-1} y$ (assume $X$ full row rank).

**Key property**:
- Interpolates: $X \hat{\beta}_{\min} = y$.
- Minimum $L^2$ norm among all interpolators.

### GD from 0 Converges to $\hat{\beta}_{\min}$

$p > n$ OLS에서 GD (from $\hat{\beta}_0 = 0$)가 수렴하는 해는 **정확히 $\hat{\beta}_{\min}$**.

**이유**: GD update는 $\beta_t \in \text{col}(X^T)$에 머무름 (gradient는 $-X^T(y - X\beta)$ form). $\text{col}(X^T)$ 위의 interpolating 해는 **unique** = min-norm.

### Ridgeless Risk의 "Surprising" Behavior

Ridge $\lambda \to 0^+$에서 risk:

$$R(\lambda \to 0^+) = \sigma^2 \cdot \psi(p/n) + \|\beta^*\|^2 \cdot \phi(p/n)$$

$\psi, \phi$는 random matrix theory로 정의되는 specific functions of $p/n$.

**놀라운 발견**:
- $p/n \to 1^-$: $\psi \to \infty$ (test MSE → $\infty$).
- $p/n \to 1^+$: $\psi$ 다시 감소.
- $p/n \to \infty$: 일부 task에서 **OLS보다 낮은 risk** ($p$ 증가가 도움!).

이것이 **Double Descent**. Peak at $p = n$ + descent at $p > n$.

### Why Does More Parameters Help?

**$p > n$에서 min-norm 해**는 자동으로 "**regularized**":
- Many interpolating options → min-norm이 특정 방향 선호.
- 그 방향이 "signal direction에 align" → bias 작음.
- Variance 증가하지만 bounded.

Specific한 "**implicit ridge**" 가 작용.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Minimum-Norm Interpolator

$$\hat{\beta}_{\min} = \arg\min \|\beta\|_2 \quad \text{s.t.} \quad X\beta = y$$

$= X^+ y$ (Moore-Penrose pseudoinverse).

### 정리 3.2 — GD from 0 → Min-Norm

Linear regression quadratic loss + GD from $\beta_0 = 0$:

$$\lim_{t \to \infty} \hat{\beta}_t = X^+ y = \hat{\beta}_{\min}$$

(Constrained to $p > n$ interpolating case.)

### 정의 3.3 — Asymptotic Regime

$n, p \to \infty$, $p/n \to \gamma \in (0, \infty)$. "Proportional asymptotics".

### 정리 3.4 — Hastie-Montanari-Rosset-Tibshirani 2019 (주 정리)

Isotropic features $x_i \sim \mathcal{N}(0, I_p)$, well-specified $y = X\beta^* + \varepsilon$ ($\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$):

**Limiting risk** of $\hat{\beta}_{\min}$ (as $n, p \to \infty, p/n \to \gamma$):

$$R(\gamma) = \|\beta^*\|^2 \cdot (1 - 1/\gamma) \cdot \mathbb{1}[\gamma > 1] + \sigma^2 \cdot \frac{\gamma}{|\gamma - 1|}$$

**Critical behavior**:
- $\gamma < 1$ (under-parameterized): $R = \sigma^2 / (1 - \gamma)$ — 고전 OLS variance.
- $\gamma = 1$: $R = \infty$ — **explosion**.
- $\gamma > 1$ (over-parameterized): $R = \|\beta^*\|^2 (1 - 1/\gamma) + \sigma^2 \gamma/(\gamma - 1)$.

### 정리 3.5 — $\lambda \to 0^+$ Ridge = Min-Norm

$$\lim_{\lambda \to 0^+} \hat{\beta}_R(\lambda) = \hat{\beta}_{\min}$$

**증명**: Ridge solution SVD form $\hat{\beta}_R = V\text{diag}(\sigma_i/(\sigma_i^2+\lambda))U^T y$. $\lambda \to 0^+$, $\sigma_i > 0$이면 $\sigma_i/(\sigma_i^2+0) = 1/\sigma_i$ → $\hat{\beta} = V\Sigma^{-1}U^T y = X^+ y$. $\square$

### 정리 3.6 — Implicit vs Explicit Regularization Trade-off

$\gamma > 1$ (over-parameterized) 에서, optimal Ridge $\lambda^* > 0$이 min-norm보다 낮은 risk:

$$R(\lambda^*) < R(\lambda = 0^+)$$

즉 **적당한 명시적 regularization이 implicit min-norm bias보다 좋을 수 있음**. 하지만 implicit만으로도 **reasonable** risk 달성.

---

## 🔬 수학적 유도

### 정리 3.4 Proof Sketch

**Bias-variance decomposition** (Ch1-04 정리 4.5 일반화):

$$R(\hat{\beta}) = \|\mathbb{E}[\hat\beta] - \beta^*\|^2 + \text{tr}(\text{Cov}(\hat\beta))$$

**Minimum-norm**: $\hat{\beta}_{\min} = X^T(XX^T)^{-1}y = X^T(XX^T)^{-1}(X\beta^* + \varepsilon)$.

$\mathbb{E}[\hat{\beta}_{\min}] = X^T(XX^T)^{-1}X\beta^* = P_{\text{row}(X)} \beta^*$ (projection onto row space of $X$).

**Bias**: $\|\beta^* - P_{\text{row}(X)}\beta^*\|^2 = \|P_{\text{row}(X)^\perp}\beta^*\|^2$.

Under $\beta^*$ isotropic, $\mathbb{E}[\|P_{\text{row}(X)^\perp}\beta^*\|^2] = \|\beta^*\|^2 \cdot (p - n)/p$ (row space는 $n$차원).

For $\gamma = p/n > 1$: Bias $= \|\beta^*\|^2(1 - n/p) = \|\beta^*\|^2(1 - 1/\gamma)$.

**Variance**: $\sigma^2 \text{tr}[X^T(XX^T)^{-1}(XX^T)^{-1}X] = \sigma^2 \text{tr}[(XX^T)^{-1}]$.

Random matrix theory: $\mathbb{E}[\text{tr}((XX^T)^{-1})]$의 limiting value via **Marchenko-Pastur distribution**.

Result: $\mathbb{E}[\text{tr}(n (XX^T)^{-1})] \to 1/(\gamma - 1)$ for $\gamma > 1$. Thus variance $= \sigma^2 \gamma/(\gamma - 1)$. $\square$

### $\gamma = 1$ Singularity 원인

$\gamma = 1$이면 $X$는 square matrix, **거의 singular** (smallest eigenvalue → 0).

$\text{tr}((XX^T)^{-1}) = \sum 1/\sigma_i^2 \to \infty$ as $\sigma_{\min} \to 0$.

따라서 variance blow-up — 이것이 Double Descent의 peak.

**실제 상황**: $\lambda \to 0^+$ with $\lambda > 0$으로 small regularization 유지하면 peak 완화.

---

## 💻 실험으로 효과 검증

### 실험 1 — Ridgeless Risk Curve 재현 (Hastie 2019 Fig 1)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 100
sigma = 1.0
beta_true_norm = 1.0

gammas = np.linspace(0.1, 3, 50)
risks = []

for gamma in gammas:
    p = int(gamma * n)
    beta = np.random.randn(p) * beta_true_norm / np.sqrt(p)
    
    # Multiple trials for expected risk
    trial_risks = []
    for _ in range(20):
        X = np.random.randn(n, p) / np.sqrt(n)
        y = X @ beta + sigma * np.random.randn(n)
        
        if p <= n:
            # OLS
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            # Min-norm pseudoinverse
            beta_hat = X.T @ np.linalg.inv(X @ X.T) @ y
        
        test_X = np.random.randn(1000, p) / np.sqrt(n)
        test_y = test_X @ beta + sigma * np.random.randn(1000)
        trial_risks.append(np.mean((test_y - test_X @ beta_hat)**2))
    
    risks.append(np.mean(trial_risks))

plt.figure(figsize=(10, 5))
plt.plot(gammas, risks, 'o-')
plt.axvline(1.0, ls='--', c='r', label='p/n = 1 (peak)')
plt.xlabel(r'$\gamma = p/n$'); plt.ylabel('Test MSE')
plt.title('Ridgeless regression risk — Double Descent')
plt.yscale('log')
plt.legend(); plt.grid(alpha=0.3); plt.show()
# → γ = 1에서 peak, γ → 0 and γ → ∞에서 finite risk
```

### 실험 2 — Theoretical formula vs simulation

```python
def theoretical_risk(gamma, beta_norm_sq=1.0, sigma_sq=1.0):
    if gamma < 1:
        return sigma_sq / (1 - gamma)
    elif gamma > 1:
        return beta_norm_sq * (1 - 1/gamma) + sigma_sq * gamma / (gamma - 1)
    else:
        return np.inf

theoretical = [theoretical_risk(g) for g in gammas]
plt.plot(gammas, risks, 'o', label='Simulation', alpha=0.6)
plt.plot(gammas, theoretical, '-', label='Theory (HMRT 2019)', lw=2)
plt.axvline(1.0, ls='--', c='r')
plt.yscale('log'); plt.legend(); plt.grid(alpha=0.3); plt.show()
# → 두 curve 거의 일치
```

### 실험 3 — Ridge regularization의 peak 완화

```python
lams = [0, 0.01, 0.1, 1.0]
gammas_fine = np.linspace(0.5, 2, 30)

fig, ax = plt.subplots(figsize=(10, 5))
for lam in lams:
    risks_lam = []
    for gamma in gammas_fine:
        p = int(gamma * n)
        beta = np.random.randn(p) / np.sqrt(p)
        trials = []
        for _ in range(10):
            X = np.random.randn(n, p) / np.sqrt(n)
            y = X @ beta + np.random.randn(n)
            beta_hat = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
            test_X = np.random.randn(500, p) / np.sqrt(n)
            test_y = test_X @ beta + np.random.randn(500)
            trials.append(np.mean((test_y - test_X @ beta_hat)**2))
        risks_lam.append(np.mean(trials))
    ax.plot(gammas_fine, risks_lam, 'o-', label=f'λ = {lam}', alpha=0.7)
ax.axvline(1.0, ls='--', c='r'); ax.set_yscale('log')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('Test MSE')
ax.set_title('Ridge λ가 peak를 완화')
ax.legend(); ax.grid(alpha=0.3); plt.show()
# → λ ≥ 0.1이면 peak 사라짐
```

### 실험 4 — GD converges to min-norm

```python
# GD from 0 확인
X = np.random.randn(50, 200) / np.sqrt(50)  # p=200 > n=50
y = np.random.randn(50)

# GD
lr = 0.01
w = np.zeros(200)
for _ in range(10000):
    w -= lr * (-X.T @ (y - X @ w))

# Min-norm pseudoinverse
w_minnorm = X.T @ np.linalg.inv(X @ X.T) @ y

print(f"||w_GD - w_minnorm||_2: {np.linalg.norm(w - w_minnorm):.6f}")
# → 매우 작음 (GD가 min-norm으로 수렴)
```

---

## 🔗 실전 활용

### Deep Learning에서의 Ridgeless Regime

**현대 LLM**: $p \gg n$. Min-norm 해가 자동으로 선택:
- GPT-3: $p \approx 175 \times 10^9$ vs $n \approx 300 \times 10^9$ tokens.
- 엄밀한 ridgeless regime은 아니지만, **over-parameterized**.

**Implicit regularization의 연속**:
- SGD trajectory가 "min-norm-like" solution 선호.
- Initialization scale이 implicit bias 결정.

### NTK Regime에서의 대응

무한 폭 NN의 NTK (Generalization Theory Deep Dive Ch3): kernel regression equivalent.

**Kernel regression에서도 min-norm**: Gradient descent가 "minimum RKHS norm" 해로 수렴 (Moore-Aronszajn).

같은 원리 — **training의 implicit regularization**이 generalization 가능하게.

### Double Descent와의 연결

Generalization Theory Deep Dive Ch4: Double Descent의 정확한 수학적 설명 이 문서의 정리 3.4. 

실전 딥러닝에서 double descent 자주 안 보이는 이유:
- 기본 weight decay, dropout 등 **implicit + explicit regularization**.
- Peak at $p = n$은 **특정 condition**에서만 sharp.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Isotropic features $x \sim \mathcal{N}(0, I)$ | 실제 features는 correlated |
| Well-specified model | Misspecification에서 결과 다름 |
| Linear model | NN의 nonlinear effect 미반영 |
| Proportional asymptotics $n, p \to \infty$ | Finite sample correction 필요 |
| $\beta^*$ isotropic | Sparse $\beta^*$에서는 다른 rate (compressed sensing) |
| GD from $0$ | Other initializations → different biases |

**주의**: 정리 3.4는 **asymptotic** 결과. Finite $n$에서는 correction term (Marchenko-Pastur의 finite-sample version).

---

## 📌 핵심 정리

$$\boxed{R(\gamma) = \|\beta^*\|^2(1 - 1/\gamma)\mathbb{1}[\gamma > 1] + \sigma^2 \gamma/|\gamma - 1|}$$

| 개념 | 의미 |
|------|------|
| **Min-norm solution** | $\hat\beta = X^+ y$ — GD from 0의 수렴 |
| **Double Descent** | Peak at $\gamma = 1$, finite risk at $\gamma \to \infty$ |
| **Ridgeless limit** | $\lambda \to 0^+$ Ridge = min-norm |
| **Implicit reg** | GD + $\beta_0 = 0$ → Ridgeless 자동 선택 |
| **Ch6 맥락** | Early stopping (Ch6-01) + SGD bias (Ch6-02) + Ridgeless (이 문서) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $n = 100, p = 150$이면 $\gamma = ?$. Theorem 3.4로 risk를 추정하라 (가정: $\|\beta^*\|^2 = 1, \sigma^2 = 0.25$).

<details>
<summary>힌트 및 해설</summary>

$\gamma = 150/100 = 1.5$.

$R(\gamma) = \|\beta^*\|^2(1 - 1/\gamma) + \sigma^2 \gamma / (\gamma - 1)$  
$= 1 \cdot (1 - 1/1.5) + 0.25 \cdot 1.5 / 0.5$  
$= 1/3 + 0.75$  
$= 1.083$.

Baseline comparison: under-parameterized $\gamma = 0.5$에서 $R = \sigma^2 / (1 - \gamma) = 0.5$. Over-parameterized $\gamma = 1.5$의 risk는 약 2배 큼 — **peak 근처이기 때문**.

$\gamma = 10$이면 $R = 1 \cdot 0.9 + 0.25 \cdot 10/9 \approx 1.18$. $\gamma$ 크게 키우면 $R$이 asymptotically $\|\beta^*\|^2$로 수렴.

</details>

**문제 2** (심화): $\lambda \to 0^+$에서 Ridge가 min-norm으로 수렴하는 이유를 SVD로 show하라.

<details>
<summary>힌트 및 해설</summary>

$X = U\Sigma V^T$ (thin SVD, $\Sigma$ is $r \times r$ where $r = \text{rank}(X)$).

Ridge: $\hat{\beta}_R = V\Sigma(\Sigma^2 + \lambda I)^{-1}U^T y$.

$\Sigma(\Sigma^2 + \lambda I)^{-1} = \text{diag}(\sigma_i/(\sigma_i^2 + \lambda))$.

$\lambda \to 0^+$: $\sigma_i/(\sigma_i^2 + 0) = 1/\sigma_i$.

$\hat{\beta}_R \to V\Sigma^{-1}U^T y = X^+ y = \hat{\beta}_{\min}$.

이는 **exact limit**, not approximation. 따라서 "Ridge with infinitesimal $\lambda$" = "min-norm solution" **exactly**.

실전적 함의: 
- GD가 $\lambda \to 0^+$ ridge path의 끝점을 찾음.
- Early stopping은 path 중간에서 멈춤.
- 두 기법이 **same trajectory의 다른 지점**.

</details>

**문제 3** (이론-실전): 실전 Deep Learning에서 "clean Double Descent"가 **잘 안 보이는** 이유 세 가지를 나열하라.

<details>
<summary>힌트 및 해설</summary>

**1. Built-in Regularization**:
- Weight decay (L2) $\lambda > 0$ → peak 완화.
- Dropout $p > 0$ → additional randomness.
- BatchNorm, LayerNorm → landscape smoothing.

이 모든 것이 "implicit $\lambda$" 역할. Peak at $p = n$ 흐릿해짐.

**2. Non-linear / SGD Effects**:
- Linear theory (정리 3.4)는 정확히 성립 안 함.
- Feature learning → effective $p$가 $p_{\text{raw}}$보다 작음.
- SGD bias가 specific implicit regularization 추가.

**3. Definition of $p$**:
- NN의 "effective parameter count" 측정 어려움.
- Architecture (ResNet, Transformer)에 따라 "effective capacity" 다름.
- Parameter 수 vs effective dof 차이 커짐.

**4. Dataset size**:
- Modern benchmarks (ImageNet $n = 1.3M$)가 너무 커 classical double descent region 도달 안 함.

**결과**: Double Descent는 **specific controlled experiments** (Nakkiran 2019)에서만 명확히 관찰. Production 모델에서는 "평범한 monotonic improvement with scale" 보이는 것이 정상.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. SGD Implicit Bias](./02-sgd-implicit-bias.md) | [📚 README로 돌아가기](../README.md) | [04. Homogeneous Networks ▶](./04-feature-implicit-bias.md) |

</div>
