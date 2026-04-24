# 01. Early Stopping = Implicit L2

## 🎯 핵심 질문

- Gradient descent의 iterate $\hat{w}_t$가 왜 Ridge regression의 **정규화 경로**와 1:1 대응하는가?
- **$t \approx 1/(\eta\lambda)$** 대응 관계의 수학적 유도는?
- "Stop before convergence"가 L2 regularization과 같은 효과를 주는 원리는?
- **Spectral filter** 관점에서 두 기법은 어떻게 통합되는가?

---

## 🔍 왜 Early Stopping이 implicit L2인가

실전의 모순:
- "Underfit될 때까지 충분히 훈련" — 반대
- "Overfit 방지하려면 일찍 멈춰라" — 실전 조언
- **Why?** 수학적 정당화?

Yao, Rosasco, Caponnetto 2007의 답: **gradient descent의 iterate가 ridge path를 따라간다**.

구체적으로:
- Ridge path: $\lambda$가 $\infty$에서 $0$으로 감소하면서 $\hat{w}_R(\lambda)$가 0에서 OLS로 이동.
- GD path: $t = 0$에서 시작해 $t$ 증가하면서 $\hat{w}_t$가 0에서 OLS로 이동.
- **두 경로가 거의 같다** (SVD 기저에서 spectral filter로 본다면).

**정확한 대응**: GD의 step $t$ = Ridge의 $\lambda = 1/(\eta t)$ 근방.

이는 **explicit + implicit regularization의 통합** (Ch6 전체의 주제)의 첫 번째 예. 같은 spectral filter 관점(Ch1-04의 Ridge SVD)이 더 일반화된다.

---

## 📐 수학적 선행 조건

- Ch1-01, Ch1-04: Ridge regression, SVD 관점의 shrinkage filter
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): GD의 iterate 공식 $\hat w_{t+1} = \hat w_t - \eta \nabla L$
- 선형대수: SVD, eigenvalue 분해
- 확률 부등식: $(1 - x)^t \approx e^{-tx}$ for small $x$

---

## 📖 직관적 이해

### GD on Linear Regression

Loss $L(w) = \tfrac{1}{2}\|y - Xw\|^2$. Gradient:

$$\nabla L = -X^T(y - Xw) = X^T X w - X^T y$$

GD update (learning rate $\eta$, starting from $\hat{w}_0 = 0$):

$$\hat{w}_{t+1} = \hat{w}_t - \eta (X^T X \hat{w}_t - X^T y)$$

Solve iteratively:

$$\hat{w}_t = (I - (I - \eta X^T X)^t) (X^T X)^{-1} X^T y$$

### Spectral Decomposition

$X = U\Sigma V^T$ SVD. $X^T X = V \Sigma^2 V^T$. GD iterate in SVD basis:

$$\hat{w}_t = V \cdot \underbrace{\text{diag}(1 - (1 - \eta\sigma_i^2)^t)}_{\text{GD filter}} \cdot \Sigma^{-1} U^T y$$

Ridge (Ch1-04):

$$\hat{w}_R = V \cdot \underbrace{\text{diag}\left(\frac{\sigma_i^2}{\sigma_i^2 + \lambda}\right)}_{\text{Ridge filter}} \cdot \Sigma^{-1} U^T y$$

### 두 Filter의 유사성

| Singular value direction | GD filter $1 - (1 - \eta\sigma_i^2)^t$ | Ridge filter $\sigma_i^2/(\sigma_i^2 + \lambda)$ |
|-----|-----|-----|
| Large $\sigma_i$ (signal) | Fast 1로 수렴 | Close to 1 |
| Small $\sigma_i$ (noise) | Slow 수렴 | Close to 0 |
| $\sigma_i^2 = 1/(\eta t) \approx \lambda$ | Filter $\approx 0.63$ | Filter $= 0.5$ |

**핵심**: 두 filter 모두 **작은 $\sigma_i$ 방향을 shrink**. Signal direction은 유지, noise direction은 억제.

### 대응 관계

$t \cdot \eta = 1/\lambda$ 근방에서 두 filter가 비슷:

$$1 - (1 - \eta\sigma^2)^t \approx 1 - e^{-t\eta\sigma^2} \approx \frac{\sigma^2 t\eta}{1 + \sigma^2 t\eta}$$

(마지막 근사는 $t\eta\sigma^2$ 작을 때.) 이는 Ridge filter $\sigma^2/(\sigma^2 + \lambda)$에서 $\lambda = 1/(t\eta)$와 동일한 형태.

**Intuition**: "훈련 $t$ steps" ≈ "ridge $\lambda = 1/(\eta t)$ regularization".

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Gradient Descent Iterate

Quadratic loss $L(w) = \tfrac{1}{2}\|y - Xw\|^2$, learning rate $\eta < 2/\sigma_{\max}^2(X)$, initialization $\hat{w}_0 = 0$.

$$\hat{w}_{t+1} = \hat{w}_t - \eta (X^T X \hat{w}_t - X^T y)$$

Closed form (정리 1.3).

### 정리 1.2 — Ridge Regression Path

$$\hat{w}_R(\lambda) = (X^T X + \lambda I)^{-1} X^T y$$

SVD form: $\hat{w}_R = V \text{diag}(\sigma_i / (\sigma_i^2 + \lambda)) U^T y$ (Ch1-04).

### 정리 1.3 — GD Iterate Closed Form (Yao 2007)

$$\hat{w}_t = (X^T X)^{-1}(I - (I - \eta X^T X)^t) X^T y$$

SVD form:

$$\hat{w}_t = V \text{diag}\left(\frac{1 - (1 - \eta\sigma_i^2)^t}{\sigma_i}\right) U^T y$$

(Assuming $|1 - \eta\sigma_i^2| < 1$ for convergence.)

### 정리 1.4 — Spectral Filter 대응

GD step $t$의 filter factor: $\phi_{\text{GD}}(\sigma; t, \eta) = 1 - (1 - \eta\sigma^2)^t$.

Ridge의 filter factor: $\phi_{\text{Ridge}}(\sigma; \lambda) = \sigma^2/(\sigma^2 + \lambda)$.

**대응**: $\lambda_t \approx 1/(\eta t)$에서 두 filter의 $L^2$ 오차가 작다:

$$\|\phi_{\text{GD}}(\cdot; t, \eta) - \phi_{\text{Ridge}}(\cdot; 1/(\eta t))\|_{L^2} \leq C/t$$

### 정리 1.5 — Rate Match: GD와 Ridge의 L2 Norm

$$\|\hat{w}_t\|_2 \leq \|\hat{w}_{\text{OLS}}\|_2$$

즉 GD iterate의 norm이 OLS 해보다 작음. $t$ 증가하면 monotonically 증가 → OLS로 수렴.

Ridge의 $\|\hat{w}_R(\lambda)\|_2$도 $\lambda$ 감소할수록 증가 → OLS 수렴.

두 path의 "**norm trajectory**"가 거의 일치한다.

---

## 🔬 수학적 유도

### 정리 1.3 증명

$\hat{w}_0 = 0$. Update $\hat{w}_{t+1} = \hat{w}_t + \eta X^T(y - X\hat{w}_t) = \hat{w}_t + \eta X^T y - \eta X^T X \hat{w}_t$.

$= (I - \eta X^T X) \hat{w}_t + \eta X^T y$.

Telescoping:

$\hat{w}_t = \sum_{k=0}^{t-1} (I - \eta X^T X)^k \eta X^T y = \eta \left[\sum_{k=0}^{t-1}(I - \eta X^T X)^k\right] X^T y$

Geometric sum (matrix version, assume $(I - \eta X^T X)$의 eigenvalues $< 1$):

$\sum_{k=0}^{t-1}(I - \eta X^TX)^k = (X^T X)^{-1}(I - (I - \eta X^T X)^t) / \eta$

(Derivation: $(I - M)\sum_{k=0}^{t-1} M^k = I - M^t$ → $\sum_{k=0}^{t-1} M^k = (I - M)^{-1}(I - M^t)$. Here $I - M = \eta X^T X$.)

Thus:

$\hat{w}_t = (X^T X)^{-1}(I - (I - \eta X^T X)^t) X^T y \quad \square$

### SVD 기저에서의 Filter

$X = U\Sigma V^T$. $X^T X = V\Sigma^2 V^T$. $(I - \eta X^T X)^t = V(I - \eta\Sigma^2)^t V^T$.

$\hat{w}_t = V(\Sigma^{-2})(I - (I - \eta\Sigma^2)^t) \Sigma U^T y = V \text{diag}(\phi(\sigma_i))U^T y$

where $\phi(\sigma_i) = (1 - (1 - \eta\sigma_i^2)^t)/\sigma_i$.

### 정리 1.4 증명 스케치 (filter 대응)

$(1 - \eta\sigma^2)^t$를 $-t$가 큰 경우 $\eta\sigma^2 \to 0$이면 $(1 - \eta\sigma^2)^t \approx e^{-t\eta\sigma^2}$.

따라서 GD filter $\approx 1 - e^{-t\eta\sigma^2}$.

Ridge filter $= \sigma^2/(\sigma^2 + \lambda) = 1 - \lambda/(\sigma^2 + \lambda)$.

대응 $\lambda = 1/(\eta t)$ 가정:

Ridge filter $= 1 - 1/(\sigma^2 \eta t + 1) \approx 1 - e^{-\sigma^2 \eta t}$ (for $\sigma^2 \eta t \gg 1$ 쪽).

두 filter의 형태가 매우 비슷 (완전 동일은 아니지만). $\square$

---

## 💻 실험으로 효과 검증

### 실험 1 — GD iterate vs Ridge path 비교

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n, d = 50, 30
X = np.random.randn(n, d) / np.sqrt(n)
w_true = np.random.randn(d)
y = X @ w_true + 0.3 * np.random.randn(n)

# Ridge path
lams = np.logspace(-3, 2, 40)
ridge_path = np.array([
    np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y) for lam in lams
])

# GD iterates
eta = 0.01
max_steps = 2000
gd_iterates = np.zeros((max_steps, d))
w = np.zeros(d)
for t in range(max_steps):
    grad = -X.T @ (y - X @ w)
    w -= eta * grad
    gd_iterates[t] = w

# 두 path에서 w_i를 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i in range(d):
    axes[0].semilogx(lams, ridge_path[:, i], alpha=0.6)
axes[0].set_xlabel(r'$\lambda$'); axes[0].set_ylabel('coef')
axes[0].set_title('Ridge path ($\lambda$ large → small)')

for i in range(d):
    axes[1].plot(gd_iterates[:, i], alpha=0.6)
axes[1].set_xlabel('GD iteration t'); axes[1].set_ylabel('coef')
axes[1].set_title('GD iterate (t = 0 → large)')
plt.tight_layout(); plt.show()
```

**관찰**: 두 plot이 **mirror image** — Ridge는 $\lambda$ 감소 방향, GD는 $t$ 증가 방향. 같은 "학습 경로".

### 실험 2 — Filter factor 직접 비교

```python
sigmas = np.logspace(-2, 1, 100)

def gd_filter(sigma, t, eta):
    return 1 - (1 - eta * sigma**2)**t
def ridge_filter(sigma, lam):
    return sigma**2 / (sigma**2 + lam)

eta = 0.1
fig, ax = plt.subplots(figsize=(9, 5))
for t in [1, 10, 100, 1000]:
    lam_equiv = 1 / (eta * t)
    ax.plot(sigmas, gd_filter(sigmas, t, eta), label=f'GD t={t}', lw=2)
    ax.plot(sigmas, ridge_filter(sigmas, lam_equiv), '--', label=f'Ridge λ={lam_equiv:.3f}', lw=2)
ax.set_xscale('log'); ax.set_xlabel(r'$\sigma$'); ax.set_ylabel('filter value')
ax.set_title('GD vs Ridge spectral filters (paired by t·η = 1/λ)')
ax.legend(); ax.grid(alpha=0.3); plt.show()
```

**관찰**: 같은 $t\eta = 1/\lambda$에서 두 filter가 매우 비슷한 모양. **완전 동일은 아님** (GD filter는 sharper transition) 하지만 trade-off 거의 같음.

### 실험 3 — Test loss at "equivalent" λ/t

```python
# For each λ (or t), compute test loss
def test_loss(w, X_test, y_test):
    return np.mean((y_test - X_test @ w)**2)

X_test = np.random.randn(200, d) / np.sqrt(n)
y_test = X_test @ w_true + 0.3 * np.random.randn(200)

# Ridge path test loss
ridge_losses = [test_loss(w, X_test, y_test) for w in ridge_path]

# GD test loss (subsample every 10 steps)
gd_losses = [test_loss(gd_iterates[t], X_test, y_test) for t in range(0, max_steps, 10)]

# Plot — x-axis는 "effective regularization" (1/t or λ)
effective_lams_gd = 1 / (np.arange(1, max_steps+1, 10) * eta)

plt.loglog(lams, ridge_losses, 'o-', label='Ridge')
plt.loglog(effective_lams_gd, gd_losses, 's-', label=r'GD (effective $\lambda = 1/\eta t$)')
plt.xlabel(r'effective $\lambda$ (or $1/\eta t$)')
plt.ylabel('Test MSE')
plt.title('Test loss: Ridge vs Early-Stopped GD')
plt.gca().invert_xaxis()  # λ 감소 = t 증가 방향
plt.legend(); plt.grid(alpha=0.3); plt.show()
# → 두 curve가 거의 일치
```

### 실험 4 — Optimal stopping time 찾기

```python
# Validation을 사용한 early stopping
X_val = np.random.randn(100, d) / np.sqrt(n)
y_val = X_val @ w_true + 0.3 * np.random.randn(100)

val_losses = [test_loss(gd_iterates[t], X_val, y_val) for t in range(max_steps)]
t_optimal = np.argmin(val_losses)
print(f"Optimal stopping time: t = {t_optimal}")
print(f"Equivalent λ: {1/(eta * t_optimal):.3f}")
# 이 λ가 Ridge의 best validation λ와 비슷해야 함
```

---

## 🔗 실전 활용

### Early Stopping in Deep Learning

Linear regression의 이론이 NN에서는 **정확히** 성립하지 않지만, **정성적으로** 유효:

- **초기 훈련**: underparameterized region — 빠른 수렴.
- **중반 훈련**: overfitting 시작.
- **후반**: validation loss 상승 — 여기가 early stop point.

**실전 가이드**:
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # continue
        else:
            self.counter += 1
            return self.counter >= self.patience
```

### GD vs NN training의 차이

- **Linear**: Closed-form equivalence to Ridge.
- **Non-linear NN**: 
  - Loss landscape non-convex → 여러 local minima.
  - Implicit regularization이 복잡 (NTK regime, feature learning 등).
  - 하지만 **영혼은 같음**: 훈련 시간이 effective capacity 제어.

### 다른 implicit regularization과의 관계

- **SGD noise** (Ch6-02): Gaussian noise가 추가 regularization.
- **Overparameterization** (Ch6-03): Ridgeless limit.
- **Early stopping**: 이 문서 — 훈련 시간 제한.

세 요소가 합쳐 modern NN의 implicit regularization을 구성.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Quadratic loss | Non-convex NN에서 근사만 유효 |
| $\hat w_0 = 0$ 초기화 | 다른 초기화에서 convergence behavior 다름 |
| Full gradient | SGD는 additional noise — Ch6-02 |
| Learning rate 고정 | Scheduled lr에서 대응 관계 수정 필요 |
| 선형 모델 | NN의 nonlinear training에는 정량적 대응 없음 |

---

## 📌 핵심 정리

$$\boxed{\hat{w}_t \approx \hat{w}_R(\lambda = 1/(\eta t)) \quad \text{for linear regression}}$$

| 개념 | 의미 |
|------|------|
| **Yao et al. 2007** | GD iterate = Ridge path correspondence |
| **Spectral filter** | GD $1 - (1-\eta\sigma^2)^t$ ≈ Ridge $\sigma^2/(\sigma^2 + \lambda)$ |
| **$t \cdot \eta = 1/\lambda$** | 핵심 대응 관계 |
| **"Stop before convergence"** | Early stopping = implicit L2 |
| **Ch6 시작** | Implicit regularization 첫 예 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\eta = 0.01$, $\lambda_{\text{equiv}} = 0.1$에 대응되는 GD step 수는?

<details>
<summary>힌트 및 해설</summary>

$t = 1/(\eta\lambda) = 1/(0.01 \times 0.1) = 1000$.

즉 **1000 steps GD**가 $\lambda = 0.1$ Ridge와 근사 동등.

일반적 rule: 작은 $\eta$ → 더 많은 steps 필요. $\eta \to 0$이면 continuous gradient flow (GF), $t$가 continuous.

</details>

**문제 2** (심화): GD filter $1 - (1-\eta\sigma^2)^t$와 Ridge filter $\sigma^2/(\sigma^2 + \lambda)$가 정확히 같아지는 $(t, \eta, \lambda, \sigma)$ 관계가 있는가?

<details>
<summary>힌트 및 해설</summary>

**정확한 equivalence는 없음**. 두 filter는 structural하게 다른 함수:

GD filter: $1 - (1 - x)^t$ where $x = \eta\sigma^2$.  
Ridge filter: $x/(x + \lambda)$.

**근사 equivalence** (for small $x$):
- $1 - (1 - x)^t \approx 1 - e^{-tx} \approx tx - (tx)^2/2$ (Taylor).
- $x/(x + \lambda) = 1/(1 + \lambda/x) \approx x/\lambda$ for small $x/\lambda$.

두 근사 값이 같으려면 $tx \approx x/\lambda$ → $\lambda \approx 1/t$.

더 정확한 match는 각 $\sigma$ 별로 다른 $\lambda$ 필요 — single $\lambda$로는 완전 match 안 됨.

**교훈**: Early stopping과 Ridge는 "유사한 regularization"이지만 **정확히 같지는 않다**. Domain 별로 미묘한 차이. 실전에서는 양쪽 다 tuning.

</details>

**문제 3** (이론-실전): Deep NN에서 early stopping이 실전에서 효과적인 경험적 이유는? Linear model의 이론이 어떻게 NN에 extend되는가?

<details>
<summary>힌트 및 해설</summary>

**Linear에서 정확한 equivalence → NN에서 유사성**:

1. **Kernel regime** (Ch3 NTK, Generalization Theory Deep Dive): 무한 폭 NN은 kernel regression → spectral filter 해석 그대로 적용.
2. **Feature learning regime**: Early training은 kernel-like, late training은 feature-adapting. Early stopping이 kernel regime 유지.
3. **Empirical observation**: Val loss가 U-shape → minimum이 early stop point.

**NN에서의 추가 복잡성**:
- Non-convex landscape: multiple saddle, local minima.
- Mini-batch SGD: extra noise regularization.
- Architecture bias: initialization의 implicit bias.

**그럼에도 early stopping이 기본 트릭으로 작동하는 이유**:
- NN의 "effective capacity"도 훈련 시간에 비례해 증가 (double descent에서도).
- Validation set이 실전 generalization의 proxy.

**실전 guideline**:
- Modern large model (GPT 등): early stopping 덜 critical (huge data로 거의 overfit 안 됨).
- Small to medium model: **필수** tool.
- Adaptive scheduling (cosine, warmup)과 함께 사용 — best of both.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Chapter 5 → 04. Temperature Scaling](../ch5-label-calibration/04-temperature-scaling.md) | [📚 README로 돌아가기](../README.md) | [02. SGD의 Implicit Bias ▶](./02-sgd-implicit-bias.md) |

</div>
