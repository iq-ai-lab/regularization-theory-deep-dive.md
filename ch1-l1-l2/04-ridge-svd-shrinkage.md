# 04. Ridge의 SVD 관점 — Shrinkage

## 🎯 핵심 질문

- Ridge 해를 $X = U\Sigma V^T$의 SVD로 어떻게 다시 쓸 수 있는가?
- 왜 **작은 singular value 방향이 더 많이 축소**되는가?
- Ridge의 shrinkage factor $\sigma_i^2/(\sigma_i^2 + \lambda)$는 어떻게 "효과적 자유도 (effective dof)"를 정의하는가?
- Principal Component Regression (PCR)과 Ridge의 차이는?

---

## 🔍 왜 SVD 관점이 필요한가

Ch1-01은 Ridge의 Bayesian 해석을, Ch1-03은 기하를 주었다. SVD 관점은 **세 번째 층**:

1. **Bayesian** (Ch1-01): "L2 = Gaussian prior MAP" — **왜** regularize하는가
2. **기하** (Ch1-03): "L1 ball 꼭짓점" — **무엇이 달라지는가** (sparsity)
3. **SVD/Spectral** (이 문서): "작은 $\sigma_i$ 방향 강한 shrink" — **어떻게** shrink하는가

SVD는 또한 **Early Stopping = Implicit L2**(Ch6-01), **Double Descent의 peak at $p=n$**, **PCR과의 비교**에 필수적이다. "Spectral filter" 관점은 regularization 이론 전체를 관통하는 통일 framework.

---

## 📐 수학적 선행 조건

- 선형대수: SVD $X = U\Sigma V^T$, singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$ ($r = \text{rank}(X)$)
- Ridge의 closed form $\hat{w}_R = (X^TX + \lambda I)^{-1}X^T y$ (Ch1-01)
- [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive): degrees of freedom의 정의 $\text{df} = \text{tr}(\text{Cov}(\hat{y}, y)/\sigma^2)$
- 기초 통계: bias-variance decomposition

---

## 📖 직관적 이해

### SVD가 하는 일

$X = U\Sigma V^T$로 $X$를 세 개의 "움직임"으로 분해:

1. $V^T$: 입력을 **feature 주축**으로 회전
2. $\Sigma$: 각 주축을 $\sigma_i$로 스케일 (중요도 순)
3. $U$: 출력 공간으로 회전

$\sigma_i$가 크다 = "$i$번째 principal direction이 데이터에서 강하게 표현됨".

### Ridge의 spectral filter

Ridge 해를 SVD 기저에서 쓰면 각 principal direction의 coefficient가 **filter $f(\sigma_i) = \sigma_i^2/(\sigma_i^2 + \lambda)$로 가중**:

- $\sigma_i \gg \sqrt{\lambda}$: $f \approx 1$ (거의 변화 없음, signal-dominated direction)
- $\sigma_i \ll \sqrt{\lambda}$: $f \approx 0$ (크게 shrink됨, noise-dominated direction)
- $\sigma_i = \sqrt{\lambda}$: $f = 1/2$ (중간)

**핵심 교훈**: Ridge는 "**noise에 더 취약한 방향을 더 많이 축소**"한다. 이것이 overfitting 방지의 메커니즘.

### Ridge vs PCR 비교

| 기법 | Filter $f(\sigma_i)$ |
|------|------|
| OLS ($\lambda = 0$) | $1$ |
| Ridge | $\sigma_i^2/(\sigma_i^2 + \lambda)$ (smooth) |
| PCR (top-$k$ PCs) | $\mathbb{1}[i \leq k]$ (hard threshold) |
| Lasso (SVD 기저에서는 non-filter) | N/A |

**Ridge는 smooth shrinkage, PCR은 hard selection**. Ridge는 모든 direction을 약간씩 쓰고, PCR은 일부 direction만 완전히 쓴다.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Thin SVD

$X \in \mathbb{R}^{n \times d}$, $r = \text{rank}(X)$. Thin SVD:

$$X = U \Sigma V^T, \quad U \in \mathbb{R}^{n \times r}, \ \Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r), \ V \in \mathbb{R}^{d \times r}$$

$U, V$는 orthonormal columns, $\sigma_1 \geq \cdots \geq \sigma_r > 0$.

### 정리 4.2 — Ridge Solution in SVD Form (주 정리)

Ridge regression의 해:

$$\boxed{\hat{w}_R = \sum_{i=1}^{r} \frac{\sigma_i}{\sigma_i^2 + \lambda} (u_i^T y) \, v_i}$$

혹은 행렬 형태로:

$$\hat{w}_R = V \, \text{diag}\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U^T y$$

### 정리 4.3 — Fitted Values의 Filter

Ridge fitted values:

$$\hat{y}_R = X\hat{w}_R = \sum_{i=1}^{r} \frac{\sigma_i^2}{\sigma_i^2 + \lambda} (u_i^T y) \, u_i = U \, \text{diag}\left(f(\sigma_i)\right) U^T y$$

**Filter** $f(\sigma) = \sigma^2/(\sigma^2 + \lambda) \in [0, 1]$, $\sigma \downarrow 0$에서 $f \to 0$, $\sigma \uparrow \infty$에서 $f \to 1$.

### 정의 4.4 — Effective Degrees of Freedom

$$\text{df}(\lambda) := \text{tr}(H_\lambda) = \sum_{i=1}^{r} \frac{\sigma_i^2}{\sigma_i^2 + \lambda}$$

여기서 $H_\lambda = X(X^TX + \lambda I)^{-1}X^T$는 Ridge의 hat matrix. $\lambda = 0$이면 $\text{df} = r$, $\lambda \to \infty$이면 $\text{df} \to 0$.

### 정리 4.5 — Bias-Variance Decomposition

$y = Xw^* + \varepsilon, \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$에서:

$$\mathbb{E}\|\hat{w}_R - w^*\|^2 = \underbrace{\sum_i \left(\frac{\lambda}{\sigma_i^2 + \lambda}\right)^2 (v_i^T w^*)^2}_{\text{bias}^2} + \underbrace{\sigma^2 \sum_i \frac{\sigma_i^2}{(\sigma_i^2 + \lambda)^2}}_{\text{variance}}$$

**Trade-off**: $\lambda \uparrow$면 bias $\uparrow$, variance $\downarrow$. 최적 $\lambda$가 존재.

### 정리 4.6 — Ridge vs PCR vs Minimum Norm Solution

| 기법 | 해 | Shrinkage |
|------|-----|-----------|
| **OLS** ($\lambda=0$, $n \geq d$) | $V\Sigma^{-1}U^T y$ | 없음 |
| **Minimum-norm LS** ($n < d$) | $\hat{w}_{\min} = \sum \sigma_i^{-1} (u_i^T y) v_i$ | (rank 방향만) |
| **Ridge** | $\sum \frac{\sigma_i}{\sigma_i^2+\lambda}(u_i^T y) v_i$ | smooth, 모든 $\sigma_i$ |
| **PCR-$k$** | $\sum_{i=1}^k \sigma_i^{-1}(u_i^T y) v_i$ | hard, top-$k$만 |

$\lambda \to 0^+$에서 Ridge → minimum-norm LS.

---

## 🔬 수학적 유도

### 정리 4.2 증명

$X^T X = V \Sigma^2 V^T$, $X^T y = V \Sigma U^T y$:

$$X^T X + \lambda I = V(\Sigma^2 + \lambda I) V^T + \lambda(I - VV^T)$$

단, $V$가 full column rank이면 $VV^T \neq I$일 수 있지만 $V(\Sigma^2 + \lambda I)V^T$ 부분만 해에 기여. 정확히는 feature 공간을 $V$-span과 그 직교보공간으로 분해:

$$\hat{w}_R = (X^TX + \lambda I)^{-1} X^T y$$

$X^T y$는 $V$-span 위에 있으므로 해도 $V$-span 위. $V$-span에서 $X^TX + \lambda I$는 $V(\Sigma^2 + \lambda I)V^T$이고 inverse는 $V(\Sigma^2+\lambda I)^{-1}V^T$:

$$\hat{w}_R = V(\Sigma^2 + \lambda I)^{-1} V^T \cdot V\Sigma U^T y = V(\Sigma^2 + \lambda I)^{-1} \Sigma U^T y$$

$(\Sigma^2 + \lambda I)^{-1} \Sigma = \text{diag}(\sigma_i/(\sigma_i^2 + \lambda))$. $\square$

### 정리 4.5 — Bias-variance 유도

$\mathbb{E}[\hat{w}_R] = (X^TX + \lambda I)^{-1}X^T X w^* = V \text{diag}(\sigma_i^2/(\sigma_i^2+\lambda))V^T w^*$.

**Bias**: $\mathbb{E}[\hat{w}_R] - w^* = V \text{diag}(\sigma_i^2/(\sigma_i^2+\lambda) - 1) V^T w^* = -V \text{diag}(\lambda/(\sigma_i^2+\lambda))V^T w^*$.

$\|\text{bias}\|^2 = \sum_i (\lambda/(\sigma_i^2 + \lambda))^2 (v_i^T w^*)^2$.

**Variance**: $\text{Cov}(\hat{w}_R) = \sigma^2 (X^TX + \lambda I)^{-1} X^T X (X^TX + \lambda I)^{-1} = \sigma^2 V \text{diag}(\sigma_i^2/(\sigma_i^2+\lambda)^2) V^T$.

$\mathbb{E}\|\hat{w}_R - \mathbb{E}\hat{w}_R\|^2 = \text{tr}(\text{Cov}) = \sigma^2 \sum_i \sigma_i^2/(\sigma_i^2+\lambda)^2$. $\square$

### $\lambda \to 0$에서의 minimum-norm solution

$p > n$ overparameterized에서 $\sigma_i > 0 \ (i \leq n)$, $\sigma_i = 0 \ (i > n)$. Ridge 해가 $\lambda \to 0$에서:

$$\hat{w}_R \to \sum_{i=1}^n \sigma_i^{-1} (u_i^T y) v_i = X^+ y$$

이것이 **minimum-norm solution** — overparameterized regime에서 "implicit regularization"(Ch6-03)의 starting point. Generalization Theory Deep Dive의 Double Descent와 직접 연결됨.

---

## 💻 실험으로 효과 검증

### 실험 1 — 여러 $\sigma_i$ 방향의 shrinkage filter 플롯

```python
import numpy as np
import matplotlib.pyplot as plt

sigmas = np.logspace(-2, 2, 400)
for lam in [0.01, 0.1, 1.0, 10.0]:
    plt.plot(sigmas, sigmas**2 / (sigmas**2 + lam), label=fr'$\lambda={lam}$')
plt.xscale('log')
plt.xlabel(r'singular value $\sigma$')
plt.ylabel(r'Ridge filter $\sigma^2/(\sigma^2+\lambda)$')
plt.title('Ridge spectral filter — 작은 σ 방향을 더 많이 shrink')
plt.axhline(0.5, ls='--', c='gray', lw=0.7, label='50% shrink')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.show()
```

**관찰**: Knee point는 $\sigma = \sqrt{\lambda}$. 이보다 작은 $\sigma$ 방향은 거의 완전히 제거됨.

### 실험 2 — Bias-variance trade-off

```python
np.random.seed(0)
n, d = 80, 50
X = np.random.randn(n, d) / np.sqrt(n)
w_star = np.random.randn(d)

U, sing, Vt = np.linalg.svd(X, full_matrices=False)
sigma_eps = 0.3

def ridge_risk(lam):
    # Analytical from Theorem 4.5
    filt = sing / (sing**2 + lam)
    bias2 = np.sum(((lam / (sing**2 + lam)))**2 * (Vt @ w_star)**2)
    var = sigma_eps**2 * np.sum(sing**2 / (sing**2 + lam)**2)
    return bias2, var, bias2 + var

lams = np.logspace(-4, 2, 100)
risks = np.array([ridge_risk(lam) for lam in lams])

plt.figure(figsize=(8, 5))
plt.loglog(lams, risks[:, 0], label='bias²')
plt.loglog(lams, risks[:, 1], label='variance')
plt.loglog(lams, risks[:, 2], 'k-', lw=2, label='total risk')
opt_lam = lams[np.argmin(risks[:, 2])]
plt.axvline(opt_lam, c='r', ls='--', label=fr'optimal $\lambda={opt_lam:.3f}$')
plt.xlabel(r'$\lambda$'); plt.ylabel('MSE')
plt.title('Ridge bias-variance trade-off (정리 4.5)')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.show()
```

### 실험 3 — Effective df vs raw parameter count

```python
n, d = 50, 30
X = np.random.randn(n, d) / np.sqrt(n)
_, sing, _ = np.linalg.svd(X, full_matrices=False)

lams = np.logspace(-4, 2, 200)
df = np.array([np.sum(sing**2 / (sing**2 + lam)) for lam in lams])

plt.semilogx(lams, df)
plt.axhline(d, ls='--', c='r', label=f'raw d={d}')
plt.axhline(0, ls='--', c='k')
plt.xlabel(r'$\lambda$'); plt.ylabel('effective df')
plt.title('Ridge의 effective degrees of freedom — λ 증가에 따라 d에서 0으로')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 실험 4 — Ridge vs PCR 시각 비교

```python
def pcr(X, y, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T[:, :k] @ np.diag(1/s[:k]) @ U[:, :k].T @ y

def ridge(X, y, lam):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ np.diag(s/(s**2 + lam)) @ U.T @ y

y = X @ w_star + sigma_eps * np.random.randn(n)

fig, ax = plt.subplots(figsize=(9, 5))
for lam in np.logspace(-3, 1, 5):
    w_r = ridge(X, y, lam)
    ax.plot(sing, [(w_r @ Vt[i]) for i in range(d)], 'o-',
            label=fr'Ridge λ={lam:.2f}', alpha=0.6)
for k in [5, 10, 20]:
    w_p = pcr(X, y, k)
    ax.plot(sing, [(w_p @ Vt[i]) for i in range(d)], 's--',
            label=f'PCR k={k}', alpha=0.5)
ax.set_xscale('log'); ax.set_xlabel(r'$\sigma_i$'); ax.set_ylabel(r'$v_i^T \hat w$')
ax.set_title('Ridge smooth shrink vs PCR hard threshold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.show()
```

---

## 🔗 실전 활용

### 언제 SVD 관점이 유용한가

1. **$p \gg n$에서 underdetermined**: Ridge의 $\lambda \to 0$이 minimum-norm 해로 가는 것을 SVD로 확인 — NTK/overparameterized regime 이해의 기초.
2. **Condition number 진단**: $\sigma_{\max}/\sigma_{\min}$이 너무 크면 OLS는 극도로 불안정. Ridge의 $\lambda$가 "effective smallest singular value"를 $\max(\sigma_{\min}, \sqrt{\lambda})$로 끌어올림.
3. **Cross-validation**: Effective df를 통해 $\lambda$ 범위를 의미 있게 스캔 (df ≈ 1, 5, 10, ...).
4. **Early stopping → Ch6-01**: GD iterate도 SVD 기저에서 spectral filter $1 - (1 - \eta\sigma_i^2)^t$로 쓸 수 있음 → Ridge와 거의 같은 shrinkage.

### 커널·NN으로의 확장

NN의 **NTK regime**에서는 $\Theta(X, X)$의 eigendecomposition이 동일한 spectral filter 해석을 준다. Generalization Theory Deep Dive Ch3의 NTK kernel regression도 같은 framework.

### 실전 tip

- Ridge가 "잘 안 먹는다"고 느끼면 → feature 표준화 후 다시.
- Ill-conditioned $X^TX$에서는 `np.linalg.lstsq` 대신 SVD-based `scipy.linalg.lstsq` 추천.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 선형 모델 | NN에서는 SVD filter가 NTK regime에서만 직접 적용 |
| Gaussian noise | Heavy-tailed에서는 bias-variance 해석 수정 필요 |
| 등방성 L2 penalty | Feature별로 다른 penalty면 $\Lambda = V^T \Lambda V$로 간단화 안 됨 |
| Rank $r$ 기준 분석 | 실전에서 "effective rank"는 singular value의 급격한 감소로 정의 |

---

## 📌 핵심 정리

$$\boxed{\hat{w}_R = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}(u_i^T y) v_i \quad — \text{ smooth shrinkage filter}}$$

| 개념 | 의미 |
|------|------|
| **Spectral filter $f(\sigma)$** | $\sigma^2/(\sigma^2+\lambda)$ — smooth transition at $\sigma = \sqrt\lambda$ |
| **Effective df** | $\sum f(\sigma_i)$ — 실질적 자유도 |
| **Ridge vs PCR** | smooth vs hard threshold |
| **$\lambda \to 0$** | minimum-norm LS ($p > n$) 또는 OLS ($p \leq n$) |
| **다음 질문** | 상관 feature 그룹·group sparsity → Ch1-05 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $X$의 singular values가 $(\sigma_1, \sigma_2, \sigma_3) = (3, 1, 0.1)$일 때 $\lambda = 1$의 Ridge filter를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$f(\sigma) = \sigma^2/(\sigma^2 + 1)$:

- $\sigma_1 = 3$: $9/10 = 0.9$ (거의 유지)
- $\sigma_2 = 1$: $1/2 = 0.5$ (절반 shrink)
- $\sigma_3 = 0.1$: $0.01/1.01 \approx 0.0099$ (거의 완전 제거)

Effective df $= 0.9 + 0.5 + 0.0099 \approx 1.41$. 원래 rank는 3이지만 Ridge가 effectively "1.4 parameter"만 사용.

</details>

**문제 2** (심화): Ridge의 bias²와 variance를 total risk로 합쳐 $\lambda^*$를 minimize하는 조건을 구하라. 한 $\sigma_i$만 고려한 단순 경우.

<details>
<summary>힌트 및 해설</summary>

$R(\lambda) = (\lambda/(\sigma^2+\lambda))^2 (v^Tw^*)^2 + \sigma_\varepsilon^2 \sigma^2/(\sigma^2+\lambda)^2$.

$(\sigma^2+\lambda)^3 dR/d\lambda$를 전개하면:

$\frac{dR}{d\lambda} = \frac{2\lambda (v^Tw^*)^2}{(\sigma^2+\lambda)^3} \sigma^2 - \frac{2\sigma_\varepsilon^2 \sigma^2}{(\sigma^2+\lambda)^3} = 0$

$\implies \lambda^* = \sigma_\varepsilon^2 / (v^Tw^*)^2$

해석: **signal 방향의 크기** $|v^T w^*|$에 **반비례**하는 $\lambda$가 최적. Signal이 강한 방향은 덜 regularize해야 한다. 전체 Ridge에서는 모든 $\sigma_i$에 같은 $\lambda$를 쓰므로 이는 평균적인 trade-off.

</details>

**문제 3** (이론-실전): $p \gg n$ overparameterized linear model에서 $\hat{w}_R|_{\lambda \to 0}$의 한계 해는? 이것이 **implicit regularization from initialization**과 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

$\lambda \to 0^+$에서 $\hat{w}_R \to X^+ y = V \Sigma^{-1} U^T y$ — **minimum-norm solution**.

- $X^+$는 Moore-Penrose pseudoinverse.
- 여러 interpolating solution 중 $\|w\|$ 최소인 것.
- **SGD가 0에서 시작해 훈련하면 실제로 이 solution으로 수렴** (Ch6-03 Ridgeless regression).
- Early stopping(Ch6-01)은 이 수렴 경로의 한 중간 지점.

즉 Ridge의 $\lambda \to 0$ 극한 = minimum-norm = SGD의 implicit bias. **모든 regularization 관점이 spectral filter로 통일**되는 예.

생성 AI 시대의 Scaling Law 와 Double Descent (Generalization Theory Deep Dive Ch4)가 overparameterized regime의 ridgeless 극한을 주요 대상으로 삼는 이유.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Sparsity의 기하학](./03-sparsity-geometry.md) | [📚 README로 돌아가기](../README.md) | [05. Elastic Net과 Group Lasso ▶](./05-elastic-net-group-lasso.md) |

</div>
