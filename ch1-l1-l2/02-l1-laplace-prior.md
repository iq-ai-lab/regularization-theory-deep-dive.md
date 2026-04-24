# 02. L1 Regularization = Laplace Prior MAP

## 🎯 핵심 질문

- $\lambda\|w\|_1$ 항은 왜 **Laplace prior**의 negative log와 정확히 같은가?
- $|w|$에서 미분 불가능한 점(원점)을 어떻게 다루는가? — **subdifferential**
- **소프트 thresholding** 연산자는 어디에서 나오는가?
- Coordinate descent로 Lasso를 푸는 알고리즘은 무엇인가?

---

## 🔍 왜 이 관점이 regularization 이해에 중요한가

L1의 "sparsity"는 종종 **관측 현상**으로 취급된다: "Lasso를 쓰면 coefficient가 0이 된다". 하지만 **왜**를 설명하려면 두 층이 필요하다.

1. **확률적 층 (이 문서)**: L1 = Laplace prior MAP. Laplace 분포는 원점에 **뾰족한 peak**를 갖기 때문에 posterior mode가 0에 "잘 붙는다".
2. **기하적 층 (Ch1-03)**: L1 ball의 **꼭짓점**에서 loss contour가 접할 확률 1 → sparse solution.

이 문서는 첫 번째 층을 완성한다. 또한 **subdifferential** 테크닉을 도입하여 smooth optimization과 다른 접근을 정식화한다 — 이는 Ch1-05의 proximal gradient(ISTA/FISTA)까지 이어진다.

---

## 📐 수학적 선행 조건

- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): prior, MAP
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Laplace 분포 $p(w) = \frac{\lambda}{2} e^{-\lambda|w|}$의 density와 log-density
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): convex function, **subdifferential** $\partial f(x) = \{g : f(y) \geq f(x) + g^T(y-x), \forall y\}$
- Ch1-01: L2 = Gaussian Prior MAP — 대응되는 구조 체감

---

## 📖 직관적 이해

### Laplace 분포의 "뾰족함"

$p(w) = \tfrac{\lambda}{2} e^{-\lambda |w|}$는 평균 0 · 분산 $2/\lambda^2$인 density로, $w = 0$에서 **cusp**(뾰족점)을 갖는다. Gaussian과 비교:

| 분포 | 원점 근처 | Tail |
|------|--------|------|
| Gaussian $\mathcal{N}(0, \sigma^2)$ | 매끄러운 bell | $e^{-w^2/(2\sigma^2)}$ (가벼운) |
| Laplace $\mathcal{L}(0, 1/\lambda)$ | **뾰족** | $e^{-\lambda|w|}$ (더 무거운) |

**두 함의**:
1. Laplace는 **원점에 더 많은 mass**를 집중 — "$w$가 **정확히** 0일 가능성이 높다"는 믿음.
2. Gaussian 대비 **heavy-tail** — "꽤 큰 $w$도 가끔은 허용한다".

이 두 속성이 Lasso의 sparsity와 feature selection 성격을 동시에 설명한다.

### 왜 "정확히 0"이 가능한가

Gaussian prior MAP는 $\hat{w} = (X^TX+\lambda I)^{-1}X^T y$ — 각 coordinate는 continuous shrinkage만. Laplace prior MAP는 **cusp** 때문에 $|X_j^T r| \leq \lambda$인 coordinate에서 **$w_j = 0$을 점 해로 받는다**. 이 점 해가 바로 **소프트 thresholding**이다.

### 핵심 대응표

| Regularization 언어 | Bayesian 언어 |
|--------|------|
| $\lambda \|w\|_1$ | Laplace prior $\mathcal{L}(0, 1/\lambda)$의 negative log |
| Sparsity | Prior cusp at 0 → MAP이 coordinate $w_j = 0$을 **강하게** 선호 |
| Soft thresholding | Laplace posterior mode의 closed form |
| $\lambda$ 크다 | Laplace 뾰족함 강화 (scale $1/\lambda$ 작아짐) |

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Laplace 분포

평균 $\mu$, scale $b > 0$의 **Laplace(이중지수) 분포** $\mathcal{L}(\mu, b)$의 density:

$$p(w) = \frac{1}{2b} \exp\left(-\frac{|w - \mu|}{b}\right)$$

$\mu = 0, b = 1/\lambda$로 두면 $p(w) = \frac{\lambda}{2} e^{-\lambda |w|}$. 분산 $= 2b^2$.

### 정의 2.2 — Lasso Problem

$$\hat{w}_{\text{Lasso}} := \arg\min_w \frac{1}{2n} \|y - Xw\|^2 + \lambda \|w\|_1, \quad \|w\|_1 = \sum_{j=1}^d |w_j|$$

### 정리 2.3 — L1 = Laplace Prior MAP (주 정리)

모델 $y|w \sim \mathcal{N}(Xw, \sigma^2 I)$, $w_j \stackrel{\text{iid}}{\sim} \mathcal{L}(0, 1/\tau)$에서:

$$\boxed{\hat{w}_{\text{MAP}} = \arg\min_w \frac{1}{2\sigma^2}\|y - Xw\|^2 + \tau\|w\|_1}$$

Lasso 편의상 $\lambda = \tau \sigma^2 / n$ 또는 scale convention에 맞게 재조정.

### 정의 2.4 — Subdifferential of $|\cdot|$

$f(w) = |w|$는 $w = 0$에서 미분 불가능하지만 convex이므로 subdifferential:

$$\partial |w| = \begin{cases} \{+1\} & w > 0 \\ [-1, +1] & w = 0 \\ \{-1\} & w < 0 \end{cases}$$

다변수에서는 $\partial \|w\|_1 = \prod_j \partial|w_j|$ (각 coordinate별 subdifferential의 곱).

### 정리 2.5 — Soft Thresholding Operator

1차원 문제 $\min_w \frac{1}{2}(z - w)^2 + \lambda |w|$의 해는:

$$\hat{w} = S_\lambda(z) := \text{sign}(z) \max(|z| - \lambda, 0) = \begin{cases} z - \lambda & z > \lambda \\ 0 & |z| \leq \lambda \\ z + \lambda & z < -\lambda \end{cases}$$

**소프트 thresholding** — $|z| \leq \lambda$이면 정확히 **0으로 shrink**한다.

### 정리 2.6 — Coordinate Descent for Lasso

Lasso (정의 2.2)는 coordinate별 update:

$$w_j \leftarrow S_{\lambda/\|X_j\|^2/n}\left( \frac{X_j^T r_j}{\|X_j\|^2} \right), \quad r_j = y - \sum_{k \neq j} X_k w_k$$

가 **수렴**(Lasso는 coordinate-separable한 $\|w\|_1$ + smooth quadratic → block coordinate descent 수렴 보장).

---

## 🔬 수학적 유도

### 정리 2.3 증명

Negative log prior:

$$-\log p(w) = -\sum_{j=1}^d \log\left(\frac{\tau}{2} e^{-\tau|w_j|}\right) = \tau \sum_j |w_j| + d\log(2/\tau) = \tau \|w\|_1 + \text{const}$$

Negative log likelihood (Ch1-01과 동일):

$$-\log p(y|w) = \frac{1}{2\sigma^2}\|y - Xw\|^2 + \text{const}$$

합쳐서 $w$ 독립 상수 제거:

$$\hat{w}_{\text{MAP}} = \arg\min_w \frac{1}{2\sigma^2}\|y - Xw\|^2 + \tau\|w\|_1 \quad \square$$

### 정리 2.5 증명 — 소프트 thresholding

$f(w) = \frac{1}{2}(z - w)^2 + \lambda |w|$. $f$는 convex. 최적성 조건 $0 \in \partial f(\hat{w})$:

$$0 \in \{-(z - \hat{w})\} + \lambda \, \partial |\hat{w}|$$

**Case 1** ($\hat{w} > 0$): $\partial|\hat{w}| = \{1\}$. $z - \hat{w} = \lambda \implies \hat{w} = z - \lambda$. 이 해가 $> 0$이려면 $z > \lambda$.

**Case 2** ($\hat{w} < 0$): 대칭으로 $\hat{w} = z + \lambda$, $z < -\lambda$일 때.

**Case 3** ($\hat{w} = 0$): $\partial|0| = [-1, 1]$, $z \in \lambda[-1, 1]$, 즉 $|z| \leq \lambda$일 때 최적.

세 case를 합치면 $\hat{w} = S_\lambda(z)$. $\square$

**핵심 교훈**: **$|z| \leq \lambda$일 때는 정확히 0** — subdifferential이 구간이기 때문. Gaussian prior의 MAP에서는 이런 "정확히 0" 해가 zero-measure 사건.

### 정리 2.6 유도 — Coordinate Descent

$J(w) = \frac{1}{2n}\|y - Xw\|^2 + \lambda\|w\|_1$. 좌표 $j$만 변화시키며 $w_k (k \neq j)$ 고정. Residual $r_j = y - \sum_{k\neq j} X_k w_k$:

$$J(w_j | w_{-j}) = \frac{1}{2n}\|r_j - X_j w_j\|^2 + \lambda |w_j| + \text{const}$$

$= \frac{\|X_j\|^2}{2n} \left(w_j - \frac{X_j^T r_j}{\|X_j\|^2}\right)^2 + \lambda|w_j| + \text{const}$

정리 2.5의 소프트 thresholding 적용 (scale $\|X_j\|^2/n$ 주의):

$$w_j \leftarrow S_{n\lambda/\|X_j\|^2}\left(\frac{X_j^T r_j}{\|X_j\|^2}\right) \quad \square$$

특히 $X$ column이 $\|X_j\|^2 = n$으로 표준화되어 있으면 더 단순한 $w_j \leftarrow S_\lambda(X_j^T r_j / n)$.

---

## 💻 실험으로 효과 검증

### 실험 1 — 소프트 thresholding 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

z = np.linspace(-3, 3, 400)
for lam in [0.1, 0.5, 1.0, 1.5]:
    plt.plot(z, soft_threshold(z, lam), label=fr'$\lambda={lam}$')
plt.plot(z, z, 'k--', lw=0.7, label='identity (no reg)')
plt.xlabel('z'); plt.ylabel(r'$S_\lambda(z)$')
plt.title('Soft thresholding — $|z| \leq \lambda$에서 정확히 0')
plt.axhline(0, c='gray', lw=0.4); plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

**관찰**: $\lambda$ 구간 내부는 **평평하게 0**, 바깥은 평행 이동.

### 실험 2 — Coordinate descent Lasso

```python
def lasso_cd(X, y, lam, max_iter=500, tol=1e-6):
    n, d = X.shape
    w = np.zeros(d)
    col_sq = (X**2).sum(axis=0)
    for it in range(max_iter):
        w_old = w.copy()
        for j in range(d):
            r_j = y - X @ w + X[:, j] * w[j]
            z_j = X[:, j] @ r_j
            thr = n * lam / col_sq[j]
            w[j] = soft_threshold(z_j / col_sq[j], thr)
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w, it

np.random.seed(0)
n, d = 60, 30
X = np.random.randn(n, d) / np.sqrt(n)
w_true = np.zeros(d); w_true[:5] = [2, -1.5, 1, -0.8, 0.5]
y = X @ w_true + 0.1 * np.random.randn(n)

w_hat, iters = lasso_cd(X, y, lam=0.05)
print(f"converged in {iters} iters")
print("nonzero coords:", np.where(np.abs(w_hat) > 1e-6)[0])
# → 0..4번 coordinate 근처만 nonzero (true signal 복원)
```

### 실험 3 — Lasso path + Laplace density overlay

```python
from scipy.stats import laplace
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# (왼쪽) Laplace vs Gaussian density
w_grid = np.linspace(-3, 3, 500)
ax[0].plot(w_grid, laplace.pdf(w_grid, scale=1), label='Laplace(0, 1)')
ax[0].plot(w_grid, np.exp(-w_grid**2/2)/np.sqrt(2*np.pi),
           label='Gaussian(0, 1)')
ax[0].set_title('Laplace의 0에서 cusp (Gaussian은 smooth)')
ax[0].legend(); ax[0].grid(alpha=0.3)

# (오른쪽) Lasso path
lams = np.logspace(-3, 0, 40)
path = np.array([lasso_cd(X, y, lam)[0] for lam in lams])
for j in range(d):
    ax[1].semilogx(lams, path[:, j], alpha=0.6)
ax[1].axhline(0, c='k', lw=0.4)
ax[1].set_xlabel(r'$\lambda$'); ax[1].set_ylabel('coef')
ax[1].set_title('Lasso path — λ↑일수록 coefficient가 정확히 0으로')
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 🔗 실전 활용

### 언제 L1(Lasso)을 선택하는가

1. **Feature selection이 목표**: 실제로 **0이 되는** coefficient가 필요할 때. Gene expression, NLP의 BoW feature, 구조화 sparse 문제.
2. **해석 가능성**: 모델이 **어떤 feature를 사용하지 않는지** 명시적으로 보고 싶을 때.
3. **High-dimensional $d \gg n$**: Ridge도 작동하지만 "실제 활성 feature 수 $\ll d$"라는 prior가 맞으면 Lasso가 더 정확.

### Lasso의 약점과 Elastic Net

- Lasso는 상관 feature 그룹 중 **하나만** 뽑는 경향(Zou-Hastie 2005).
- 해결: Elastic Net $\lambda_1\|w\|_1 + \lambda_2\|w\|^2$ (Ch1-05).

### 딥러닝에서의 L1

- Weight에 직접 L1을 주면 "weight sparsity" — 훈련 후 pruning의 근사.
- 훨씬 자주 쓰이는 형태는 **activation sparsity** (e.g. `F.relu` + L1 activity regularization), **group sparsity** (L2,1 norm)로 채널 전체 제거.
- Lottery Ticket Hypothesis([Generalization Theory Deep Dive Ch6](https://github.com/iq-ai-lab/generalization-theory-deep-dive))와의 연결 — magnitude pruning도 일종의 L1-driven selection.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Laplace prior가 sparsity의 "정답" | 실제 $w$가 **정확히 0**인 coordinates가 많지 않으면 Lasso가 over-sparsify |
| iid prior $p(w) = \prod_j p(w_j)$ | feature 간 구조(group, graph)를 반영 못함 → Group Lasso로 확장 |
| Single $\lambda$ | 사실상 feature별 $\lambda_j$가 더 정확할 수 있음 (adaptive Lasso) |
| Convex subdifferential 기반 최적화 | non-convex SCAD, MCP 같은 "덜 biased" 대안이 있음 |
| Linear model | NN에서 L1은 weight sparsity를 제한적으로만 달성 |

**주의**: Lasso의 MAP은 posterior mode만 반환한다. Posterior는 여전히 Laplace가 아닌 복잡한 분포 — "Bayesian Lasso" (Park-Casella 2008)는 이를 MCMC로 다룬다.

---

## 📌 핵심 정리

$$\boxed{\hat{w}_{\text{Lasso}} = \arg\min \tfrac{1}{2n}\|y-Xw\|^2 + \lambda\|w\|_1 = \hat{w}_{\text{MAP}} \text{ under Laplace prior}}$$

| 개념 | 의미 |
|------|------|
| **L1 regularization** | Laplace prior의 negative log |
| **Sparsity** | Laplace 원점 cusp → subdifferential이 구간 → 정확히 0 해 가능 |
| **Soft thresholding $S_\lambda$** | 1D Lasso의 closed form, $\|z\| \leq \lambda$에서 0 |
| **Coordinate descent** | Lasso의 수렴 보장 알고리즘 |
| **다음 질문** | 기하적으로는 왜? → L1 ball의 꼭짓점 → Ch1-03 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $z = 0.7, \lambda = 0.3$에서 $S_\lambda(z) = ?$. $\lambda = 1.0$이면?

<details>
<summary>힌트 및 해설</summary>

- $\lambda = 0.3$: $|z| = 0.7 > 0.3$이므로 $S_\lambda(z) = \text{sign}(0.7)(0.7 - 0.3) = 0.4$.
- $\lambda = 1.0$: $|z| = 0.7 \leq 1.0$이므로 **정확히 0**.

이 단순 1D 예에서 "$|z| \leq \lambda$이면 정확히 0"이라는 Lasso의 핵심 메커니즘이 드러난다.

</details>

**문제 2** (심화): Laplace(0, 1)과 Gaussian(0, 2)는 둘 다 분산 2를 갖는다. MAP 관점에서 두 prior는 무엇이 다른가? "원점 질량"이라는 관점을 사용하라.

<details>
<summary>힌트 및 해설</summary>

분산이 같아도 **density 모양**이 완전히 다르다. Laplace는 원점에 뾰족한 peak(density $\lambda/2$), Gaussian은 매끄러운 bell. MAP은 posterior **mode**를 선택하므로:

- Gaussian prior: likelihood가 $w_j$를 0에서 아주 약간만 끌어당겨도 mode가 0 근방의 "매끄러운 valley"에 떨어진다 — shrinkage만.
- Laplace prior: cusp의 날카로운 "끌개"가 **정확히 0에 붙는** 해를 좋아한다 — sparsity.

이것이 같은 분산이어도 sparsity의 유무를 결정하는 이유. Bayes 관점에서는 "prior의 **꼬리와 peak shape**이 regularization의 종류를 결정한다"로 요약.

</details>

**문제 3** (이론-실전): Adaptive Lasso (Zou 2006)는 $\lambda_j = \lambda / |\hat{w}_j^{\text{init}}|^\gamma$로 coordinate별 $\lambda$를 둔다. Bayesian 관점에서 이는 어떤 prior에 대응하는가? Oracle property를 간단히 서술하라.

<details>
<summary>힌트 및 해설</summary>

Adaptive Lasso는 feature별 Laplace prior의 **scale 파라미터**를 weight magnitude에 따라 다르게 주는 것 ($b_j = 1/\lambda_j$). 큰 $|\hat{w}_j^{\text{init}}|$에 대해서는 $\lambda_j$가 작아져 거의 unpenalized, 작은 것에 대해서는 강한 L1.

**Oracle property** (Zou 2006): 적절한 $\gamma > 0$에서 adaptive Lasso는 $n \to \infty$에서 (a) 진짜 active set를 정확히 식별하고 (b) non-zero coefficient는 oracle LS와 같은 asymptotic 분포를 갖는다. 일반 Lasso는 (b)를 보장하지 않는다 (coefficient가 bias됨).

이는 "모든 feature를 같은 prior로"보다 "사전 정보를 써서 feature별 prior 강도를 조정"하는 것이 **통계적으로** 더 강력함을 보인다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. L2 = Gaussian Prior MAP](./01-l2-gaussian-prior.md) | [📚 README로 돌아가기](../README.md) | [03. Sparsity의 기하학 ▶](./03-sparsity-geometry.md) |

</div>
