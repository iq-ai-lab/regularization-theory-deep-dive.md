# 03. Sparsity의 기하학과 KKT

## 🎯 핵심 질문

- **L1 ball**과 **L2 ball**은 왜 기하적으로 다른가?
- 왜 **꼭짓점에서 loss contour가 접할 확률이 1**인가?
- 제약형 $\min \|y - Xw\|^2 \text{ s.t. } \|w\|_1 \leq t$의 **KKT 조건**은 어떻게 sparse coordinate를 강제하는가?
- Lagrangian form과 constraint form은 어떻게 동치인가?

---

## 🔍 왜 기하적 관점이 필요한가

Ch1-02는 Laplace prior의 **cusp**라는 확률적 설명을 주었다. 같은 sparsity 현상을 **기하적**으로 보는 것은 두 가지 이익이 있다.

1. **시각적 직관**: L1 ball의 "다이아몬드 꼭짓점"이 왜 feature selection을 일으키는지 2차원 그림으로 곧바로 이해.
2. **일반화 가능성**: L1·L2 말고도 Group Lasso (L2,1 norm), nuclear norm (matrix spectral), total variation 등 다양한 regularizer의 기하를 **같은 KKT 프레임**으로 분석. 이는 Ch1-05의 확장으로 이어진다.

**핵심 통찰**: Sparsity는 **원점의 non-smoothness에서 나온다**. L1의 꼭짓점, Group Lasso의 모서리, nuclear norm의 낮은 rank 표면 — 모두 "non-smooth set"의 한 표현.

---

## 📐 수학적 선행 조건

- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): convex set, extreme point, **KKT 조건**, Lagrangian duality
- Ch1-02: Soft thresholding, subdifferential
- 선형대수: ellipsoid의 정의 $\{w : w^T A w \leq c\}$, gradient의 기하적 의미(등고선에 수직)

---

## 📖 직관적 이해

### L1 ball vs L2 ball

2차원에서 $t = 1$ 크기의 두 ball:

| Norm | 집합 | 모양 |
|------|------|------|
| $\|w\|_1 \leq 1$ | $|w_1| + |w_2| \leq 1$ | **다이아몬드 (회전 45°의 정사각형)** |
| $\|w\|_2 \leq 1$ | $w_1^2 + w_2^2 \leq 1$ | **원** |

- L1 ball은 **꼭짓점 4개**와 모서리에서 non-smooth. 꼭짓점 위치: $(\pm 1, 0), (0, \pm 1)$ — **정확히 한 coordinate만 non-zero**.
- L2 ball은 **어디서나 smooth**. 모든 boundary point가 equivalent.

### 등고선과 ball의 접점

제약 최적화 $\min \|y - Xw\|^2 \text{ s.t. } \|w\|_p \leq t$는:

1. Loss contour(타원, $X^TX$에 의한 이차형)를 그린다.
2. 그중 ball과 **접하는** 가장 작은 contour를 찾는다.
3. 접점이 해.

**L2 ball**: 접점은 원 위 **어느 곳이나** 가능. 일반적으로 모든 coordinate가 nonzero.

**L1 ball**: 접점이 **꼭짓점** 위치에 "끌린다" — 꼭짓점에서는 다양한 방향의 normal cone이 있어서 generic 타원이 꼭짓점과 접하기 **쉽다**. 꼭짓점은 한 coordinate만 nonzero이므로 **sparse solution**.

### "생성 확률 1"의 의미

3차원 이상에서도 마찬가지로 L1 ball의 **저차원 면**(꼭짓점, 모서리, faces)에서 접할 확률이 양(positive). Lebesgue 측도 관점에서 **generic**한 loss ellipsoid에 대해 sparse 해가 나온다.

**비유**: 사각 탁자 위에 동전을 랜덤하게 던진다. 꼭짓점에 아슬아슬하게 걸리기보다 "면" 위에 떨어질 것 같지만, "L1 최적화"는 동전이 탁자와 **닿는 최고 지점**을 묻는다. 탁자가 평평하지 않고 **꼭짓점 방향으로 부풀어** 있으면 꼭짓점에 닿게 된다.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Lagrangian and Constraint Forms

두 동치적 최적화 문제:

$$\text{(Lag)} \quad \min_w \tfrac{1}{2}\|y - Xw\|^2 + \lambda \|w\|_1$$

$$\text{(Con)} \quad \min_w \tfrac{1}{2}\|y - Xw\|^2 \quad \text{s.t. } \|w\|_1 \leq t$$

### 정리 3.2 — Lagrangian-Constraint 동치성

모든 $\lambda > 0$에 대해, 해당 $t(\lambda) \geq 0$이 존재하여 (Lag)과 (Con)의 **해가 같다**. 역도 성립. 이 대응은 convex duality로 엄밀히 보장된다.

### 정리 3.3 — KKT 조건 for Constraint Form

$f(w) = \tfrac{1}{2}\|y-Xw\|^2$와 제약 $g(w) = \|w\|_1 - t \leq 0$의 KKT:

$$\begin{aligned}
&\text{Stationarity: } \nabla f(\hat{w}) + \mu \cdot v = 0, \ v \in \partial \|\hat{w}\|_1 \\
&\text{Primal feasibility: } \|\hat{w}\|_1 \leq t \\
&\text{Dual feasibility: } \mu \geq 0 \\
&\text{Complementary slackness: } \mu (\|\hat{w}\|_1 - t) = 0
\end{aligned}$$

### 정리 3.4 — Sparse Coordinate Condition

위 KKT에서 $\mu > 0$이고 $\hat{w}_j = 0$이면:

$$|X_j^T (y - X\hat{w})| \leq \mu$$

즉 **residual과 feature의 inner product가 작을수록 그 feature는 selected되지 않는다**. 이 조건은 Lasso의 "feature 활성화 임계치"를 정식화한다.

### 정리 3.5 — L1 Ball Extreme Points and Sparsity

$d$차원 L1 ball $B_1(t) = \{w : \|w\|_1 \leq t\}$의 **extreme points**(꼭짓점)는 정확히 $2d$개:

$$\{\pm t \cdot e_j : j = 1, \ldots, d\}$$

여기서 $e_j$는 $j$번째 표준 기저 벡터. **각 꼭짓점은 정확히 하나의 coordinate만 nonzero**.

### 정리 3.6 — Generic Uniqueness of Sparse Lasso Solutions

$X$의 column이 일반 위치(general position, any $k \leq n$ columns linearly independent) 하에서, Lasso 해는 **unique**이며 $|\text{supp}(\hat{w})| \leq n$.

---

## 🔬 수학적 유도

### 정리 3.4 유도 — Sparsity로부터의 조건

KKT stationarity $X^T(X\hat{w} - y) + \mu v = 0$에서 $v \in \partial \|\hat{w}\|_1$. Coordinate $j$별로:

$$X_j^T(X\hat{w} - y) = -\mu v_j, \quad v_j \in \begin{cases} \{\text{sign}(\hat{w}_j)\} & \hat{w}_j \neq 0 \\ [-1, 1] & \hat{w}_j = 0 \end{cases}$$

**$\hat{w}_j = 0$인 경우**: $v_j \in [-1, 1]$이므로

$$|X_j^T(X\hat{w} - y)| \leq \mu \quad \square$$

이는 **feature $j$의 residual correlation이 $\mu$ 이하이면 active set에 포함되지 않는다**는 의미. $\mu$는 Lagrange 승수 $= \lambda$(Lagrangian form).

### 왜 꼭짓점에서 접하는가 — 기하적 주장

Loss ellipsoid $E_c = \{w : \|y - Xw\|^2 = c\}$의 level set. 최적해는 "ball과 접하는 가장 작은 $c$"의 ellipsoid의 접점:

1. **Smooth boundary (L2 ball)**: ellipsoid와 sphere의 접선이 **1차원** 공간. 접점이 coordinate 축 위에 있을 확률은 0 (measure-zero).
2. **Non-smooth boundary (L1 ball)**: 꼭짓점에서 $\partial B_1(t)$의 **normal cone**이 **full-dimensional** (여러 방향을 포함). 따라서 **양의 확률**로 loss gradient가 이 cone 안에 있어 꼭짓점이 최적.

### 정리 3.6 증명 스케치

Osborne, Presnell, Turlach 2000: Lasso KKT는 active set $A = \text{supp}(\hat{w})$를 고정하면 non-active coord는 $|X_j^T r| \leq \lambda$로 고정. active coord들은 sign $s_A$가 정해지면 $X_A^T X_A w_A = X_A^T y - \lambda s_A$로 unique하게 결정 (general position 가정으로 $X_A^T X_A$ invertible, $|A| \leq n$).

따라서 $(A, s_A)$가 결정되면 해가 unique. 서로 다른 $(A, s_A)$가 같은 해를 주는 것은 $X$의 특수 구성에서만 일어나며 일반 위치에서는 제외된다.

---

## 💻 실험으로 효과 검증

### 실험 1 — 2D L1 vs L2 ball 접선 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# 참 loss 타원: (w - w*)^T A (w - w*) 형태
w_star = np.array([0.8, 0.6])
A = np.array([[2, 0.5], [0.5, 1]])

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, norm_name in zip(axes, ['L2', 'L1']):
    # ball
    theta = np.linspace(0, 2*np.pi, 400)
    if norm_name == 'L2':
        ball_x, ball_y = np.cos(theta), np.sin(theta)
    else:
        ball_x = np.sign(np.cos(theta)) * (np.abs(np.cos(theta)))
        # L1 ball: |x|+|y|=1 의 parametric 대안
        ball_x = np.concatenate([[1,0,-1,0,1], [1]])
        ball_y = np.concatenate([[0,1,0,-1,0], [0]])
    ax.plot(ball_x, ball_y, 'b-', lw=2, label=f'{norm_name} ball')

    # 여러 크기의 loss 타원
    w1, w2 = np.meshgrid(np.linspace(-1.8, 1.8, 200), np.linspace(-1.8, 1.8, 200))
    Z = A[0,0]*(w1-w_star[0])**2 + 2*A[0,1]*(w1-w_star[0])*(w2-w_star[1]) + A[1,1]*(w2-w_star[1])**2
    ax.contour(w1, w2, Z, levels=[0.3, 0.8, 1.5, 2.5], colors='r', alpha=0.5)

    ax.plot(*w_star, 'r*', ms=15, label='unconstrained min')
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect('equal')
    ax.axhline(0, c='gray', lw=0.4); ax.axvline(0, c='gray', lw=0.4)
    ax.set_title(f'{norm_name} constrained minimum')
    ax.legend()

plt.suptitle('L2는 smooth 접선 / L1은 꼭짓점에 끌림 → sparse')
plt.tight_layout(); plt.show()
```

**관찰**: Loss 타원을 서서히 줄여보면 L1 ball에서는 꼭짓점 $(1, 0)$ 혹은 $(0, 1)$에 먼저 닿는 경우가 많다(타원 방향에 따라).

### 실험 2 — KKT 조건 수치적 확인

```python
from numpy.linalg import solve

np.random.seed(1)
n, d = 40, 20
X = np.random.randn(n, d) / np.sqrt(n)
w_true = np.zeros(d); w_true[:5] = [1.5, -1, 0.8, 0.5, -0.3]
y = X @ w_true + 0.1 * np.random.randn(n)

from scipy.optimize import minimize
def lasso_obj(w, X, y, lam):
    return 0.5 * np.sum((y - X @ w)**2) + lam * np.sum(np.abs(w))

# subgradient-aware 최적화는 복잡하므로 scikit-learn 사용 권장이지만
# 여기서는 개념 확인용: 앞선 문서의 coordinate descent 재사용
def soft_thr(z, lam): return np.sign(z) * np.maximum(np.abs(z) - lam, 0)
def lasso_cd(X, y, lam, iters=500):
    w = np.zeros(X.shape[1]); col2 = (X**2).sum(axis=0)
    for _ in range(iters):
        for j in range(X.shape[1]):
            r = y - X @ w + X[:, j] * w[j]
            w[j] = soft_thr(X[:, j] @ r / col2[j], lam / col2[j])
    return w

lam = 0.05
w_hat = lasso_cd(X, y, lam)

# KKT 확인
residual = y - X @ w_hat
grad = X.T @ residual              # 이것이 정확성 조건으로 |grad_j| <= lam (w_j=0일 때)
print("nonzero indices:", np.where(np.abs(w_hat) > 1e-6)[0])
print("\ncoord j | w_j      | |X_j^T r|  | test")
for j in range(d):
    status = "active" if abs(w_hat[j]) > 1e-6 else "inactive"
    satisfied = "✓" if (abs(w_hat[j]) > 1e-6) or (abs(grad[j]) <= lam + 1e-4) else "✗"
    print(f"  {j:3d}  | {w_hat[j]:+.4f} | {abs(grad[j]):.4f}    | {status} {satisfied}")
# → 모든 inactive coord에서 |X_j^T r| <= lambda (KKT 검증)
```

### 실험 3 — 3D L1 ball의 extreme points 수 확인

```python
# d=3일 때 extreme points: ±e1, ±e2, ±e3 → 6개
# d=d일 때 2d개 (정리 3.5)
for d in [2, 3, 5, 10]:
    # 꼭짓점은 표준 기저의 ±
    n_vertices = 2 * d
    # L1 ball의 volume = 2^d / d! (for radius 1)
    vol = 2**d / np.math.factorial(d)
    print(f"d={d:2d}: 꼭짓점 수={n_vertices}, L1 unit ball volume={vol:.6f}")
# → 차원이 증가해도 꼭짓점 수는 선형 증가, volume은 exponentially 감소 (dense cluster가 0 주변)
```

---

## 🔗 실전 활용

### 기하 관점이 주는 실전 가이드

1. **Feature standardization**: 기하 해석은 모든 feature가 "공평하게 경쟁"하는 상황을 가정. 스케일이 다른 feature는 먼저 표준화(z-score) 해야 L1 ball의 기하가 의미 있음.
2. **Solution path의 piecewise linear 성질**: LARS (Least Angle Regression, Efron 2004)는 Lasso path가 $\lambda$에 대해 piecewise linear임을 이용해 전체 path를 O($n d \min(n, d)$)에 계산.
3. **Correlated features**: 기하적으로 상관된 두 feature는 ellipsoid의 한 축이 길어진 경우 → L1 ball의 꼭짓점 중 하나만 선택되기 쉬움 (Elastic Net이 이를 완화, Ch1-05).

### Group Lasso의 기하

Group Lasso $\sum_g \|w_g\|_2$의 제약 집합은 "여러 L2 ball의 합성체"로 구성 — 각 group의 모서리가 "한 group 전체가 0"이 되는 sparse face. 이 일반화된 non-smoothness가 group-level sparsity를 만든다.

### Nuclear Norm의 기하 (저차원 rank)

행렬 $W$에 대해 $\|W\|_* = \sum_i \sigma_i$ (singular value 합) constraint는 **spectral simplex** 모양으로, rank-deficient 행렬이 corner가 된다 — matrix completion의 low-rank 해가 나오는 원리.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Generic loss ellipsoid | 특수하게 정렬된 문제에서는 L1도 sparse 안 줄 수 있음 |
| Feature 표준화 | 비표준화 데이터에서 기하 직관은 왜곡됨 |
| Convex (L1, L2) | Non-convex penalties (SCAD, MCP)는 더 aggressive sparsity, 하지만 KKT 확장 필요 |
| $p$-norm 관점 | $0 < p < 1$ quasi-norm은 non-convex이지만 더 sparse |
| Unique solution | correlated feature에서 Lasso 해는 set(다중해) 가능 |

---

## 📌 핵심 정리

$$\boxed{\text{L1 ball의 꼭짓점 = 1-sparse point } \implies \text{ generic ellipsoid가 꼭짓점에 접할 확률 > 0} \implies \text{sparsity}}$$

| 개념 | 의미 |
|------|------|
| **L1 ball** | 꼭짓점 $2d$개가 표준 기저 위 — 1-sparse |
| **L2 ball** | Smooth — 어디서나 접할 수 있음, sparsity 없음 |
| **KKT stationarity** | $\|X_j^T(y-X\hat{w})\| \leq \lambda$ 이면 $\hat{w}_j = 0$ |
| **Lag ↔ Con 동치** | $\lambda \leftrightarrow t(\lambda)$ 일대일 |
| **다음 질문** | L2는 무슨 일을 하는가 — shrinkage의 spectral 분석 → Ch1-04 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $d = 3$에서 L1 unit ball의 꼭짓점·모서리·면의 개수를 각각 구하라.

<details>
<summary>힌트 및 해설</summary>

$\|w\|_1 \leq 1$은 3D에서 **정팔면체 (octahedron)**. Euler 공식 $V - E + F = 2$:

- **꼭짓점(V)**: $2d = 6$ ($\pm e_1, \pm e_2, \pm e_3$).
- **모서리(E)**: 12. 각 꼭짓점의 "반대 부호" 아닌 다른 축 꼭짓점들과 연결 (4 connections per vertex ÷ 2).
- **면(F)**: 8. 각 "octant"(부호 조합)마다 한 면.

$V - E + F = 6 - 12 + 8 = 2$ ✓.

이 순서 — 꼭짓점(1-sparse), 모서리(2-sparse face 위), 면(dense 내부) — 이 높은 차원에서는 $k$-sparse face의 수가 $\binom{d}{k} 2^k$로 주어진다.

</details>

**문제 2** (심화): $d = 2$에서 loss가 타원 $(w_1 - a)^2 + c(w_2 - b)^2 = C$인 경우, 어떤 조건에서 Lasso의 해가 정확히 꼭짓점 $(t, 0)$이 되는가?

<details>
<summary>힌트 및 해설</summary>

꼭짓점 $(t, 0)$에서의 normal cone은 $\{(u_1, u_2) : u_1 \geq |u_2|\}$ (L1 ball의 경우). Loss gradient $\nabla f = (2(w_1 - a), 2c(w_2 - b))$를 $(t, 0)$에 대입하면 $(2(t - a), -2cb)$.

KKT: $-\nabla f \in \lambda \cdot \partial\|\hat{w}\|_1$. 즉 $(2(a - t), 2cb)$가 normal cone 안에 있어야 함:

1. $(a - t) > 0$ 즉 $a > t$ (active constraint 방향).
2. $|2cb| \leq 2(a - t) \iff c |b| \leq a - t$.

이 두 조건이 만족되면 꼭짓점 $(t, 0)$이 해. 일반적으로 $(a, b)$가 "L1-boundary를 지나 $w_1$-축에 가까운" 영역에 있으면 그 축의 꼭짓점이 해.

</details>

**문제 3** (이론-실전): Ridge ($L^2$) 해 $\hat{w}_R = (X^TX + \lambda I)^{-1}X^T y$는 sparse하지 않다. 이것을 "L2 ball 기하"로 설명하고, Lasso로 바꾸면 sparsity가 생기는 근본 이유를 한 문장으로 요약하라.

<details>
<summary>힌트 및 해설</summary>

L2 ball은 boundary의 **모든 점에서 smooth** — normal cone이 1차원(sphere의 접선 normal). 따라서 generic loss ellipsoid가 boundary에 접할 때 접점은 "아무 위치"이며, coordinate 축 위에 있을 확률은 zero(Lebesgue measure 0). 결과: 모든 coordinate가 nonzero.

**한 문장 요약**: "Sparsity는 regularizer의 **원점 근처 non-smoothness**에서 나온다 — L1의 꼭짓점이 그 non-smooth set".

이 통찰은 group sparsity(Group Lasso), low-rank(nuclear norm), piecewise constant(total variation) 등 다양한 sparsity 개념을 **같은 원리**로 통일한다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. L1 = Laplace Prior MAP](./02-l1-laplace-prior.md) | [📚 README로 돌아가기](../README.md) | [04. Ridge의 SVD Shrinkage ▶](./04-ridge-svd-shrinkage.md) |

</div>
