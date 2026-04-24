# 05. Elastic Net과 Group Lasso

## 🎯 핵심 질문

- Lasso의 "상관 feature 그룹 중 하나만" 문제를 Elastic Net은 어떻게 해결하는가?
- **Group Lasso**는 feature-group 단위 sparsity를 어떻게 강제하는가?
- **Proximal gradient (ISTA/FISTA)**로 이 문제들을 어떻게 푸는가?
- 이 기법들은 각각 어떤 Bayesian prior에 대응하는가?

---

## 🔍 왜 확장이 필요한가

Ch1-02·03에서 본 Lasso는 두 가지 근본 약점이 있다.

1. **상관 feature의 임의 선택** (Zou-Hastie 2005): feature $x_1, x_2$가 거의 동일할 때 Lasso는 **둘 중 하나만** 선택하며 그 선택은 사실상 **noise에 따라** 결정된다. 해석·안정성 모두에서 나쁨.
2. **구조 정보 무시**: gene expression에서 같은 pathway의 유전자, NLP에서 같은 topic의 단어, one-hot encoded categorical의 한 변수 등은 "**함께 선택되거나 함께 제외되어야**" 한다. Lasso는 이를 강제하지 못한다.

**Elastic Net**은 (1)을, **Group Lasso**는 (2)를 해결한다. 두 기법 모두 Bayesian으로 여전히 prior로 해석 가능하며 (Spike-and-slab 근사, 그룹별 Laplace prior), proximal gradient로 효율적으로 풀린다.

이 문서는 Ch1의 마무리로서 "L1/L2라는 두 축을 **조합·구조화**하면 어떤 관리자 기법이 탄생하는가"를 보여준다.

---

## 📐 수학적 선행 조건

- Ch1-02: Soft thresholding, coordinate descent
- Ch1-03: L1 ball 기하, KKT
- Ch1-04: Ridge의 spectral 효과
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): **proximal operator** $\text{prox}_f(z) = \arg\min_w \tfrac{1}{2}\|w-z\|^2 + f(w)$, Nesterov acceleration

---

## 📖 직관적 이해

### Elastic Net: L1 + L2의 장점 결합

$$\min_w \tfrac{1}{2n}\|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|^2$$

- **L1 부분**: sparsity 제공 (Ch1-02).
- **L2 부분**: 상관 feature들의 **coefficient를 고르게 나눔** — 둘 다 선택 (grouping effect).

**비유**: Lasso는 "한 명만 뽑는 선거", Elastic Net은 "비례대표". 유사한 후보들이 **공동 지지**를 받을 수 있다.

### Group Lasso: 그룹 단위 sparsity

Feature를 $G$개 그룹으로 분할: $w = (w_{g_1}, w_{g_2}, \ldots, w_{g_G})$. Group Lasso:

$$\min_w \tfrac{1}{2n}\|y - Xw\|^2 + \lambda \sum_g \sqrt{p_g} \|w_g\|_2$$

- 각 그룹 내부는 **L2 norm** — smooth.
- 그룹 단위는 **L1 norm of L2 norms** — 그룹 전체가 0이거나, 그룹 전체가 나란히 nonzero.

**직관**: "각 그룹을 **하나의 slot**으로 간주하고, 그 slot이 전체로 on/off". 결과적으로 "**active group의 sparse 선택**".

### 기하적 그림

- **Elastic Net ball**: L1 ball과 L2 ball의 볼록 조합 — 꼭짓점이 "둥글어진" 형태. 여전히 non-smooth(sparsity) + smoother edges(grouping).
- **Group Lasso ball**: 각 그룹별로 L2 ball, 그 위에 L1 제약 — "ball of balls" 모양. 그룹 경계가 low-dim face = group sparse.

### Bayesian 대응표

| 기법 | Prior |
|------|------|
| Ridge | Gaussian $\mathcal{N}(0, \sigma_w^2 I)$ |
| Lasso | iid Laplace $\mathcal{L}(0, 1/\lambda)$ |
| Elastic Net | Gaussian scale mixture — closed form 대응 complicated |
| Group Lasso | Group-wise multivariate Laplace — 각 그룹이 radially symmetric |
| Spike-and-slab | $w_j \sim \pi \delta_0 + (1-\pi) \mathcal{N}(0, \tau^2)$ — 정확한 sparsity prior |

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Elastic Net Problem (Zou & Hastie 2005)

$$\hat{w}_{\text{EN}} = \arg\min_w \frac{1}{2n}\|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|^2$$

### 정리 5.2 — Elastic Net의 Ridge 변형으로의 Reduction

Elastic Net은 augmented problem으로 Lasso에 **reduction** 가능. $X_{\text{aug}} = \tfrac{1}{\sqrt{1+\lambda_2}}\begin{pmatrix} X \\ \sqrt{n\lambda_2} I \end{pmatrix}$, $y_{\text{aug}} = \begin{pmatrix} y \\ 0 \end{pmatrix}$일 때:

$$\hat{w}_{\text{EN}} \cdot \sqrt{1+\lambda_2} = \arg\min \tfrac{1}{2n}\|y_{\text{aug}} - X_{\text{aug}} w\|^2 + \frac{\lambda_1}{\sqrt{1+\lambda_2}} \|w\|_1$$

이는 Elastic Net에 대한 **모든 Lasso 알고리즘** 적용 가능함을 보인다.

### 정리 5.3 — Elastic Net의 Coordinate Update

$$w_j \leftarrow \frac{S_{\lambda_1}(X_j^T r_j / n)}{\|X_j\|^2/n + 2\lambda_2}$$

### 정의 5.4 — Group Lasso Problem (Yuan & Lin 2006)

Feature index를 disjoint groups $\{g_1, \ldots, g_G\}$로 분할, $X_g$는 group $g$의 column. Group weight $\sqrt{p_g}$ ($p_g = |g|$로 그룹 크기):

$$\hat{w}_{\text{GL}} = \arg\min_w \frac{1}{2n}\|y - Xw\|^2 + \lambda \sum_{g=1}^G \sqrt{p_g} \|w_g\|_2$$

### 정리 5.5 — Block Soft Thresholding (Group Proximal Operator)

$\min_w \tfrac{1}{2}\|z - w\|^2 + \lambda \|w\|_2$의 해:

$$\text{prox}_{\lambda \|\cdot\|_2}(z) = \left(1 - \frac{\lambda}{\|z\|_2}\right)_+ z$$

여기서 $(\cdot)_+ = \max(\cdot, 0)$. 즉 **전체 벡터가 같은 비율로 shrink**, $\|z\|_2 \leq \lambda$면 **벡터 전체가 0**.

### 정리 5.6 — Group Lasso KKT

$\hat{w}_g = 0$이면 반드시 $\|X_g^T(y - X\hat{w})/n\|_2 \leq \lambda \sqrt{p_g}$. 그룹 residual이 작을수록 그 그룹은 deactivated.

### 정리 5.7 — Proximal Gradient (ISTA)

Convex objective $F(w) = f(w) + g(w)$ ($f$ smooth, $g$ non-smooth convex)에서:

$$w^{(t+1)} = \text{prox}_{\eta g}\left(w^{(t)} - \eta \nabla f(w^{(t)})\right)$$

Lasso: $g = \lambda\|\cdot\|_1$, prox = soft thresholding.  
Group Lasso: $g = \lambda \sum \|w_g\|_2$, prox = block soft thresholding (정리 5.5, 그룹별).

**FISTA** (Beck-Teboulle 2009)는 Nesterov momentum으로 $O(1/t^2)$ 수렴 (ISTA는 $O(1/t)$).

---

## 🔬 수학적 유도

### 정리 5.3 유도 (coordinate update)

$w_j$만 변화, 나머지 고정:

$$J(w_j|w_{-j}) = \frac{1}{2n}\|r_j - X_j w_j\|^2 + \lambda_1 |w_j| + \lambda_2 w_j^2 + \text{const}$$

$= \left(\frac{\|X_j\|^2}{2n} + \lambda_2\right) w_j^2 - \frac{X_j^T r_j}{n} w_j + \lambda_1|w_j| + \text{const}$

이차항 계수 $a = \|X_j\|^2/n + 2\lambda_2$ (factor 2 주의: $\lambda_2 w_j^2$의 derivative는 $2\lambda_2 w_j$), 1차 계수 $b = -X_j^T r_j / n$. Soft threshold은 $|w_j|$의 coefficient $\lambda_1$:

$$w_j = S_{\lambda_1/a}(-b/a) = \frac{S_{\lambda_1}(X_j^T r_j/n)}{a} \quad \square$$

### 정리 5.5 유도 — Block Soft Thresholding

$\min_w \tfrac{1}{2}\|z-w\|^2 + \lambda\|w\|_2$.

**Case 1** ($w \neq 0$): $\nabla = 0 \implies w - z + \lambda w/\|w\|_2 = 0 \implies w(1 + \lambda/\|w\|_2) = z$.

$\|w\|_2 = \|z\|_2 - \lambda$ (양변 norm 취하고 풀면), 이 해가 $> 0$이려면 $\|z\|_2 > \lambda$. 이때:

$$w = \frac{\|z\|_2 - \lambda}{\|z\|_2} z = \left(1 - \frac{\lambda}{\|z\|_2}\right) z$$

**Case 2** ($w = 0$): $0 \in -\{z\} + \lambda \partial \|0\|_2 = -\{z\} + \lambda B_2(1)$. $z \in \lambda B_2(1) \iff \|z\|_2 \leq \lambda$.

합쳐 쓰면 $\text{prox}(z) = (1 - \lambda/\|z\|_2)_+ z$. $\square$

**핵심**: 1D에서는 $S_\lambda(z) = \text{sign}(z)(|z|-\lambda)_+$, group으로 확장하면 **방향은 유지하면서 magnitude만 shrink**. 그룹 전체가 직교 단위 벡터 스타일로 shrink되거나 완전히 0.

### Group Lasso의 ISTA update

$w_g^{(t+1)} = \text{prox}_{\eta\lambda\sqrt{p_g}\|\cdot\|_2}(w_g^{(t)} - \eta \nabla_{w_g} \text{loss})$

각 iteration마다 먼저 smooth gradient step, 그다음 각 group별 block soft threshold.

---

## 💻 실험으로 효과 검증

### 실험 1 — Elastic Net의 grouping effect

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet

np.random.seed(0)
n, d = 60, 10
# 첫 5개 feature는 거의 완전히 상관 (grouped signal)
base = np.random.randn(n, 1)
X = np.hstack([base + 0.05 * np.random.randn(n, 1) for _ in range(5)] +
              [np.random.randn(n, 1) for _ in range(5)])

w_true = np.zeros(d); w_true[:5] = 0.4    # 5개 상관 feature 모두 0.4
y = X @ w_true + 0.1 * np.random.randn(n)

lasso = Lasso(alpha=0.02).fit(X, y)
enet  = ElasticNet(alpha=0.02, l1_ratio=0.5).fit(X, y)

fig, ax = plt.subplots(figsize=(9, 4))
idx = np.arange(d)
ax.bar(idx - 0.2, lasso.coef_, 0.4, label='Lasso')
ax.bar(idx + 0.2, enet.coef_, 0.4, label='Elastic Net')
ax.scatter(idx, w_true, c='k', marker='*', s=80, label='true', zorder=5)
ax.set_xlabel('feature j'); ax.set_ylabel('coefficient')
ax.set_title('Correlated features 0–4 — Lasso는 몇 개만, EN은 고르게 분산')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

**관찰**: Lasso는 5개 상관 feature 중 **1~2개만** 크게 뽑고 나머지를 0으로 두지만, Elastic Net은 **5개 모두에 coefficient 분산**.

### 실험 2 — Group Lasso 구현 (Proximal Gradient)

```python
def group_lasso_pg(X, y, groups, lam, eta=0.01, iters=1000):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        # smooth gradient step
        grad = -X.T @ (y - X @ w) / n
        w_half = w - eta * grad
        # block soft threshold
        for g in groups:
            norm_g = np.linalg.norm(w_half[g])
            thr = eta * lam * np.sqrt(len(g))
            if norm_g > thr:
                w[g] = (1 - thr / norm_g) * w_half[g]
            else:
                w[g] = 0
    return w

# Group 구조: 10개 feature를 5개 그룹(각 2개)
groups = [list(range(2*i, 2*i+2)) for i in range(5)]
w_true = np.zeros(10); w_true[0] = 1.2; w_true[1] = -0.8   # group 0만 active
w_true[4] = 0.5; w_true[5] = 0.3                             # group 2도 active
X = np.random.randn(100, 10) / np.sqrt(100)
y = X @ w_true + 0.1 * np.random.randn(100)

w_gl = group_lasso_pg(X, y, groups, lam=0.1)
print("Group Lasso coefficients:")
for g_idx, g in enumerate(groups):
    active = np.any(np.abs(w_gl[g]) > 1e-4)
    print(f"  group {g_idx} (idx {g}): {w_gl[g]} — {'ACTIVE' if active else 'inactive'}")
# → group 0, 2만 ACTIVE, 나머지는 전체가 0
```

### 실험 3 — Sparsity pattern 시각 비교

```python
from sklearn.linear_model import Lasso, MultiTaskLasso

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
# 왼쪽: Lasso path
lams = np.logspace(-3, 0, 30)
lasso_path = np.array([Lasso(alpha=lam, max_iter=10000).fit(X, y).coef_ for lam in lams])
enet_path = np.array([ElasticNet(alpha=lam, l1_ratio=0.5, max_iter=10000).fit(X, y).coef_ for lam in lams])
gl_path = np.array([group_lasso_pg(X, y, groups, lam) for lam in lams])

for ax, path, title in zip(axes, [lasso_path, enet_path, gl_path],
                           ['Lasso', 'Elastic Net', 'Group Lasso']):
    for j in range(10):
        ax.semilogx(lams, path[:, j])
    ax.axhline(0, c='k', lw=0.4)
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('coef')
    ax.set_title(title); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
# → Lasso는 coefficient가 하나씩 0에서 "튀어나옴",
#   EN은 상관 feature들이 동시에 활성,
#   Group Lasso는 2개씩 (그룹 단위로) 동시에 활성
```

---

## 🔗 실전 활용

### 선택 가이드

| 상황 | 권장 |
|------|------|
| 상관 feature가 거의 없다 | Lasso |
| 상관 그룹이 있지만 구조 모름 | Elastic Net (`l1_ratio ≈ 0.5`부터) |
| 자연스러운 그룹 구조 (one-hot, pathway) | Group Lasso |
| 그룹 내부도 sparse하게 | Sparse Group Lasso (Simon 2013): $\|w\|_1 + \sum \|w_g\|_2$ |
| Matrix-valued (low-rank) | Nuclear norm (여기의 행렬 확장) |

### Adam에서의 구현

딥러닝 실전에서 **weight decay = L2**는 Ridge의 특수 경우(하지만 Ch7-03에서 본 Adam 왜곡 주의). **Structured L2 / L1**은 PyTorch의 `weight_decay` 인자로는 불가 — custom regularizer 작성 필요:

```python
# ElasticNet-style regularizer
def elastic_loss(model, lam1, lam2):
    l1 = sum(p.abs().sum() for p in model.parameters())
    l2 = sum((p**2).sum() for p in model.parameters())
    return lam1 * l1 + lam2 * l2
```

### 현대 딥러닝과의 연결

- **Weight pruning**: Group Lasso with channel-wise groups → **structured sparsity** (He et al. 2017), 추론 가속.
- **Low-rank adapters**: LoRA의 $\Delta W = A B$ 구조는 nuclear norm regularization과 유사한 low-rank inductive bias.
- **Mixture of Experts**: expert별 activation이 group sparse.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 그룹 경계가 known | 실제로는 "그룹이 뭐인가"가 미지 → multi-task Lasso / GSGL |
| Convex penalty | Non-convex SCAD/MCP가 더 aggressive sparsity, bias 감소 |
| Feature 독립성 가정 | Graph Lasso처럼 feature 간 graph structure는 별도 기법 필요 |
| 모든 그룹 같은 가중치 | 실전에서는 그룹 크기·중요도 고려한 weighted Group Lasso |
| Single-task | Multi-task에서는 L2,1 norm으로 task 간 공유 feature 선택 |

---

## 📌 핵심 정리

$$\boxed{\text{Elastic Net} = \lambda_1\|w\|_1 + \lambda_2\|w\|^2 \quad \text{Group Lasso} = \lambda \sum_g \sqrt{p_g}\|w_g\|_2}$$

| 개념 | 의미 |
|------|------|
| **Elastic Net** | L1 sparsity + L2 grouping, Lasso의 상관-불안정 해결 |
| **Group Lasso** | 그룹 단위 on/off, feature 구조 정보 활용 |
| **Block soft threshold** | $\text{prox}_{\lambda\|\cdot\|_2}(z) = (1-\lambda/\|z\|_2)_+ z$ |
| **Proximal gradient** | smooth step + proximal step — 일반 convex-composite 최적화 |
| **Ch1 마무리** | Prior / Geometry / Spectrum / Structure 네 관점 통합 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\|z\|_2 = 1.2, \lambda = 0.5$의 block soft threshold 결과는? $\lambda = 1.5$면?

<details>
<summary>힌트 및 해설</summary>

- $\lambda = 0.5$: $(1 - 0.5/1.2)_+ \cdot z = 0.583 \cdot z$ — 같은 방향으로 58.3% 크기.
- $\lambda = 1.5$: $1 - 1.5/1.2 = -0.25 < 0$, clipped to 0. **전체 벡터가 0**.

핵심: "**그룹 전체가 같은 비율로 축소 또는 전체가 0**". 이것이 group-level on/off의 근본 메커니즘.

</details>

**문제 2** (심화): Elastic Net의 grouping effect를 정량화하라. 두 feature $x_1 = x_2$ (완전히 동일)일 때 EN의 해는 $\hat{w}_1 = \hat{w}_2$인 이유를 Lagrangian으로 보여라.

<details>
<summary>힌트 및 해설</summary>

$x_1 = x_2$이므로 목적함수는 $(w_1, w_2)$에 대해 $w_1 + w_2$ 결합과 $(w_1^2 + w_2^2)$에 의존. Fix $s = w_1 + w_2$. 그러면:

- Loss와 L1: $s$에만 의존 (왜냐하면 $\|y - x_1(w_1+w_2)\|^2$와 $|w_1| + |w_2| \geq |w_1+w_2|$는 $w_1, w_2$ 같은 부호일 때 같음).
- L2: $w_1^2 + w_2^2 \geq s^2/2$ (Cauchy-Schwarz; 등호 $w_1 = w_2 = s/2$에서).

L2가 **등호 조건**을 강제 → EN은 $w_1 = w_2$를 선택. 직관: **L2가 "공평하게 나눠라"를 강제**, L1은 sparsity만.

Lasso($\lambda_2 = 0$)에서는 이 constraint가 없어 $(s, 0)$이나 $(0, s)$도 같은 해 → degenerate, 선택은 numerical 불안정으로 결정.

</details>

**문제 3** (이론-실전): Group Lasso로 CNN의 **채널 pruning**을 수행하려면 어떤 group 구조를 써야 하는가? Channel pruning과 L2,1 norm의 관계를 설명하라.

<details>
<summary>힌트 및 해설</summary>

CNN의 conv layer weight $W \in \mathbb{R}^{c_\text{out} \times c_\text{in} \times k \times k}$. **Channel pruning**은 "출력 channel $i$ 전체를 제거" 할지 결정. 이를 위한 group 구조:

- 각 **출력 채널 전체**를 하나의 그룹: $g_i = \{W_{i,:,:,:}\}$, group size $p_i = c_\text{in} \cdot k^2$.
- 이 때 penalty $\lambda \sum_i \|W_{i,:,:,:}\|_2$는 **matrix L2,1 norm** (각 row의 L2 norm을 합).

Block soft thresholding이 각 출력 channel을 그룹으로 0 또는 유지 → 훈련 후 단순히 0이 된 출력 channel을 제거하면 **하드웨어 친화적 pruning** (structured sparsity).

비교: element-wise L1은 개별 weight를 0으로 만들지만 unstructured → 실제 추론 가속 어려움. Group Lasso로 structured sparsity를 얻어야 FLOPs 실감 절감.

관련: Lebedev-Lempitsky 2016, He et al. 2017 "Channel Pruning for Accelerating Very Deep Neural Networks", NISP 등.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Ridge의 SVD Shrinkage](./04-ridge-svd-shrinkage.md) | [📚 README로 돌아가기](../README.md) | [Chapter 2 → 01. Dropout = 앙상블 ▶](../ch2-dropout/01-dropout-ensemble.md) |

</div>
