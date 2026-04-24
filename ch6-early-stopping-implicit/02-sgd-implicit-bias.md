# 02. SGD의 Implicit Regularization

## 🎯 핵심 질문

- **Soudry et al. 2018**: separable logistic에서 GD가 **max-margin SVM 해**로 수렴하는 이유는?
- 수렴 rate $O(\log t / \sqrt{\log\log t})$의 의미는?
- SGD의 **SDE approximation** $d\theta = -\nabla L dt + \sqrt{2T} dB_t$가 어떻게 flat minimum을 선호하게 하는가?
- **Batch size effect**: large batch가 왜 sharp minimum으로, small batch가 왜 flat minimum으로 유도하는가?

---

## 🔍 왜 SGD의 implicit bias가 중요한가

Ch6-01은 early stopping이 L2 regularization의 implicit version임을 보였다. **SGD 자체**도 regularizer로 작용:

1. **Max-margin selection**: Separable classification에서 SGD가 명시적 regularization 없이도 **max-margin** 해를 고른다.
2. **Flat minimum preference**: Large batch ↔ sharp minimum, small batch ↔ flat minimum (Keskar 2017).
3. **SDE noise**: Mini-batch gradient의 noise가 Brownian motion 역할 → escape from sharp minima.

이 세 현상이 **modern NN의 generalization**을 설명하는 핵심 퍼즐 조각이다 (Generalization Theory Deep Dive Ch5).

이 문서는:
1. Soudry 2018의 max-margin 수렴.
2. SGD의 SDE 해석.
3. Flat vs sharp minimum 직관.

---

## 📐 수학적 선행 조건

- Logistic regression, binary classification
- SVM의 max-margin classifier
- SDE (Stochastic Differential Equation) 기초
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): SGD 수렴 분석

---

## 📖 직관적 이해

### Separable Classification with Logistic Loss

Binary data $\{(x_i, y_i)\}, y_i \in \{-1, +1\}$. **Linearly separable**: $\exists w^*$ s.t. $y_i (w^{*T} x_i) > 0 \ \forall i$.

Logistic loss: $L(w) = \sum_i \log(1 + e^{-y_i w^T x_i})$. $L \to 0$ when $y_i w^T x_i \to \infty$ → $\|w\| \to \infty$.

**Question**: $\|w\| \to \infty$이지만 **direction** $\hat{w} := w/\|w\|$은 어디로 수렴?

### Soudry 2018의 Answer

GD의 iterate $w_t$의 direction이 **max-margin SVM 해 $w_{\text{SVM}}$**로 수렴:

$$\frac{w_t}{\|w_t\|} \to \frac{w_{\text{SVM}}}{\|w_{\text{SVM}}\|}$$

$w_{\text{SVM}} = \arg\min_w \tfrac{1}{2}\|w\|^2$ s.t. $y_i w^T x_i \geq 1 \ \forall i$ (SVM primal).

**함의**: GD 명시적 regularization 없이도 **"가장 좋은 해"** (max margin) 선택. This is **implicit regularization**.

### Rate $O(\log t / \sqrt{\log\log t})$

$\|w_t/\|w_t\| - w_{\text{SVM}}/\|w_{\text{SVM}}\|\| = O(\log t / \sqrt{\log \log t})$.

**매우 느림**: 정밀 수렴을 위해 엄청난 $t$ 필요. 하지만 **모든 gradient step이 opposite direction으로 가지 않음** → 구조적 수렴.

### SDE Approximation of SGD

SGD update: $\theta_{t+1} = \theta_t - \eta \nabla L_B(\theta_t)$ where $L_B$ is mini-batch loss.

Continuous limit (small $\eta$, Li 2017):

$$d\theta = -\nabla L(\theta) dt + \sqrt{\eta \cdot \Sigma(\theta)} dB_t$$

$\Sigma(\theta) = \text{Cov}(\nabla L_B(\theta))$ is the mini-batch gradient covariance. $B_t$는 Brownian motion.

**$\sqrt{\eta}$ noise**: larger $\eta$ = more noise. Smaller batch = larger $\Sigma$ = more noise.

### Flat vs Sharp Minima (Keskar 2017)

Loss surface의 **curvature**:
- **Sharp minimum**: $\nabla^2 L$ eigenvalues large → small basin.
- **Flat minimum**: eigenvalues small → large basin.

**주장**: Flat minima가 더 좋은 generalization.

- Train set과 test set이 약간 다른 분포에서 sampled.
- Flat minimum의 $\theta$ 근방의 $\theta'$도 low loss → test에서 robust.
- Sharp minimum은 $\theta'$에서 high loss — test에서 fragile.

### Small Batch = Flat Minimum

SGD noise ($\sqrt{\eta \Sigma}$) 가 sharp minimum에서 "**escape**"시킴. Flat minimum에서는 noise가 해 근처에 머물게.

**Keskar 2017 실험**: Large batch (8192)가 sharp minimum에, small batch (128) 이 flat에 수렴.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Separable Dataset

$\{(x_i, y_i)\}$ separable $\iff \exists w^* : y_i w^{*T} x_i > 0 \ \forall i$.

### 정의 2.2 — Max-Margin SVM

$$w_{\text{SVM}} = \arg\min_w \tfrac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i w^T x_i \geq 1 \ \forall i$$

Dual form: $w_{\text{SVM}} = \sum_i \alpha_i y_i x_i$, $\alpha_i$ support vectors에만 non-zero.

### 정리 2.3 — Soudry 2018 (주 정리)

Linearly separable logistic regression에서, GD with constant $\eta$:

$$\lim_{t \to \infty} \frac{w_t}{\|w_t\|} = \frac{w_{\text{SVM}}}{\|w_{\text{SVM}}\|}$$

수렴 rate:

$$\left\|\frac{w_t}{\|w_t\|} - \frac{w_{\text{SVM}}}{\|w_{\text{SVM}}\|}\right\| = O\left(\frac{\log \log t}{\log t}\right)$$

(정확히 $O(\log \log t / \log t)$ — 실전에 매우 느림.)

### 정리 2.4 — SDE Approximation (Li 2017)

Small learning rate $\eta \to 0$에서 SGD:

$$\theta_{t+1} = \theta_t - \eta \nabla L_B(\theta_t)$$

는 다음 SDE에 수렴:

$$d\theta = -\nabla L(\theta) dt + \sqrt{\eta \cdot \Sigma(\theta)} dB_t$$

여기서 $\Sigma = \text{Cov}_{B}[\nabla L_B]$ — batch gradient의 covariance.

### 정리 2.5 — Jastrzebski 2017: Effective Temperature

Flat minimum 주변의 stationary distribution of SGD ≈ Gibbs distribution at temperature:

$$T_{\text{eff}} = \frac{\eta}{B}$$

$B$: batch size. **Large batch = low temperature = sharp minimum**.

**Generalization gap**:

$$\text{gap} \propto \text{curvature of minimum} \propto 1/T_{\text{eff}} \propto B/\eta$$

즉 large batch / small learning rate가 **worse generalization**.

### 정리 2.6 — Keskar 2017 (empirical observation)

Batch size 256 vs 8192:
- Same architecture, same training data, same steps.
- **Large batch** (8192): **0.5-1% worse test accuracy**, sharper minimum.
- **Small batch** (256): better accuracy, flatter minimum.

---

## 🔬 수학적 유도

### Soudry 2018 증명 Idea

GD update: $w_{t+1} = w_t - \eta \nabla L(w_t)$.

Logistic gradient: $\nabla L(w) = -\sum_i y_i x_i / (1 + e^{y_i w^T x_i})$.

**핵심 관찰**: 시간이 지나면 **support vectors**에 $\nabla$이 집중. Support vectors $i$: $y_i w^T x_i$ 가장 작은 samples.

**Asymptotic behavior**: $w_t \approx \log t \cdot w_{\text{SVM}} + \text{small correction}$.

Therefore $w_t / \|w_t\| \to w_{\text{SVM}} / \|w_{\text{SVM}}\|$.

Full proof (Soudry 2018)는 복잡한 경계 분석 필요 — support set identification + correction term bounds.

### SDE 유도 (간략)

SGD update: $\theta_{t+1} - \theta_t = -\eta \nabla L_B$.

Decompose: $\nabla L_B = \nabla L + \xi$, $\xi$ = mini-batch noise ($\mathbb{E}[\xi] = 0$, $\text{Cov}(\xi) = \Sigma(\theta)/B$).

$\theta_{t+1} - \theta_t = -\eta \nabla L - \eta \xi$

Continuous limit ($\eta \to 0$):
- Drift: $-\nabla L dt$.
- Diffusion: $\eta \xi$에서 $\sqrt{\eta \Sigma/B} dB_t$ ($dB_t \sim \mathcal{N}(0, dt)$).

### Flat minimum의 Gibbs distribution

Stationary distribution of SDE with drift $-\nabla L$ and diffusion $\sqrt{2T_{\text{eff}}}$:

$p^*(\theta) \propto e^{-L(\theta)/T_{\text{eff}}}$

$L(\theta) \approx L_{\min} + \tfrac{1}{2}(\theta - \theta^*)^T H (\theta - \theta^*)$ (quadratic 근사).

Stationary distribution Gaussian: $\mathcal{N}(\theta^*, T_{\text{eff}} H^{-1})$.

**Flat minimum** (small $\|H\|$) → wider stationary distribution → robustness to perturbation.

---

## 💻 실험으로 효과 검증

### 실험 1 — Linearly separable data에서 GD → max margin

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D separable data
np.random.seed(0)
n = 20
X = np.vstack([np.random.randn(n, 2) + [3, 0], np.random.randn(n, 2) + [-3, 0]])
y = np.hstack([np.ones(n), -np.ones(n)])

# GD on logistic
def gd_logistic(X, y, lr=0.01, steps=10000):
    n, d = X.shape
    w = np.zeros(d)
    trajectory = [w.copy()]
    for _ in range(steps):
        yp = X @ w
        grad = -X.T @ (y / (1 + np.exp(y * yp))) / n
        w -= lr * grad
        trajectory.append(w.copy())
    return np.array(trajectory)

traj = gd_logistic(X, y)

# Compute max-margin SVM 해 (scikit-learn)
from sklearn.svm import LinearSVC
svm = LinearSVC(C=1e6, dual=True, max_iter=100000)
svm.fit(X, y)
w_svm = svm.coef_.ravel()

# 방향 수렴 측정
norms = np.linalg.norm(traj, axis=1)
directions = traj / norms[:, None]
svm_dir = w_svm / np.linalg.norm(w_svm)
angle_errors = [np.arccos(np.clip(d @ svm_dir, -1, 1)) for d in directions[1:]]

plt.loglog(angle_errors)
plt.xlabel('GD step'); plt.ylabel(r'angle error $\|w_t/\|w_t\| - w_{SVM}/\|w_{SVM}\|\|$')
plt.title('Soudry 2018: GD direction → SVM direction')
plt.grid(alpha=0.3); plt.show()
# → angle error 천천히 감소 (log rate)
```

### 실험 2 — Small vs large batch의 minimum flatness

```python
import torch
import torch.nn as nn

def train_with_batch(batch_size, lr=0.01, steps=5000):
    # Simple 2-layer NN on synthetic task
    net = nn.Sequential(nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 10))
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    
    # Train
    X = torch.randn(5000, 50)
    y = torch.randint(0, 10, (5000,))
    for t in range(steps):
        idx = torch.randint(0, 5000, (batch_size,))
        loss = nn.CrossEntropyLoss()(net(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net

# Train with different batch sizes
net_small = train_with_batch(batch_size=32)
net_large = train_with_batch(batch_size=1024)

# Estimate minimum flatness via Hessian trace approximation
def hessian_trace(model, loss_fn, data, num_samples=20):
    """Approximation via Hutchinson trick."""
    # Simplified — in practice use torch.func or pyhessian
    traces = []
    for _ in range(num_samples):
        v = [torch.randn_like(p) for p in model.parameters()]
        loss = loss_fn(model(data[0]), data[1])
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        gv = sum((g * vv).sum() for g, vv in zip(grad, v))
        hv = torch.autograd.grad(gv, model.parameters())
        traces.append(sum((h * vv).sum() for h, vv in zip(hv, v)).item())
    return np.mean(traces)

# Flatness proxy: hessian trace (작을수록 flat)
# Keskar 2017의 finding: small batch가 더 flat
```

### 실험 3 — SGD noise의 "temperature" 시각화

```python
# Loss landscape의 2D slice에서 SGD trajectory
# Small batch: 더 많은 exploration, 점차 flat region 찾음
# Large batch: GD처럼 straight line, sharp region에 빠질 가능성

# (복잡하지만 Li 2018, Xing 2018 등 참고)
```

### 실험 4 — Batch size와 generalization gap

```python
batch_sizes = [32, 128, 512, 2048]
results = {}
for B in batch_sizes:
    net = train_with_batch(batch_size=B, lr=0.01 * (B/128)**0.5)  # lr scale
    train_acc = evaluate(net, X_train, y_train)
    test_acc = evaluate(net, X_test, y_test)
    results[B] = (train_acc, test_acc, train_acc - test_acc)

print("Batch | Train | Test | Gap")
for B, (tr, te, g) in results.items():
    print(f"{B:5d} | {tr:.3f} | {te:.3f} | {g:.4f}")
# → 큰 batch에서 generalization gap 큼 (Keskar 2017)
```

---

## 🔗 실전 활용

### Learning Rate와 Batch Size의 관계

**Linear scaling rule** (Goyal 2017): $\text{new lr} = \text{old lr} \times (\text{new batch} / \text{old batch})$.

Effective temperature $T_{\text{eff}} = \eta/B$을 유지하기 위한 방법. Large batch에서도 generalization 유지하려면 lr 증가.

**한계**: 매우 큰 batch ($\geq 64K$)에서는 linear scaling이 break down — LARS, LAMB 같은 adaptive scaling 필요.

### Mini-batch Noise Engineering

- **Small batch (< 128)**: 강한 noise, flat minimum 선호, generalization OK.
- **Medium batch (256 - 1024)**: Standard 선택.
- **Large batch (> 4096)**: Distributed training 유리, 하지만 generalization 저하 risk — warmup + LR scaling 필수.

### Gradient Noise Injection

"SGD noise = regularization"이라면, 명시적으로 Gaussian noise 추가는?

- **Stochastic Weight Averaging** (SWA, Ch7-01): 여러 SGD iterate 평균 → flat minimum.
- **Gaussian noise on gradients**: 경험적으로 hurt보다 help 적음 — 적절한 $\eta, B$ tuning이 더 중요.

### 현대 LLM Recipe

- **AdamW** (Ch7-03): SGD 변형, implicit bias 다름.
- **Cosine schedule**: End of training에서 lr $\to 0$ → deterministic convergence.
- **Warmup**: 처음에 lr $\to \eta_{\max}$ 증가 → large batch compensation.

이 모든 선택이 "**implicit bias를 원하는 방향으로 유도**" 시도.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear separable data | Non-separable에서 max-margin 정의 모호 |
| Logistic loss | Hinge loss, cross-entropy도 유사 결과 (다른 rate) |
| Constant $\eta$ | Scheduled lr에서 수렴 분석 수정 필요 |
| SDE approximation | $\eta \to 0$ 극한, 실전 $\eta$에서는 discrete effect |
| Flat minimum ↔ generalization | Dinh 2017: re-parameterization으로 반례 가능 |
| Single task | Multi-task에서는 implicit bias 복합적 |

**주의**: "Flat minimum → generalization"은 **heuristic**. Dinh 2017은 같은 function을 sharp minimum에 매개변수화 가능함을 보여 이 주장을 비판.

---

## 📌 핵심 정리

$$\boxed{\text{SGD = GD + noise} \to \text{max-margin solution + flat minimum preference}}$$

| 개념 | 의미 |
|------|------|
| **Soudry 2018** | Separable logistic GD → max-margin SVM 해 |
| **Rate** | $O(\log t / \sqrt{\log \log t})$ — 매우 느린 convergence |
| **SDE approximation** | SGD $\to d\theta = -\nabla L dt + \sqrt{\eta\Sigma}dB$ |
| **$T_{\text{eff}} = \eta/B$** | Effective temperature, flat minimum 선호 |
| **Batch size 효과** | Small batch → flat, large batch → sharp |

---

## 🤔 생각해볼 문제

**문제 1** (기초): SDE의 $T_{\text{eff}} = \eta/B$ 공식에서, 같은 $T_{\text{eff}}$ 유지하려면 batch 2배 증가 시 lr은 어떻게?

<details>
<summary>힌트 및 해설</summary>

$T_{\text{eff}} = \eta/B$ constant → $\eta$도 2배 증가. 이것이 **linear scaling rule**.

예: lr=0.1, batch=128 → lr=0.2, batch=256.

**실전 tip**: Linear scaling은 batch size ~1000까지 유효. 그 이상은 **square root scaling** ($\eta \propto \sqrt{B}$) 또는 LARS/LAMB.

</details>

**문제 2** (심화): Soudry 2018의 $O(\log t / \sqrt{\log\log t})$ rate가 매우 느린 이유를 직관적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

$\|w_t\| \to \infty$ 수렴의 **two-stage** 성격:

1. **Magnitude growth**: $\|w_t\| \sim \log t$ (logistic gradient의 sum).
2. **Direction convergence**: Support vectors 중 non-support error가 점차 감소.

Direction error는 non-support sample의 "**margin deficit**"에 의해 결정:
- "Almost-support" vector의 margin이 1에 가까워지는 rate는 slow (loss가 $1/(1 + e^x)$ shape).
- 이 slow rate가 overall $\log t / \sqrt{\log\log t}$로 나타남.

**실전**: 수렴 rate는 매우 느리지만 **initial iteration에서 이미 좋은 direction**. Early stopping이 finetune 할 필요 없이 좋은 generalization.

**이론 vs 실전**: 이 결과는 "SGD의 infinite training 행동"을 설명. Finite training에서는 approx max-margin 방향으로 가지만 exact 아님.

</details>

**문제 3** (이론-실전): "Flat minimum → generalization" 주장의 Dinh 2017 반례를 설명하라. 이 반례가 flat minimum 개념을 완전히 반박하는가?

<details>
<summary>힌트 및 해설</summary>

**Dinh 2017의 반례**: Same function $f$ 를 두 가지 다른 parameterization으로 표현 가능.

$f(x) = w_1 w_2 x$ (2-dim parameter).
- $(w_1, w_2) = (1, 1)$: flat minimum in 2D parameter space.
- $(w_1, w_2) = (c, 1/c)$ for large $c$: sharp minimum (one direction very sensitive).

**두 parameterization이 같은 function을 주지만** flatness는 drastically 다름. 따라서 "**flatness는 parameterization-dependent**" — generalization은 parameterization-invariant.

**함의**:
- "Flat minimum → generalization"이 boolean 명제로는 틀림.
- 하지만 specific architecture / parameterization에서 경험적으로 correlate.

**후속 연구**:
- **Kaddour 2022**: Flat minima의 **specific measure** (sharpness definition)가 중요.
- **Adaptive SAM** 등: scale-invariant sharpness measure 제안.
- **Normalization** (BN, LN): parameterization normalization → flatness 개념 stable.

**실전 결론**: Flat minimum은 **useful heuristic** — SWA, SAM 같은 기법이 flat 추구해서 유용. 하지만 formal proof는 architecture-specific. 

"SGD가 flat minimum 찾는다"는 주장은 **specific contexts에서만** 정량적으로 유효. 일반 이론은 아직 열린 문제.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Early Stopping = L2](./01-early-stopping-as-l2.md) | [📚 README로 돌아가기](../README.md) | [03. Ridgeless Regression ▶](./03-ridgeless-regression.md) |

</div>
