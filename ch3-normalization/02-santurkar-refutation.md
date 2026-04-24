# 02. Santurkar 2018의 BN 신화 반박

## 🎯 핵심 질문

- Santurkar 2018 "How Does Batch Normalization Help Optimization?"의 **실험적 반박**은 정확히 무엇이었는가?
- ICS가 BN의 정당화가 **아니라면** 실제 메커니즘은 무엇인가?
- **Loss/gradient의 Lipschitz smoothness**를 어떻게 증명하는가?
- 이 결과는 다른 normalization 기법 선택에 어떤 함의를 주는가?

---

## 🔍 왜 이 문서가 중요한가

Ch3-01은 Ioffe & Szegedy의 **원래 주장(ICS)** 을 그대로 소개했다. 이 문서는 그 주장의 **수정**이다. 이 패턴 — "신화 vs 실제" — 는 Dropout의 세 해석(Ch2)에서도 반복되고, BN 이후의 모든 normalization 설계에 영향을 미친다.

Santurkar의 **3가지 기여**:

1. **실험적 반박**: BN 후 activation에 인위 noise 주입 → ICS 다시 유발 → 그래도 BN이 여전히 훈련 가속. 즉 ICS 완화는 **충분조건도 필요조건도 아님**.
2. **실제 메커니즘 제안**: BN이 loss landscape의 **Lipschitz 상수와 gradient Lipschitz 상수를 감소** → 더 smooth → 큰 lr 허용.
3. **정량적 증명**: Convex quadratic에 대해 BN의 smoothness 개선을 rigorous bound로 제시.

**함의**:
- BN의 효과는 "분포 안정"이 아니라 "**optimization landscape 기하**".
- 이는 **왜 LayerNorm, GroupNorm, WeightNorm도 효과적**인지 통합 설명 — 이들도 각자 다른 방식으로 landscape을 smoothing.
- **SAM (Ch7-02)이 명시적으로 sharpness를 최소화**하는 것과 연결 — BN은 암묵적, SAM은 명시적.

---

## 📐 수학적 선행 조건

- Ch3-01: BN의 수식, forward/backward
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Lipschitz 연속성, smoothness, condition number
- 해석학: Lipschitz 상수 $L$-smoothness $\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$
- 선형대수: Hessian의 spectral norm

---

## 📖 직관적 이해

### ICS 정의와 측정

Ioffe의 non-formal definition: "Layer $\ell$의 input 분포가 훈련 중 변화하는 것".

**측정**: 훈련 중 각 step에서 layer $\ell$의 input의 mean/variance 변화량:

$$\Delta \mu_\ell^{(t)} = \|\mu_\ell^{(t)} - \mu_\ell^{(t-1)}\|, \quad \Delta \sigma_\ell^{(t)} = \|\sigma_\ell^{(t)} - \sigma_\ell^{(t-1)}\|$$

BN이 이를 감소시킨다는 것이 원 주장.

### Santurkar의 실험 — 반박

"BN 뒤에 artificial noise를 주입하면" 측정된 ICS가 **다시 증가**. 구체적으로:

$$y = \text{BN}(x) + \mathcal{N}(0, \Sigma_{\text{large}}) + \text{time-varying shift}$$

이 수정된 네트워크에서:
- ICS 측정치: 매우 큼 (noise 때문).
- **훈련 속도**: 여전히 표준 BN과 거의 동일.

**결론**: ICS가 완화되지 않아도 BN은 여전히 효과적 → ICS는 **BN의 효과를 설명하지 못한다**.

### 진짜 메커니즘 — Lipschitzness

Santurkar는 loss $L$와 gradient $\nabla L$의 Lipschitz 상수를 측정:

- With BN: $L$이 smoothly 감소, $\nabla L$도 매끄럽게 변화.
- Without BN: $L$이 **step-wise jump**, $\nabla L$이 요동.

**즉 BN은 "loss surface의 sharpness"를 줄인다** → 큰 lr이 안전, convergence 가속.

### Sharpness와 Flat Minima

"Flat minimum → generalization"(Keskar 2017, Ch6-02)와 직결. BN이 flat region을 만들어 SGD가 그 안에 머물기 쉽게 한다. 이는 Ch7-02 **SAM**의 명시적 sharpness penalty와 같은 목적을 달성.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — $L$-Lipschitz Function

함수 $f: \mathbb{R}^d \to \mathbb{R}$이 **$L$-Lipschitz**:

$$|f(x) - f(y)| \leq L \|x - y\|, \quad \forall x, y$$

### 정의 2.2 — $\beta$-Smoothness (Gradient Lipschitz)

$f$가 **$\beta$-smooth**:

$$\|\nabla f(x) - \nabla f(y)\| \leq \beta \|x - y\|, \quad \forall x, y$$

동치: Hessian이 bounded spectral norm, $\|\nabla^2 f(x)\|_{\text{op}} \leq \beta$.

### 정리 2.3 — Santurkar 2018의 주 정리 (Informal)

Convex quadratic loss $L(y) = \tfrac{1}{2} y^T A y$ ($A \succ 0$)와 BN layer에서, 다음이 성립:

$$\|\nabla L \circ \text{BN}\|_{\text{Lip}} \leq \frac{1}{\sigma_B} \|\nabla L\|_{\text{Lip}}$$

즉 BN의 gradient Lipschitz 상수는 pre-BN의 $1/\sigma_B$로 축소. $\sigma_B > 1$ ($\sigma_B^2$ 크다)일수록 개선 크다.

### 정리 2.4 — BN의 Loss Lipschitzness

마찬가지로 loss 자체의 Lipschitz 상수도:

$$\|L \circ \text{BN}\|_{\text{Lip}} \leq \frac{\text{const}}{\sigma_B} \|L\|_{\text{Lip}}$$

두 정리 합쳐 BN이 **condition number**를 개선 → **큰 learning rate 안전성** (Nesterov-smooth optimization의 convergence rate $O(\beta / \mu)$에서 $\beta$ 감소).

### 정리 2.5 — BN Rescaling Property

BN layer $y = \text{BN}(x)$에 대해 weight $W$를 $c W$로 스케일해도 $y$ 불변:

$$\text{BN}(c W x) = \text{BN}(W x)$$

**Gradient의 scale invariance**:

$$\nabla_W L(c W) = \frac{1}{c} \nabla_W L(W)$$

즉 lr의 효과가 $1/c$로 자동 보정 → large effective lr.

---

## 🔬 수학적 유도

### 정리 2.3 증명 스케치 (1D case)

$x \in \mathbb{R}^m$ (batch of $m$), BN 없이 $y = x$, BN 있이 $\hat{y} = (x - \mu)/\sigma$ ($\mu = \bar{x}, \sigma = \text{std}(x)$).

$\nabla_x \hat{y}$ 계산: $\hat{y}_i = (x_i - \mu)/\sigma$. 체인 룰:

$$\frac{\partial \hat{y}_i}{\partial x_j} = \frac{1}{\sigma} \left(\delta_{ij} - \frac{1}{m} - \frac{(x_i - \mu)(x_j - \mu)}{m \sigma^2}\right)$$

**Jacobian** $J = \partial \hat{y}/\partial x$는 symmetric, $\|J\|_{\text{op}} = 1/\sigma$ (singular value decomposition, 1 외의 eigenvalue 모두 $\leq 1/\sigma$).

Loss $L(\hat{y})$의 gradient w.r.t. $x$: $\nabla_x L = J^T \nabla_{\hat{y}} L$. Lipschitzness:

$$\|\nabla_x L\| \leq \|J\|_{\text{op}} \|\nabla_{\hat{y}} L\| = \frac{1}{\sigma} \|\nabla_{\hat{y}} L\|$$

따라서 BN 뒤 loss의 gradient Lipschitz는 $1/\sigma$로 scaled. $\square$

### Gradient Lipschitz vs Loss Lipschitz

- Loss Lipschitz: $|L(x) - L(y)| \leq L_0 \|x - y\|$ — $L$이 천천히 변화.
- Gradient Lipschitz: $\|\nabla L(x) - \nabla L(y)\| \leq L_1 \|x - y\|$ — gradient 자체가 천천히 변화 (= loss의 Hessian bounded).

Santurkar는 **both**가 BN으로 감소함을 실험으로 보인다. 이론은 convex quadratic에서만 rigorous.

### 실제 NN에서의 경험적 검증

Santurkar의 Fig 4 재현: ResNet / VGG 훈련 중 각 step에서:

$$L_{\text{smooth}}(\theta, \eta) = \max_{\eta' \in [0, \eta]} \left|L(\theta - \eta' \nabla L(\theta)) - L(\theta)\right|$$

"$\eta$로 step 한 후 loss가 실제 경험하는 변화의 최대값". BN 있는 경우 이 값이 **훨씬 smooth** (적은 jitter).

---

## 💻 실험으로 효과 검증

### 실험 1 — ICS "완화" 실패 실험 재현

```python
import torch
import torch.nn as nn

class BN_plus_noise(nn.Module):
    """BN 후 time-varying noise 추가 — ICS 재유발."""
    def __init__(self, channels, noise_std=1.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        self.noise_std = noise_std
        self.step = 0
    def forward(self, x):
        y = self.bn(x)
        if self.training:
            # time-varying shift and noise
            shift = self.noise_std * torch.sin(torch.tensor(self.step * 0.1))
            noise = self.noise_std * torch.randn_like(y)
            y = y + shift + noise
            self.step += 1
        return y

# 동일 아키텍처에서 BN vs BN+noise 훈련
# 측정: layer-input 분포 변화 (ICS) vs 훈련 loss 수렴
# → BN+noise는 ICS "나쁨"에도 훈련 수렴은 유사 (Santurkar 2018)
```

### 실험 2 — Loss landscape smoothness 측정

```python
import numpy as np

def measure_landscape_lipschitz(model, data_loader, loss_fn, n_samples=50):
    """각 step에서 gradient Lipschitz 상수 경험적 추정."""
    for x, y in data_loader:
        model.zero_grad()
        loss = loss_fn(model(x), y); loss.backward()
        grad_orig = [p.grad.clone() for p in model.parameters()]

        # 파라미터를 δ만큼 perturb 여러 번
        lip = []
        for _ in range(n_samples):
            delta_norm = 1e-3
            deltas = [torch.randn_like(p) * delta_norm for p in model.parameters()]
            with torch.no_grad():
                for p, d in zip(model.parameters(), deltas): p.add_(d)
            model.zero_grad()
            loss2 = loss_fn(model(x), y); loss2.backward()
            # gradient norm change
            diff = sum((p.grad - g0).norm()**2 for p, g0 in zip(model.parameters(), grad_orig))**0.5
            lip.append(diff.item() / delta_norm)
            with torch.no_grad():
                for p, d in zip(model.parameters(), deltas): p.sub_(d)  # restore
        return np.mean(lip)
```

### 실험 3 — BN 있는/없는 loss surface 시각화

```python
import matplotlib.pyplot as plt

def loss_along_direction(model, loss_fn, batch, d1, d2, R=1.0, N=30):
    """파라미터 공간의 2D 평면에서 loss surface."""
    params = [p.data.clone() for p in model.parameters()]
    alpha = np.linspace(-R, R, N); beta = np.linspace(-R, R, N)
    Z = np.zeros((N, N))
    x, y = batch
    for i, a in enumerate(alpha):
        for j, b in enumerate(beta):
            with torch.no_grad():
                for p, p0, e1, e2 in zip(model.parameters(), params, d1, d2):
                    p.copy_(p0 + a*e1 + b*e2)
            Z[i, j] = loss_fn(model(x), y).item()
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), params): p.copy_(p0)
    return alpha, beta, Z

# BN 있는 network와 없는 network의 같은 지점에서 loss surface plot
# → BN 있는 쪽이 훨씬 flat, 적은 ridge
```

### 실험 4 — 큰 lr의 robustness 증명

```python
class MLP(nn.Module):
    def __init__(self, use_bn):
        super().__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(20, 64); self.bn1 = nn.BatchNorm1d(64) if use_bn else nn.Identity()
        self.fc2 = nn.Linear(64, 64); self.bn2 = nn.BatchNorm1d(64) if use_bn else nn.Identity()
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# CIFAR-10 아래에서 다양한 lr로 훈련 수렴 여부 test
# → BN 없으면 lr > 0.1에서 발산, BN 있으면 lr = 1.0도 수렴 (Santurkar의 관찰)
```

---

## 🔗 실전 활용

### BN 선택의 재해석

BN의 현대적 해석: "landscape smoothing을 주는 함수 재매개변수화".

이 관점에서:
- BN이 안 되면 (small batch) → **다른 smoothing 기법** 찾기 (GroupNorm, LayerNorm).
- BN이 너무 "독재"적이면 → **WeightNorm** (Ch3-04): weight space에서 같은 효과를 다른 방식으로.
- Landscape 직접 modeling → **SAM** (Ch7-02).

### LR scheduling과의 상호작용

BN의 gradient scale invariance (정리 2.5) 덕분에:
- Cosine annealing / warmup이 덜 critical — BN이 자동으로 scale matching.
- 그러나 **layer-wise learning rate** (LARS, Layer-wise Adaptive Rate Scaling)는 BN 있을 때 덜 필요.

### 매우 깊은 네트워크

ResNet-1000 같은 극단적 깊이에서 BN이 없으면 gradient explode/vanish.
- **Fixup** (Ch3-05): BN 없이 초기화로 landscape 조정.
- **Layer Scale** (CaiT, 2021): residual branch의 $\alpha_l$로 flow 조절.

이 모두 Santurkar의 해석("BN = landscape tool")을 지지하는 증거.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Convex quadratic loss | 실제 NN에서는 non-convex, 증명은 경험적 확장 |
| Lipschitz 상수가 유한 | 무한 Lipschitz (e.g. sharp loss)에서는 적용 불가 |
| Batch size 큼 | Small batch에서는 BN 자체가 불안정 |
| 실험적 ICS 측정치 | 다른 ICS 정의 사용하면 결론 달라질 가능성 |
| 단일 실험 재현 | 이후 연구 중 BN의 **또 다른** 효과 발견 가능성 열림 |

**주의**: Santurkar의 결론은 "**ICS는 정당화가 아니다**"이지 "**ICS가 존재하지 않는다**"가 아님. ICS가 일부 영향이 있을 수도 있지만 **효과의 주 원인**은 아니다.

---

## 📌 핵심 정리

$$\boxed{\text{BN 효과} = \text{Loss/Gradient Lipschitz 상수 감소} \neq \text{ICS 완화}}$$

| 개념 | 의미 |
|------|------|
| **원래 주장 (Ioffe)** | BN이 ICS를 완화 |
| **Santurkar 반박** | ICS가 "나쁘더라도" BN 효과 여전 |
| **실제 메커니즘** | Loss landscape의 smoothness 개선 |
| **수학적 근거** | 정리 2.3 — gradient Lipschitz $\leq 1/\sigma_B$ |
| **함의** | Large lr, flat minima, 다른 normalization의 존재 이유 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): ICS 측정치 $\Delta\mu_\ell^{(t)}$가 훈련 중 어떻게 변하는지를 실험으로 확인하는 방법을 설명하라.

<details>
<summary>힌트 및 해설</summary>

Hook을 사용해 각 layer input의 통계량을 매 step마다 수집:

```python
input_stats = {l: [] for l in target_layers}
def hook(module, input, output):
    input_stats[module].append((input[0].mean().item(), input[0].std().item()))
for layer in target_layers:
    layer.register_forward_hook(hook)

# 훈련 루프 실행
# 이후 연속된 step 간 mean/std의 차이를 플롯
for layer, stats in input_stats.items():
    means = np.array([s[0] for s in stats])
    plt.plot(np.abs(np.diff(means)))
```

BN 있는 네트워크와 없는 네트워크를 동일하게 측정. Santurkar의 발견은 "심지어 noise injected BN이 ICS가 큰데도 훈련이 잘 됨"이 확인됨.

</details>

**문제 2** (심화): 정리 2.3에서 $1/\sigma_B$의 scaling이 왜 condition number를 개선하는가? Convergence rate에 미치는 영향을 서술하라.

<details>
<summary>힌트 및 해설</summary>

$\beta$-smooth, $\mu$-strongly convex loss의 GD는 $O(\log(1/\epsilon) \cdot \beta/\mu)$ iterations로 수렴.

BN이 $\beta$를 $1/\sigma_B$배로 감소 → **condition number $\beta/\mu$ 감소** → iteration 수 감소. 

예를 들어 $\sigma_B = 10$이면 (대표 NN activation variance) iteration 수가 **10배 감소**.

단 $\mu$ (strong convexity)는 BN에 의해 증가 안 함 (보통 감소도 안 함). 따라서 $\beta/\mu$의 ratio가 순수 개선.

이것이 "**BN이 3배 빠른 훈련**" 같은 경험적 관찰의 이론적 설명. Large-batch training (LARS, LAMB)도 비슷하게 adaptive scaling으로 effective condition number 조정.

</details>

**문제 3** (이론-실전): Santurkar의 결과는 "**LayerNorm도 같은 효과**"를 주장하지만, LayerNorm은 어떤 축에서 정규화하는가? BN과 LN의 landscape smoothing 효과는 정성적으로 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

- **BN**: batch 축 정규화 — 같은 feature의 batch 내 통계량.
- **LN**: feature 축 정규화 — 같은 sample의 feature 내 통계량.

**공통**: 둘 다 **$1/\sigma$의 gradient rescaling** 가능. 정리 2.3과 유사한 Lipschitz 감소.

**차이**:
- BN: batch 차원에 의존 → small batch에서 noisy $\sigma_B$.
- LN: batch 독립 → RNN, batch size 1, sequence model에서 안정.

**다른 smoothing quality**:
- BN: feature별 scaling (각 feature가 같은 range).
- LN: sample별 scaling (각 sample이 같은 magnitude).

CNN에서는 BN이 더 적절 (feature가 semantic, channel별 의미 있음). Transformer에서는 LN이 적절 (token간 semantic, feature간 scale matching 더 중요).

두 기법은 **같은 "landscape tool" family**지만 **다른 axis에 smoothness를 부여**. Santurkar의 framework가 두 기법을 통합 설명한다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Batch Normalization](./01-batch-norm.md) | [📚 README로 돌아가기](../README.md) | [03. Layer Normalization ▶](./03-layer-norm.md) |

</div>
