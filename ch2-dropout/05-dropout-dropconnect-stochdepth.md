# 05. Dropout vs DropConnect vs Stochastic Depth

## 🎯 핵심 질문

- **DropConnect** (Wan 2013)는 Dropout과 무엇이 같고 무엇이 다른가?
- **Stochastic Depth** (Huang 2016)는 ResNet block 전체를 skip — 이것이 왜 효과적인가?
- 세 기법의 **앙상블 크기**와 **effective regularization strength**는 각각 얼마인가?
- Linear model에서 "activation drop = weight row drop" 등가성은 무엇인가?

---

## 🔍 왜 이 확장이 필요한가

Dropout이 "**activation**을 drop"이라면, 자연스러운 변형은:

- Weight 자체를 drop? → **DropConnect**
- Layer 전체를 건너뛰기? → **Stochastic Depth**

이 변형들은 단순히 "Dropout의 친척"이 아니라, **regularization의 granularity (입도)**가 다른 기법들이다:

| 기법 | Drop 단위 |
|------|----------|
| Dropout | activation (뉴런 출력) |
| Spatial Dropout | activation channel |
| DropConnect | weight (connection) |
| Stochastic Depth | entire residual block |
| LayerDrop (Fan 2020) | entire layer in Transformer |

각 단위는 다른 inductive bias를 준다. 이 문서는 세 가지를 **앙상블 관점**에서 비교하고 실전 선택 가이드를 제공한다.

---

## 📐 수학적 선행 조건

- Ch2-01: Dropout의 앙상블 해석
- Ch2-04: Dropout의 변종
- ResNet 구조 (Neural Network Theory Deep Dive)
- Bernoulli 기댓값·분산 산술

---

## 📖 직관적 이해

### DropConnect

$y = W \cdot \text{act}(h)$가 아닌 $y = \tilde{W} \cdot \text{act}(h)$, $\tilde{W}_{ij} = m_{ij} W_{ij}$, $m_{ij} \sim \text{Bern}(1-p)$.

Weight 각 entry를 **독립**으로 drop. $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$이면 $2^{d_{\text{out}} \cdot d_{\text{in}}}$개의 서브-weight-configuration.

**Dropout과의 비교**:
- Dropout은 activation 한 칼럼(혹은 행)을 전체 drop — $W$ 기준으로는 **column 전체**가 기여 안 함.
- DropConnect는 weight 단위 → 더 fine-grained, 앙상블 크기 훨씬 큼.

### Stochastic Depth

ResNet의 residual block $x \to x + F(x)$에서 block $F$를 확률 $p_\ell$로 **완전히 drop**:

$$h_{\ell+1} = h_\ell + b_\ell \cdot F_\ell(h_\ell), \quad b_\ell \sim \text{Bern}(1-p_\ell)$$

$b_\ell = 0$이면 $h_{\ell+1} = h_\ell$ — identity만 통과. Effective depth가 매 batch마다 랜덤하게 줄어듦.

**Huang 2016의 주장**:
- 훈련: 평균 effective depth $\sum_\ell (1-p_\ell) L < L$ → 빠른 훈련, vanishing gradient 완화.
- Test: 모든 block 활성 → 전체 capacity 사용.
- **앙상블**: 각 mask가 다른 "depth의 ResNet". 자연스러운 ensemble of networks of varying depth.

### 앙상블 크기 비교 ($N$ 뉴런 layer, $W = d \cdot d$ weight, $L$ blocks)

| 기법 | 조합 수 |
|------|------|
| Dropout | $2^N$ |
| DropConnect | $2^{d \cdot d}$ — **훨씬 큼** |
| Stochastic Depth | $2^L$ — 작지만 각 구성이 architecturally 다름 |

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — DropConnect Layer

$$\tilde{y} = (M \odot W) \cdot a, \quad M_{ij} \stackrel{\text{iid}}{\sim} \text{Bernoulli}(1-p)$$

여기서 $a$는 activation input, $M \in \{0, 1\}^{d_{\text{out}} \times d_{\text{in}}}$.

### 정리 5.2 — Dropout = DropConnect with Row-shared Mask

Activation $a_j$에 dropout mask $m_j$ 곱하는 것은 $W$의 $j$번째 **column** 전체에 같은 mask 적용:

$$y_i = \sum_j W_{ij} (m_j a_j) = \sum_j (m_j W_{ij}) a_j$$

따라서 Dropout은 DropConnect의 **column-group special case** — mask가 column별로 독립이면서 column 내부는 공유.

### 정의 5.3 — Stochastic Depth Block

Residual block $F_\ell$ (conv + BN + ReLU + conv)에 대해:

$$h_{\ell+1} = \begin{cases} h_\ell + F_\ell(h_\ell) & b_\ell = 1 \quad (\text{prob } 1-p_\ell) \\ h_\ell & b_\ell = 0 \quad (\text{prob } p_\ell) \end{cases}$$

**Linear decay schedule** (Huang 2016 권장):

$$p_\ell = \frac{\ell}{L} \cdot p_L, \quad p_L \in [0, 0.5]$$

초기 layer는 거의 drop 안 됨, 후반 layer는 drop 많음.

### 정리 5.4 — Expected Effective Depth

$$\mathbb{E}[\text{depth}] = \sum_{\ell=1}^L (1 - p_\ell) = L - \sum_\ell p_\ell$$

Linear decay 하에선 $= L - p_L L/2 = L(1 - p_L/2)$. $p_L = 0.5$면 평균 $0.75 L$.

### 정리 5.5 — DropConnect의 Variance (linear model)

Linear $y = Wx$의 DropConnect: $\tilde{y} = (M \odot W) x$. $\mathbb{E}[\tilde{y}] = (1-p) Wx$. Variance:

$$\text{Var}(\tilde{y}_i) = p(1-p) \sum_j W_{ij}^2 x_j^2$$

Dropout의 variance (activation drop): $\text{Var}(\tilde{y}_i) = p(1-p) \sum_j W_{ij}^2 x_j^2$ — 동일. 그러나 **correlation 구조**가 다름.

### 정리 5.6 — Stochastic Depth와 Residual Flow의 보존

Block이 dropped되어도 identity는 유지 → **gradient flow가 유지** (ResNet의 identity path). 일반 layer drop과 달리 정보 흐름이 끊기지 않는다. 이것이 매우 깊은 네트워크(1000+ layer)에서도 효과적인 이유.

---

## 🔬 수학적 유도

### 정리 5.2 — Dropout과 DropConnect의 관계

$y_i = \sum_j \tilde{W}_{ij} \tilde{a}_j$.

**Dropout** ($\tilde{a}_j = m_j a_j$, $\tilde{W}_{ij} = W_{ij}$):

$y_i = \sum_j W_{ij} (m_j a_j)$

**DropConnect row-shared** ($\tilde{W}_{ij} = m_j W_{ij}$, $\tilde{a}_j = a_j$):

$y_i = \sum_j (m_j W_{ij}) a_j = \sum_j m_j W_{ij} a_j$

두 식 정확히 같음. 따라서 **activation-level Dropout은 DropConnect의 column-shared mask special case**.

**Full DropConnect** ($\tilde{W}_{ij} = m_{ij} W_{ij}$, independent $m_{ij}$):

$y_i = \sum_j m_{ij} W_{ij} a_j$

서로 다른 $i$에서 다른 mask → **더 다양한 앙상블**. $\square$

### DropConnect의 기댓값 loss (linear case)

Ch2-03의 계산과 유사. $\mathbb{E}_M[\|y - \tilde{W} X^T\|^2]$을 전개:

$$\mathbb{E}[\tilde W \tilde W^T]_{ii'} = \sum_j \mathbb{E}[m_{ij} m_{i'j}] W_{ij} W_{i'j}$$

$m_{ij}, m_{i'j}$ 독립($i \neq i'$)이면 $\mathbb{E} = (1-p)^2$; $i = i'$이면 $(1-p)$.

결국 DropConnect도 **adaptive L2**-style regularization을 주지만 계수가 Dropout과 조금 다름 (더 세밀한 feature interaction penalty).

### Stochastic Depth의 effective 기댓값

$h_{\ell+1} = h_\ell + b_\ell F_\ell(h_\ell)$. $\mathbb{E}_b[h_{\ell+1}] = h_\ell + (1-p_\ell) F_\ell(h_\ell)$. 즉 기댓값으로는 $F_\ell$이 $(1-p_\ell)$-scaled로 적용. Huang 2016의 test time: $F_\ell(h_\ell) \cdot (1-p_\ell)$로 scale (Dropout의 weight scaling과 동일 원리).

---

## 💻 실험으로 효과 검증

### 실험 1 — DropConnect 구현과 Dropout 비교

```python
import torch
import torch.nn as nn

class DropConnect(nn.Module):
    def __init__(self, in_features, out_features, p=0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.p = p
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(self.linear.weight, 1 - self.p)) / (1 - self.p)
            W_masked = self.linear.weight * mask
            return x @ W_masked.T + self.linear.bias
        else:
            return self.linear(x)

# 비교: Dropout + Linear vs DropConnect
x = torch.randn(8, 100)
dropout_layer = nn.Sequential(nn.Dropout(0.3), nn.Linear(100, 10))
dropconnect_layer = DropConnect(100, 10, p=0.3)

# Mask 다양성 비교 (앙상블 크기의 proxy)
dropout_layer.train(); dropconnect_layer.train()
dropout_outs = torch.stack([dropout_layer(x) for _ in range(500)])
dropconnect_outs = torch.stack([dropconnect_layer(x) for _ in range(500)])

# 샘플 간 variance (앙상블 분산)
print("Dropout variance :", dropout_outs.var(dim=0).mean().item())
print("DropConnect var  :", dropconnect_outs.var(dim=0).mean().item())
# → DropConnect의 variance가 약간 더 큼 (더 독립적인 mask)
```

### 실험 2 — Stochastic Depth ResNet block

```python
class StochasticDepthBlock(nn.Module):
    def __init__(self, channels, drop_prob=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            if torch.rand(1).item() < self.drop_prob:
                return x     # block skipped
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu(out + x)
        else:
            # Expectation scaling
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu((1 - self.drop_prob) * out + x)

# ResNet with linearly decayed drop probabilities
def build_stoch_resnet(L=20, channels=64, p_L=0.5):
    return nn.Sequential(*[
        StochasticDepthBlock(channels, drop_prob=(l / L) * p_L) for l in range(L)
    ])

net = build_stoch_resnet()
# 훈련 시 매 batch마다 다른 effective depth
net.train()
for batch in range(3):
    x = torch.randn(2, 64, 8, 8)
    active_blocks = sum(1 for m in net if isinstance(m, StochasticDepthBlock)
                        and torch.rand(1).item() > m.drop_prob)
    print(f"Batch {batch}: active blocks = {active_blocks}/{len(net)}")
```

### 실험 3 — 세 기법의 regularization 강도 비교 (toy)

```python
import matplotlib.pyplot as plt

torch.manual_seed(0)
X = torch.randn(300, 20)
true_w = torch.randn(20); true_w[5:] = 0
y = X @ true_w + 0.2 * torch.randn(300)

def train_and_test(model, epochs=2000, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(X[:200]) - y[:200])**2).mean()
        loss.backward(); opt.step()
    model.eval()
    test_loss = ((model(X[200:]) - y[200:])**2).mean().item()
    return test_loss

results = {}
for name, builder in [
    ('No reg',      lambda: nn.Linear(20, 1)),
    ('Dropout 0.3', lambda: nn.Sequential(nn.Dropout(0.3), nn.Linear(20, 1))),
    ('DropConnect 0.3', lambda: DropConnect(20, 1, p=0.3)),
]:
    losses = []
    for seed in range(5):
        torch.manual_seed(seed)
        losses.append(train_and_test(builder()))
    results[name] = (sum(losses)/5, min(losses), max(losses))

for name, (mean, lo, hi) in results.items():
    print(f"{name:20s} test MSE = {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
```

---

## 🔗 실전 활용

### 선택 가이드

| Situation | 권장 |
|-----------|------|
| FC layer의 기본 regularization | Standard Dropout |
| CNN의 채널 수준 | Spatial Dropout (Ch2-04) |
| 아주 큰 weight matrix (e.g. FC의 수백만 param) | DropConnect — 더 fine-grained |
| 매우 깊은 ResNet (100+ layer) | Stochastic Depth — gradient flow + 훈련 가속 |
| Transformer | LayerDrop (Fan 2020): Stochastic Depth의 Transformer 버전 |

### DropConnect의 단점

**구현 비용**: Weight masking이 forward pass 속도를 늦춤 (GPU에서 elementwise mask application은 메모리 대역폭 증가).

**실전 사용 빈도**: Dropout만큼 광범위하지는 않음. 특수 효과가 필요할 때 (e.g. 매우 wide FC layer) 선택.

### Stochastic Depth의 실전 효과

- **훈련 속도**: 평균 $(1 - p_L/2) L$-depth로 훈련 → 25% 이상 빠름.
- **Generalization**: ResNet-110 CIFAR-10에서 6% error → 5% (Huang 2016).
- **GPU memory**: Drop된 block의 forward pass 생략 → memory 절약.
- **LayerDrop (Fan 2020)**: 같은 원리를 Transformer에 적용, inference 시 일부 layer 생략으로 속도 향상.

### 조합

- Dropout + Stochastic Depth: 각각 다른 granularity → **함께 쓰면 이익 (직교적)**.
- DropConnect + Dropout: 중복되어 효과 제한적.
- Spatial Dropout + Stochastic Depth: CNN ResNet에서 표준 조합.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Independent mask | 실제로 network 내부 상관성 있으면 effective 다름 |
| ResNet 구조 | Stochastic Depth는 **identity path 필요** — plain network에는 적용 불가 |
| 훈련 시 효과 | Test 시 expectation scaling의 오차 |
| 같은 $p$ 모든 feature | Adaptive depth (learnable $p$)는 Concrete Dropout 스타일로만 가능 |
| 앙상블 크기 = capacity | 실제로는 일부 mask만 "유효" — 중복 많음 |

**주의**: "앙상블 크기가 크면 regularization 강함"이 항상 true 아님. 너무 많은 subnetwork는 각각이 **학습 불충분** — underfit. 그래서 $p$를 너무 크게 두면 성능 저하.

---

## 📌 핵심 정리

$$\boxed{\text{Dropout: activation } | \text{ DropConnect: weight } | \text{ Stochastic Depth: block}}$$

| 기법 | Granularity | 앙상블 크기 | 대표 사용처 |
|------|---------|----------|----------|
| **Dropout** | neuron | $2^N$ | FC, Transformer FFN |
| **DropConnect** | weight | $2^{d \times d}$ | Very wide FC |
| **Stochastic Depth** | block | $2^L$ (다른 arch!) | Deep ResNet, LayerDrop |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $d_{\text{in}} = d_{\text{out}} = 500$, $p = 0.3$. Dropout과 DropConnect의 앙상블 크기는 각각?

<details>
<summary>힌트 및 해설</summary>

- Dropout: $2^{d_{\text{out}}} = 2^{500}$ (or $2^{d_{\text{in}}}$, depending on which side drop) ≈ $10^{150}$.
- DropConnect: $2^{d_{\text{in}} \cdot d_{\text{out}}} = 2^{250,000}$ ≈ $10^{75,257}$.

DropConnect의 "capacity"가 **천문학적으로** 크다. 하지만 실전에서 둘의 성능 차이는 훨씬 작음 — 앙상블 크기가 regularization 효과를 직접 결정하지 않음을 보여주는 예.

</details>

**문제 2** (심화): Linear decay schedule에서 $p_L = 0.5$, $L = 100$일 때 block별 drop probability와 기댓값 effective depth를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$p_\ell = (\ell / L) \cdot p_L = 0.005 \ell$. Layer 1은 drop 거의 없음, layer 100은 50% drop.

기댓값 effective depth:
$$\mathbb{E}[\text{depth}] = \sum_{\ell=1}^{100} (1 - 0.005\ell) = 100 - 0.005 \cdot \frac{100 \cdot 101}{2} = 100 - 25.25 = 74.75$$

즉 평균 75 layer만 활성 → **25% 훈련 시간 절감**. 큰 ResNet에서 의미 있는 속도 향상.

</details>

**문제 3** (이론-실전): 왜 Stochastic Depth는 **ResNet에서만** 잘 작동하는가? VGG같은 plain network에 적용하면 어떤 문제가 생기는가?

<details>
<summary>힌트 및 해설</summary>

**ResNet**: $h_{\ell+1} = h_\ell + F_\ell(h_\ell)$. Block drop 시 $h_{\ell+1} = h_\ell$ → **identity path로 정보 통과**. Gradient도 **그대로 흐름** (chain rule에서 $\partial h_{\ell+1}/\partial h_\ell = I$).

**VGG/plain network**: $h_{\ell+1} = F_\ell(h_\ell)$. Block drop = $h_{\ell+1} = ?$. Identity를 사용하면 dimension mismatch (channel 수 다름, spatial resolution 다름). 강제로 skip하면 차원 맞추기 위한 추가 layer 필요.

실전에서 VGG-style에 Stochastic Depth를 **억지로** 적용하면:
- Dimension matching을 위한 $1 \times 1$ conv 필요 → 추가 parameter.
- Gradient flow 완전히 끊김 — 훈련 불안정.

교훈: **Architectural regularization은 아키텍처와 깊이 연동**. Stochastic Depth는 ResNet의 "skip connection이 identity bias"라는 설계 선택의 장점을 추가로 활용.

Transformer에 Stochastic Depth 적용 (LayerDrop, Fan 2020): Transformer도 residual 구조라 자연스럽게 적용 가능. Tran-XL, DeiT 등에서 효과 보임.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Dropout 변종](./04-dropout-variants.md) | [📚 README로 돌아가기](../README.md) | [Chapter 3 → 01. Batch Normalization ▶](../ch3-normalization/01-batch-norm.md) |

</div>
