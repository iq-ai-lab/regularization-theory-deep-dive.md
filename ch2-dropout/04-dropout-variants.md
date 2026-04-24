# 04. Dropout 변종 — Spatial · Variational · Concrete

## 🎯 핵심 질문

- CNN에서 **element-wise Dropout**이 왜 효과가 제한적인가? Spatial Dropout은 어떻게 해결하는가?
- RNN에서 "매 step마다 다른 mask"가 왜 문제인가? **Variational Dropout**의 해결책은?
- Dropout rate $p$도 **학습**할 수 있는가? **Concrete Dropout**의 Gumbel-softmax 트릭은?
- 이 세 변종은 Ch2-02 VI 관점에서 어떻게 통합 해석되는가?

---

## 🔍 왜 변종이 필요한가

표준 Bernoulli Dropout은 FC layer의 iid hidden units을 가정한다. 현대 NN은 이 가정이 맞지 않는다:

1. **CNN**: 같은 채널 내 인접 pixel은 **강한 공간 상관** — 일부 pixel을 drop해도 이웃이 대체.
2. **RNN/Transformer**: time/position 축의 **구조 정보**가 있어 매 step 다른 mask면 시퀀스 정보가 일관성 없이 손실.
3. **Dropout rate $p$**: 보통 hyperparameter로 search — 하지만 **layer별 / feature별로 다른 rate가 최적**. Search space 폭증.

이 세 문제를 해결하는 **Spatial, Variational RNN, Concrete Dropout**을 본다. 모두 Gal-Ghahramani의 VI framework(Ch2-02)에서 variational family를 **구조적으로 제한** 또는 **rate $p$를 학습 가능화**한 것으로 해석 가능.

---

## 📐 수학적 선행 조건

- Ch2-01, Ch2-02, Ch2-03 (전체 관점의 자연스러운 확장)
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): CNN의 spatial 구조, RNN의 recurrent weight
- Gumbel-softmax / Concrete relaxation (Maddison 2017, Jang 2017): discrete variable의 미분 가능 근사
- Reparameterization trick

---

## 📖 직관적 이해

### Spatial Dropout (Tompson et al. 2015)

CNN의 conv output $h \in \mathbb{R}^{C \times H \times W}$. 표준 Dropout은 각 pixel-channel을 독립으로 drop.

**문제**: CNN feature map은 **공간적으로 smooth** — 한 pixel을 drop해도 인접 pixel이 거의 같은 정보 → noise 주입 효과 미미.

**해결**: **채널 단위 drop** — 한 번에 전체 $H \times W$ feature map을 drop. 한 채널이 사라지면 대체 불가능.

$$\tilde{h}_{c, i, j} = m_c \cdot h_{c, i, j}, \quad m_c \stackrel{\text{iid}}{\sim} \text{Bernoulli}(1-p)$$

### Variational RNN Dropout (Gal & Ghahramani 2016)

RNN $h_t = f(W_h h_{t-1} + W_x x_t)$. "Naive" RNN dropout: 매 time step마다 다른 mask.

**문제**: Time step $t$와 $t+1$에서 완전히 다른 subnetwork → recurrent computation이 **inconsistent**. 정보 전달이 붕괴.

**해결**: **시간에 걸쳐 같은 mask 공유**.

$$\tilde{h}_t = m \odot h_t, \quad m \text{ fixed for all } t$$

Gal의 VI 관점: **한 sequence 내에서 variational posterior의 같은 샘플을 유지** → consistent computation. LSTM에서 크게 유효 (언어 모델링 perplexity 개선).

### Concrete Dropout (Gal, Hron, Kendall 2017)

Dropout rate $p$를 학습 가능하게 만들고 싶다. 하지만 $z \sim \text{Bernoulli}(1-p)$은 **미분 불가능**.

**해결**: Gumbel-softmax (Concrete distribution)로 **continuous relaxation**:

$$\tilde{z} = \sigma\left(\frac{1}{\tau}\left(\log \frac{1-p}{p} + \log u - \log(1-u)\right)\right), \quad u \sim U(0, 1)$$

$\tau \to 0$에서 $\tilde{z} \to \text{Bernoulli}(1-p)$, $\tau > 0$에서는 smooth approximation. $p$에 대한 gradient 가능 → **layer별 dropout rate 학습**.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Spatial Dropout

Conv output $h \in \mathbb{R}^{B \times C \times H \times W}$에 대해:

$$\tilde{h}_{b, c, i, j} = m_{b, c} \cdot h_{b, c, i, j}, \quad m_{b, c} \sim \text{Bernoulli}(1-p)$$

각 (batch, channel) 쌍마다 하나의 scalar mask. Element-wise dropout보다 **거친** (coarser) 규모.

### 정의 4.2 — Variational RNN Dropout

Sequence $(x_1, \ldots, x_T)$의 hidden states $(h_1, \ldots, h_T)$에 대해:

$$\tilde{h}_t = m^h \odot h_t, \quad \tilde{x}_t = m^x \odot x_t, \quad \forall t \in [1, T]$$

$m^h, m^x$는 sequence 시작 시 한 번 샘플되어 **전체 sequence 동안 고정**. 각 sequence마다 다시 샘플.

### 정의 4.3 — Concrete Distribution

Temperature $\tau > 0$, location $\alpha = \log \pi / (1-\pi)$ (logit of keep probability $\pi$)의 **BinaryConcrete**$(\alpha, \tau)$:

$$\tilde{z} = \sigma\left(\frac{\alpha + L}{\tau}\right), \quad L = \log U - \log(1-U), \ U \sim U(0, 1)$$

($L$은 **standard logistic** 분포.) $\tau \to 0$에서 $\tilde{z} \to \text{Bernoulli}(\pi)$ in distribution.

### 정의 4.4 — Concrete Dropout Layer

$$\tilde{h} = \tilde{z} \odot h, \quad \tilde{z} \sim \text{BinaryConcrete}(\alpha, \tau)$$

Trainable parameter: $\alpha$ (per-layer scalar). Loss에 **regularization term**:

$$\mathcal{L}_{\text{reg}}(\alpha) = -K \cdot [\pi \log \pi + (1-\pi) \log(1-\pi)] \cdot \text{const}$$

이는 entropy-based penalty로, **너무 낮거나 높은 $p$를 방지** (uniform Bernoulli prior의 KL).

### 정리 4.5 — Spatial Dropout의 Variational Interpretation

Gal 2016의 VI framework에서 variational family를 제한:

$$q(W_l) = M_l \text{diag}(z_l), \quad z_{l, i} = z_{l, i'} \text{ for all pixels } i, i' \text{ in same channel}$$

(같은 채널의 pixel들은 같은 mask 공유.) 이는 Dropout의 "channel group"에 대한 **group-level** VI.

### 정리 4.6 — Concrete Dropout의 미분 가능성

$\nabla_\alpha \tilde{z}$는 chain rule로 계산 가능:

$$\frac{\partial \tilde z}{\partial \alpha} = \frac{1}{\tau} \tilde z (1 - \tilde z)$$

이것이 $\alpha$(따라서 $p$)를 **gradient descent로 최적화 가능**하게 만든다.

---

## 🔬 수학적 유도

### 왜 Spatial Dropout이 CNN에 유효한가

CNN feature map의 spatial correlation 구조:

$$\text{Cov}(h_{c, i, j}, h_{c, i+1, j}) \approx \text{Var}(h_{c, i, j}) \cdot \rho$$

여기서 $\rho$는 공간적 correlation (보통 높다, 0.7~0.9).

**Element-wise dropout**의 information loss: $h$를 한 pixel drop해도 인접 pixel이 **$\rho$-correlated copy**를 준다. Effective information removed: $1 - \rho^2$.

**Spatial dropout**: 전체 채널 drop → 정보 회복 가능성 $\approx 0$ (채널 간 correlation은 낮음). Effective information removed: $\approx 1$.

결론: 같은 mask probability에서 spatial dropout이 **훨씬 강한 regularization** 효과.

### Variational RNN Dropout의 VI 정당화

RNN의 weight $W$를 time-invariant로 가정 (실제 그러함). VI posterior:

$$q(W) = M \text{diag}(z), \quad z \sim \text{Bernoulli}(1-p)$$

Sequence 입력 $(x_1, \ldots, x_T)$의 likelihood $p(y|x_{1:T}, W)$는 **같은 $W$**에 반복 적용. 따라서 MC estimator:

$$\mathbb{E}_q[\log p(y|x, W)] \approx \log p(y|x, W^{(s)}), \quad W^{(s)} \sim q$$

**한 sample per sequence** — 즉 **sequence 전체에 같은 mask**. 매 step 다른 mask는 $W^{(s)}$가 step마다 다르다는 의미로, VI의 "same $W$ across sequence" 가정을 깨뜨린다. Gal 2016은 이 구분이 LSTM에서 perplexity를 크게 개선함을 실험적으로 보임 (PTB).

### Gumbel-Softmax 근사 유도

Bernoulli($\pi$) 샘플은 $z = \mathbb{1}[U < \pi]$ ($U \sim U(0, 1)$)로 쓸 수 있다. $\sigma(\text{logit})$의 구조를 이용한 reparameterization:

$$z \stackrel{d}{=} \lim_{\tau \to 0^+} \sigma\left(\frac{\log(\pi/(1-\pi)) + \log U - \log(1-U)}{\tau}\right)$$

Sigmoid와 logistic noise의 조합이 $\tau \to 0$에서 indicator로 수렴.

$\tau > 0$: smooth approximation — **chain rule이 $\pi$에 대해 작동**. 훈련 후 $\tau \to 0$ (혹은 실전에서 $\tau = 0.1$ 정도).

---

## 💻 실험으로 효과 검증

### 실험 1 — Spatial vs Element-wise Dropout (CIFAR-10 CNN)

```python
import torch
import torch.nn as nn

class SpatialDropout2d(nn.Module):
    """nn.Dropout2d와 동일 (channel 단위 drop)."""
    def __init__(self, p): super().__init__(); self.drop = nn.Dropout2d(p)
    def forward(self, x): return self.drop(x)

class ElementDropout2d(nn.Module):
    def __init__(self, p): super().__init__(); self.drop = nn.Dropout(p)
    def forward(self, x):
        B, C, H, W = x.shape
        return self.drop(x.view(B, -1)).view(B, C, H, W)

# Test: 인접 pixel의 상관성 측정
x = torch.randn(16, 32, 8, 8)
x = nn.Conv2d(32, 32, 3, padding=1)(x)   # smooth feature map

sd = SpatialDropout2d(0.3); sd.train()
ed = ElementDropout2d(0.3); ed.train()

# Mask pattern 시각화 (첫 샘플, 첫 채널)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x_sd = sd(x); x_ed = ed(x)
mask_sd = (x_sd[0, 0] == 0).float()
mask_ed = (x_ed[0, 0] == 0).float()
axes[0].imshow(mask_sd, cmap='gray'); axes[0].set_title('Spatial — 전체 채널 on/off')
axes[1].imshow(mask_ed, cmap='gray'); axes[1].set_title('Element — 무작위 위치')
plt.show()
```

**관찰**: Spatial은 "전체 패치 사라짐", element는 "일부 pixel만 군데군데". Spatial이 더 강한 규칙화 효과.

### 실험 2 — Variational RNN Dropout 구현

```python
class VariationalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, p=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.p = p

    def forward(self, xs):   # xs: (T, batch, input_size)
        T, B, _ = xs.shape
        # Variational mask: sequence 시작 시 1번 샘플, time 동안 고정
        if self.training:
            mask_h = torch.bernoulli(torch.full((B, self.hidden_size), 1 - self.p)) / (1 - self.p)
            mask_x = torch.bernoulli(torch.full((B, xs.shape[-1]), 1 - self.p)) / (1 - self.p)
        else:
            mask_h = mask_x = 1.0
        h = torch.zeros(B, self.hidden_size); c = torch.zeros(B, self.hidden_size)
        outputs = []
        for t in range(T):
            x_t = xs[t] * mask_x if self.training else xs[t]
            h_t_masked = h * mask_h if self.training else h
            h, c = self.lstm_cell(x_t, (h_t_masked, c))
            outputs.append(h)
        return torch.stack(outputs)

# 비교: naive RNN dropout은 time별로 다른 mask
class NaiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, p=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.drop = nn.Dropout(p)
    def forward(self, xs):
        T, B, _ = xs.shape
        h = torch.zeros(B, self.hidden_size); c = torch.zeros(B, self.hidden_size)
        outputs = []
        for t in range(T):
            # 매 t마다 다른 mask!
            x_t = self.drop(xs[t])
            h, c = self.lstm_cell(x_t, (self.drop(h), c))
            outputs.append(h)
        return torch.stack(outputs)

# Language modeling 실험: Variational이 perplexity가 낮음 (Gal 2016)
```

### 실험 3 — Concrete Dropout으로 rate 학습

```python
class ConcreteDropout(nn.Module):
    def __init__(self, init_logit=-2.0, temperature=0.1, reg_weight=1e-4):
        super().__init__()
        self.logit_p = nn.Parameter(torch.tensor(init_logit))  # init: σ(-2) ≈ 0.12
        self.temperature = temperature
        self.reg_weight = reg_weight

    def forward(self, x):
        if self.training:
            p = torch.sigmoid(self.logit_p)
            u = torch.rand_like(x).clamp(1e-6, 1 - 1e-6)
            z_concrete = torch.sigmoid(
                (self.logit_p + torch.log(u) - torch.log(1 - u)) / self.temperature
            )
            return x * z_concrete / (1 - p.detach())
        else:
            return x

    def reg(self):
        p = torch.sigmoid(self.logit_p)
        # Entropy term + prior KL (간략화)
        return self.reg_weight * (p * torch.log(p.clamp_min(1e-10)) + (1-p) * torch.log((1-p).clamp_min(1e-10)))

# Layer별로 Concrete Dropout 삽입하고 reg term을 loss에 추가
# 훈련 후 각 layer의 learned p 확인 — 보통 input은 낮은 p, deep에서는 높은 p 경향
```

---

## 🔗 실전 활용

### 선택 가이드

| Context | 권장 |
|---------|------|
| CNN (ResNet, VGG) | **Spatial Dropout** (`nn.Dropout2d`) — channel 단위 |
| LSTM/GRU (언어 모델) | **Variational Dropout** — time 축 공유 mask |
| Transformer (self-attention) | **Standard Dropout** on attention weights / FFN outputs |
| Bayesian uncertainty 중요 | **Concrete Dropout** — rate 학습으로 uncertainty 더 정확 |
| 작은 네트워크 + 제한된 compute | Standard Dropout으로 충분 |

### PyTorch 구현 메모

- `nn.Dropout2d` = Spatial Dropout for 4D tensors.
- PyTorch의 `nn.LSTM` 내장 dropout은 **변종 아님** — 각 layer 사이에만 적용, time 축이 아닌 layer 축 dropout. Variational RNN dropout은 custom LSTMCell로 구현 필요.
- Transformer의 "attention dropout"은 softmax 확률에 dropout → effectively sparse attention. 이는 위 어떤 변종도 아닌 특수 응용.

### 현대 LLM에서

GPT/Llama 같은 매우 큰 Transformer에서는 **dropout이 거의 쓰이지 않는다** — data/compute scale이 충분해 추가 regularization 불필요, inference 시 MC Dropout의 비용이 감당 안 됨. 대신 **data augmentation (text-level)**, **weight decay**, **label smoothing** 조합.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Spatial correlation 높음 | 작은 feature map(8x8 이하)에서는 효과 감소 |
| RNN의 time-invariant $W$ | Time-varying gating(LSTM cell state)에서 섬세한 해석 필요 |
| Concrete의 continuous relaxation | $\tau$ 조정 필요, 훈련·inference 일관성 issue |
| Learned $p$ | Concrete Dropout은 small batch에서 수렴 불안정 |

---

## 📌 핵심 정리

$$\boxed{\text{Spatial: channel mask } | \text{ Variational: time-shared mask } | \text{ Concrete: learnable } p}$$

| 변종 | 해결 문제 | 핵심 메커니즘 |
|------|---------|-------------|
| **Spatial Dropout** | CNN의 spatial redundancy | channel 단위 drop |
| **Variational RNN Dropout** | RNN의 sequence 일관성 | time 축 mask 공유 |
| **Concrete Dropout** | $p$ 수동 tuning | Gumbel-softmax로 $p$ 학습 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 32×32 feature map, 64 channel에서 $p = 0.3$일 때 element-wise와 spatial dropout 각각 평균 몇 개의 원소가 0이 되는가?

<details>
<summary>힌트 및 해설</summary>

Total elements: $32 \times 32 \times 64 = 65,\!536$.

- Element-wise: 각 원소 독립 → $0.3 \times 65536 \approx 19,\!660$.
- Spatial: 각 채널 독립 → $0.3 \times 64 \approx 19$ channels drop, 그 채널들 전체 → $19 \times 1024 \approx 19,\!500$.

**거의 같은 원소 수**가 0이 되지만 **공간 구조가 완전히 다름**. Spatial은 19개 channel이 완전히 사라짐 (정보 완전 손실), element-wise는 각 채널의 일부 pixel만 손실 (인접 pixel로 recovery 가능).

</details>

**문제 2** (심화): Gumbel-softmax 근사에서 $\tau = 0.01$로 매우 작게 두면 어떤 문제가 생기는가? $\tau$ annealing 전략의 의미는?

<details>
<summary>힌트 및 해설</summary>

$\tau \to 0$에서:
- 장점: $\tilde z$가 $\{0, 1\}$에 가까움 → 실제 Bernoulli와 close.
- 단점: $\nabla_\alpha \tilde z$이 **극도로 sharp** — gradient가 대부분 0 또는 explode. Training 불안정.

**해결**: **temperature annealing** — 초기에는 큰 $\tau$ (e.g. 1.0)로 smooth gradient, 훈련 진행하며 $\tau \to 0.1$로 감소. Softmax temperature scheduling과 유사.

실전: Jang et al. 2017은 $\tau$를 0.5에서 0.1로 linearly decay. 그러면 early training은 exploratory, late training은 exploitative.

</details>

**문제 3** (이론-실전): Transformer의 attention dropout은 softmax 이후의 attention weight에 dropout. 이것은 어떤 regularization 효과를 주는가? Spatial vs Variational vs Concrete 중 어느 것과 가장 유사한가?

<details>
<summary>힌트 및 해설</summary>

Attention matrix $A = \text{softmax}(QK^T / \sqrt{d})$. $A$의 각 (i, j) entry에 dropout → "token $i$가 token $j$를 attend하는 연결을 확률 $p$로 끊는다".

**유사성**:
- **Spatial에 약간 유사**: attention head는 "channel"과 같음, 하지만 attention dropout은 channel level이 아닌 connection level.
- **"Edge dropout"이라 부르는 것이 더 정확** — graph neural network의 dropout-edge와 유사.
- **Variational**은 아님 — time 축 공유가 아니라 (i, j) pair별 독립.

효과:
- 훈련 중 attention이 **임의의 token을 놓칠 수도 있다** → 모델이 여러 attention head에 redundancy를 퍼뜨림.
- Dropout mask의 평균 = 원래 attention weight의 scaled version.

실전: GPT-3 같은 큰 모델에서는 attention dropout $p \approx 0.1$이 일반적. 너무 크면 훈련 불안정.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Dropout = Adaptive L2](./03-dropout-adaptive-l2.md) | [📚 README로 돌아가기](../README.md) | [05. DropConnect vs Stochastic Depth ▶](./05-dropout-dropconnect-stochdepth.md) |

</div>
