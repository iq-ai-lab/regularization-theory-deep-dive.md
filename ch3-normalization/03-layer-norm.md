# 03. Layer Normalization (Ba et al. 2016)

## 🎯 핵심 질문

- LayerNorm은 어떤 축에서 정규화하는가? BN과의 **축의 차이**는 실용적으로 무엇을 의미하는가?
- 왜 **RNN, Transformer에서 LN이 표준**인가? BN이 왜 부적합한가?
- **Pre-LN Transformer vs Post-LN**의 gradient flow 차이는?
- Xiong 2020이 증명한 pre-LN의 수학적 장점은?

---

## 🔍 왜 LayerNorm이 등장했는가

BN의 두 가지 치명적 한계:

1. **Batch size 의존**: $\mu_B, \sigma_B^2$가 batch size 작으면 noise. RNN의 sequence training에서는 batch 축이 애매.
2. **Inference inconsistency**: Running stats가 train과 test distribution이 다르면 어긋남.

Ba, Kiros, Hinton 2016은 **feature 축으로 정규화**하는 LayerNorm을 제안. 효과:
- Batch size에 무관.
- Sequence length 달라도 안정 (각 token 독립 정규화).
- Running stats 불필요 (train/eval 동일).

결과: **Transformer의 사실상 표준** (Attention is All You Need, Vaswani 2017부터). 현재 거의 모든 LLM이 LN 혹은 그 변종(RMSNorm, Ch3-06)을 사용.

이 문서는 LN의 수식과 특성, 그리고 Transformer에서의 **pre-LN vs post-LN 논쟁** 을 정리한다.

---

## 📐 수학적 선행 조건

- Ch3-01: BN의 수식, forward/backward
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Transformer의 attention + FFN 구조
- 선형대수: 1D Gaussian 정규화

---

## 📖 직관적 이해

### BN vs LN의 축 차이

Input $x \in \mathbb{R}^{B \times D}$ ($B$: batch, $D$: feature):

| 기법 | Statistics 계산 축 | 결과 |
|------|----------|--------|
| **BN** | batch 축 ($B$) | 각 feature가 batch 내에서 정규화 |
| **LN** | feature 축 ($D$) | 각 sample이 feature 내에서 정규화 |

- BN: "모든 sample의 첫 feature를 같이 정규화" — feature-wise.
- LN: "sample 1의 모든 feature를 같이 정규화" — sample-wise.

### 왜 LN이 sequence에 맞는가

RNN: $h_t = f(W_h h_{t-1} + W_x x_t)$. 매 step에서 hidden $h_t$를 정규화하려면:
- BN: batch 내 같은 time-step $t$의 $h_t$들로 통계. 하지만 **time-varying batch statistics** — sequence 길이가 다르면 통계 다름.
- LN: 각 $h_t$를 **그 자체의 feature 분포**로 정규화. Time-step, sequence length 모두 독립.

Transformer의 각 token attention / FFN output도 같은 이유로 LN 선호.

### Pre-LN vs Post-LN Transformer

**Post-LN** (원 Transformer, 2017): $x_{\ell+1} = \text{LN}(x_\ell + F(x_\ell))$  
**Pre-LN** (Xiong 2020): $x_{\ell+1} = x_\ell + F(\text{LN}(x_\ell))$

**그림**:
```
Post-LN:  x → F → (+) → LN → next
          ↑________↑
          residual

Pre-LN:   x → LN → F → (+) → next
                         ↑
                      residual (x 그대로)
```

**Xiong 2020의 발견**: Post-LN은 deep Transformer에서 **warmup 없이 훈련 불안정**. Pre-LN은 warmup 없이도 수렴. 이유: Pre-LN의 gradient가 **layer-wise bounded**, Post-LN은 $O(\sqrt{L})$로 증폭.

현재 (2024 기준) LLaMA, GPT 계열 모두 pre-LN (혹은 pre-RMSNorm).

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — LayerNorm Operation

Input $x \in \mathbb{R}^D$ (single sample, $D$ features), hyperparam $\epsilon > 0$:

$$\mu = \frac{1}{D}\sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$$

여기서 $\gamma, \beta \in \mathbb{R}^D$ 학습 가능 (BN처럼).

Sequence input $x \in \mathbb{R}^{B \times T \times D}$: 각 $(b, t)$ 위치에서 $D$-축으로 정규화.

### 정의 3.2 — BatchNorm vs LayerNorm Axis

$x \in \mathbb{R}^{B \times D}$:

$$\text{BN: } \mu_j = \frac{1}{B}\sum_i x_{ij}, \quad \text{LN: } \mu_i = \frac{1}{D}\sum_j x_{ij}$$

BN은 **column-wise**, LN은 **row-wise**.

### 정리 3.3 — LN의 Train/Eval 일관성

LN에서는 running statistics가 없음. $y = \text{LN}(x)$가 **train과 eval에서 동일한 함수**.

### 정의 3.4 — Transformer Block의 Pre-LN / Post-LN

Input $x$, attention $A$, FFN $F$:

$$\text{Post-LN: } x' = \text{LN}(x + A(x)), \ y = \text{LN}(x' + F(x'))$$

$$\text{Pre-LN: } x' = x + A(\text{LN}(x)), \ y = x' + F(\text{LN}(x'))$$

### 정리 3.5 — Post-LN Gradient Explosion (Xiong et al. 2020)

$L$-layer Post-LN Transformer에서, 초기화 시:

$$\|\nabla_{x_0} y\| = \Theta(\sqrt{L})$$

즉 depth 증가에 따라 gradient norm이 **$\sqrt{L}$**로 증가. Pre-LN은:

$$\|\nabla_{x_0} y\| = \Theta(1)$$

**depth 무관**. 이것이 Post-LN이 learning rate warmup을 필수로 하는 이유.

### 정리 3.6 — LN Scale Invariance

$y = \text{LN}(x)$에 대해 $x \to c x$ ($c > 0$) 적용해도 $y$ 불변. 이유: $\mu, \sigma$가 같이 $c$배 되어 cancellation.

**함의**: LN이 있는 layer는 **input scale에 완전 invariant** (BN과 같은 property, 단 axis 다름).

---

## 🔬 수학적 유도

### 정리 3.5 증명 스케치

Post-LN: $x_{\ell+1} = \text{LN}(x_\ell + F_\ell(x_\ell))$. $F_\ell$이 small (초기화에서 $\approx 0$)이면 $x_{\ell+1} \approx \text{LN}(x_\ell)$.

$\partial x_{\ell+1}/\partial x_\ell \approx J_{\text{LN}}$ (LN Jacobian). LN's Jacobian은 $\|\cdot\| = 1$ 근방이지만 **방향성 있는 scaling** — specific direction에서 $\sqrt{L}$ 성장.

반면 Pre-LN: $x_{\ell+1} = x_\ell + F_\ell(\text{LN}(x_\ell))$. $F_\ell \approx 0$이면 $x_{\ell+1} \approx x_\ell$. Jacobian = $I + O(\text{small})$, product가 **bounded**.

엄밀 증명은 각 direction의 perturbation에 대한 statistical 분석 (Xiong et al. 2020 Theorem 1). $\square$

**핵심 교훈**: Pre-LN의 residual path가 **직접 $x$를 통과** → gradient가 $I$-like 행렬을 곱 → explosion 없음.

### LN의 Forward와 Backward

Forward: Ch3-01 BN과 축만 바꾸면 된다.

Backward: $\hat{x}_i$가 $\mu, \sigma$를 통해 모든 $x_j$에 의존:

$$\frac{\partial \hat{x}_i}{\partial x_j} = \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{D} - \frac{\hat{x}_i \hat{x}_j}{D}\right)$$

전체 $D$-축 정규화를 사용하므로 **feature 간 coupling이 있음** — 한 feature의 변화가 모든 feature gradient에 영향.

---

## 💻 실험으로 효과 검증

### 실험 1 — BN vs LN 축의 차이 확인

```python
import torch
import torch.nn as nn

x = torch.randn(8, 16)     # batch=8, features=16

# BN: 각 feature column이 zero mean, unit variance
bn = nn.BatchNorm1d(16); bn.train()
y_bn = bn(x)
print("BN column stats (should be ≈0, ≈1):")
print("  mean:", y_bn.mean(dim=0)[:3].tolist())
print("  std :", y_bn.std(dim=0)[:3].tolist())

# LN: 각 sample row가 zero mean, unit variance
ln = nn.LayerNorm(16)
y_ln = ln(x)
print("\nLN row stats (should be ≈0, ≈1):")
print("  mean:", y_ln.mean(dim=1)[:3].tolist())
print("  std :", y_ln.std(dim=1)[:3].tolist())
```

### 실험 2 — Small batch에서 BN vs LN 안정성

```python
import matplotlib.pyplot as plt
import numpy as np

D = 32
def simulate_normalization(batch_size, n_trials=100):
    results_bn, results_ln = [], []
    for _ in range(n_trials):
        x = torch.randn(batch_size, D)
        # BN (batch statistics 사용)
        mu_bn, var_bn = x.mean(0), x.var(0, unbiased=False)
        y_bn = (x - mu_bn) / torch.sqrt(var_bn + 1e-5)
        # LN
        mu_ln, var_ln = x.mean(1, keepdim=True), x.var(1, keepdim=True, unbiased=False)
        y_ln = (x - mu_ln) / torch.sqrt(var_ln + 1e-5)
        # 각 output의 분산이 1에서 얼마나 벗어나는지 측정
        results_bn.append(y_bn.var(0).std().item())
        results_ln.append(y_ln.var(1).std().item())
    return np.mean(results_bn), np.mean(results_ln)

batch_sizes = [1, 2, 4, 8, 16, 64]
bns, lns = [], []
for b in batch_sizes:
    bn_v, ln_v = simulate_normalization(b)
    bns.append(bn_v); lns.append(ln_v)
    print(f"batch={b:3d}: BN noise={bn_v:.4f}, LN noise={ln_v:.4f}")

# → BN은 batch size ↓일수록 매우 noisy, LN은 일정
```

### 실험 3 — Pre-LN vs Post-LN Transformer 훈련 안정성

```python
class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads=4, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x):
        if self.pre_ln:
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x, x, x)[0])
            x = self.ln2(x + self.ffn(x))
        return x

# 12-layer Transformer, lr = 1e-3 (warmup 없이)
# Pre-LN은 수렴, Post-LN은 발산 또는 매우 느림
```

### 실험 4 — LN Scale Invariance

```python
x = torch.randn(4, 16)
ln = nn.LayerNorm(16)
y1 = ln(x)
y2 = ln(100.0 * x)
print("Scale invariance:")
print("  max |y1 - y2|:", (y1 - y2).abs().max().item())
# → 매우 작음 (정리 3.6)
```

---

## 🔗 실전 활용

### Transformer Recipe

현재 (2024 기준) 표준:
- **Pre-LN** (or Pre-RMSNorm, Ch3-06)
- **Warmup + Cosine LR**
- **AdamW** (Ch7-03)

Llama / Mistral / Qwen / GPT-2/3/4 계열 모두 이 조합.

### 언제 BN, 언제 LN

| Task | 권장 |
|------|------|
| ImageNet CNN | BatchNorm |
| Object detection, small batch | GroupNorm (Ch3-04) |
| RNN language model | LayerNorm |
| Transformer (encoder/decoder) | LayerNorm 또는 RMSNorm |
| GAN training | Instance Norm 또는 Layer Norm |
| Graph Neural Network | LayerNorm (node-feature 축) |

### 흔한 실수

1. **Transformer에 BN 사용**: gradient instability, 훈련 실패 가능.
2. **LN 위치 오류**: Pre vs Post를 섞어 쓰면 의도치 않은 행동.
3. **LN에 running stats 추가 시도**: LN은 필요 없음 — train/eval 자동으로 같음.
4. **LN의 $\gamma, \beta$를 tie 시도**: 모든 feature에 scalar $\gamma$만 두면 표현력 상실 (효과는 RMSNorm으로 재등장, Ch3-06).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Feature 축 정규화가 의미 | Feature들이 같은 scale로 의미 있을 때 유효 |
| $D$ 적당히 큼 | 작은 $D$ (e.g. 2, 3)에서는 statistics가 불안정 |
| Pre-LN이 항상 옳음 | Post-LN이 더 깊은 표현 학습하는 case 있음 (narrow-Post, Wang 2022) |
| Transformer에만 중심 | Vision Transformer는 BN이나 GroupNorm과 섞어 쓰기도 |

---

## 📌 핵심 정리

$$\boxed{\text{LN: feature-axis 정규화 — batch independent, pre-LN Transformer 표준}}$$

| 개념 | 의미 |
|------|------|
| **LN axis** | feature 차원 $D$ (BN의 batch 축과 대조) |
| **No running stats** | train = eval (BN과 차별점) |
| **Batch independence** | batch size 1에서도 안정 |
| **Pre-LN 장점** | Xiong 2020 — gradient $O(1)$ (Post는 $O(\sqrt{L})$) |
| **다음 질문** | GN, IN, WN의 다른 축과 방식 → Ch3-04 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $D = 4$, $x = (1, 3, 5, 7)$일 때 LN의 $\hat{x}$를 계산하라 ($\epsilon = 0, \gamma = 1, \beta = 0$).

<details>
<summary>힌트 및 해설</summary>

$\mu = (1+3+5+7)/4 = 4$. $\sigma^2 = ((-3)^2 + (-1)^2 + 1^2 + 3^2)/4 = 20/4 = 5$, $\sigma = \sqrt{5}$.

$\hat{x} = (x - 4)/\sqrt{5} = (-3, -1, 1, 3)/\sqrt{5} \approx (-1.34, -0.45, 0.45, 1.34)$.

확인: $\sum \hat{x}_i = 0$, $\sum \hat{x}_i^2 / 4 = 20/5/4 = 1$ ✓.

</details>

**문제 2** (심화): Post-LN Transformer를 warmup 없이 훈련하면 무슨 일이 일어나는가? Xiong 2020의 "gradient size blow-up"을 수치적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

12-layer Post-LN에서 output gradient norm $\approx \sqrt{12} \approx 3.46$배. 24-layer면 $\sqrt{24} \approx 4.9$배.

**실전 영향**: lr = 1e-3 (작게 보임)이지만 gradient가 3.5배이면 effective update가 3.5e-3로 폭발. First few steps에서 파라미터가 너무 많이 움직여 local minimum 밖으로 날아감.

**Warmup의 역할**: 처음 몇 steps에서 lr를 천천히 증가 (0부터 target까지) → 초기의 big gradient가 nominal lr와 곱해질 때 absolute update가 안전.

Pre-LN이 warmup 덜 필요한 이유: gradient가 $O(1)$이므로 lr을 더 aggressive하게 설정 가능.

</details>

**문제 3** (이론-실전): LN이 있는 Transformer에서 **weight decay**의 올바른 적용법은 무엇인가? $\gamma, \beta$에도 weight decay를 주는 것이 맞는가?

<details>
<summary>힌트 및 해설</summary>

일반적으로 **$\gamma, \beta$에는 weight decay를 주지 않는다**. 이유:

1. **$\gamma, \beta$는 normalization의 표현력 복원** — 0으로 shrink되면 정보 손실.
2. **$\gamma = 1, \beta = 0$ 초기값이 좋은 default**. Weight decay는 이를 0 방향으로 끌어당겨 효과 약화.
3. **Bias term과 유사**: bias도 보통 wd 안 줌.

PyTorch 관습:
```python
# 올바른 wd 적용
decay, no_decay = [], []
for n, p in model.named_parameters():
    if 'norm' in n or 'bias' in n:
        no_decay.append(p)
    else:
        decay.append(p)
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0}
])
```

GPT/Llama 훈련 코드에서 이 관습 확인 가능. $\gamma, \beta$ 외에도 embedding layer의 weight도 보통 wd 제외.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Santurkar 2018 반박](./02-santurkar-refutation.md) | [📚 README로 돌아가기](../README.md) | [04. GN · IN · WN ▶](./04-gn-in-wn.md) |

</div>
