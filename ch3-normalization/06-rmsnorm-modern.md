# 06. RMSNorm과 현대 Transformer

## 🎯 핵심 질문

- **RMSNorm** (Zhang & Sennrich 2019)이 LN에서 **centering을 제거**한 이유는?
- 수식 차이와 계산 효율: 왜 $\sim$25% 빠른가?
- LLama, Mistral, Qwen 같은 **현대 LLM이 RMSNorm을 채택**한 이유는?
- **Pre-RMSNorm + SwiGLU + RoPE** 조합의 의미는?

---

## 🔍 왜 "LayerNorm에서 centering 제거"인가

LayerNorm의 두 조작:
1. **Centering** ($x - \mu$): mean을 0으로.
2. **Rescaling** ($/\sigma$): variance를 1로.

**Zhang & Sennrich 2019의 관찰**: 많은 실험에서 **centering**은 학습 동역학에 **거의 기여하지 않으며** rescaling만으로도 BN/LN 수준의 효과를 얻는다.

**제안**: RMSNorm — centering 없이 **RMS(root mean square)로 정규화**.

$$\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_i x_i^2}, \quad \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

- $\beta$ (bias) 없음 (centering이 없으므로).
- $\gamma$만 학습 (per-dimension scale).

**결과**:
- 계산 감소: mean 계산과 한 번의 subtraction 제거. $\sim$7-64% 효율 개선 (model 크기에 따라).
- 성능: LN과 거의 동등, 가끔 약간 더 좋음.
- **현대 LLM의 표준**: Llama (2023), Mistral, Qwen, Gemma 모두 RMSNorm.

이 문서는 RMSNorm의 수식과 현대 Transformer stack에서의 역할을 정리한다.

---

## 📐 수학적 선행 조건

- Ch3-03: LayerNorm의 수식과 pre/post 차이
- Transformer architecture (Neural Network Theory Deep Dive)
- 기본 확률: RMS vs standard deviation의 관계

---

## 📖 직관적 이해

### RMS vs Standard Deviation

$D$-dim vector $x$:
- $\sigma_x = \sqrt{\frac{1}{D}\sum(x_i - \mu)^2}$ (std with centering).
- $\text{RMS}(x) = \sqrt{\frac{1}{D}\sum x_i^2}$ (RMS without centering).

만약 $\mu = 0$ (centered data)이면 $\sigma = \text{RMS}$. 실전 Transformer의 residual stream은 **주로 zero-centered**이므로 (residual summation의 cancellation), $\sigma \approx \text{RMS}$. Centering이 잉여.

### 계산 비용 비교

LayerNorm (PyTorch implementation):
1. $\mu = \text{mean}(x)$ — 1 reduction.
2. $\text{var} = \text{mean}((x - \mu)^2)$ — 1 subtraction + 1 reduction.
3. $\hat{x} = (x - \mu) / \sqrt{\text{var} + \epsilon}$ — 1 subtraction + 1 division.
4. $y = \gamma \hat{x} + \beta$.

RMSNorm:
1. $\text{rms} = \sqrt{\text{mean}(x^2)}$ — 1 reduction.
2. $y = \gamma \cdot x / (\text{rms} + \epsilon)$.

**2단계 제거** → ~25% faster (exact speedup varies; Zhang-Sennrich 2019 reports 7-64%).

### 현대 Transformer Stack

Llama-style:

```
x → Pre-RMSNorm → Attention (with RoPE) → (+) → x
x → Pre-RMSNorm → SwiGLU FFN → (+) → x'
```

- **Pre-RMSNorm**: pre-normalization (Ch3-03).
- **RoPE** (Rotary Position Embedding): attention 내 position encoding.
- **SwiGLU**: FFN activation (GELU 대신).
- **AdamW + cosine schedule + weight decay 0.1**.

이 조합이 **LLM 훈련의 de facto 표준**.

---

## ✏️ 엄밀한 정의·정리

### 정의 6.1 — RMSNorm (Zhang & Sennrich 2019)

$x \in \mathbb{R}^D$에 대해:

$$\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2 + \epsilon}$$

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}$$

$\gamma \in \mathbb{R}^D$ learnable. **$\beta$ 없음**.

### 정의 6.2 — SwiGLU Activation (Shazeer 2020)

FFN의 activation을 **gated linear unit + Swish**로:

$$\text{SwiGLU}(x, W, V, W_2) = W_2 \cdot (\text{SiLU}(xW) \odot xV)$$

여기서 $\text{SiLU}(z) = z \cdot \sigma(z)$ (Swish). 두 branch: 하나는 SiLU 거치고, 하나는 그대로, elementwise 곱.

- 기존 GELU FFN: $W_2 \text{GELU}(W_1 x)$.
- SwiGLU: 두 개의 입력 projection + gating.

### 정리 6.3 — RMSNorm과 LN의 Equivalence on Centered Data

$\mu(x) = 0$이면 $\text{LN}(x) \cdot \gamma/\gamma_{\text{LN}} + \beta = \text{RMSNorm}(x)$ (up to $\beta$ term).

**함의**: Residual stream이 centered이면 두 기법 **정확히 동치**. 훈련에서 $x$가 점진적으로 centered되도록 optimizer가 학습.

### 정리 6.4 — RMSNorm의 Scale Invariance

$x \to c x$ 적용 시 $\text{RMSNorm}(cx) = \text{RMSNorm}(x)$. LN의 scale invariance(정리 3.6)와 동일.

Centering은 **shift invariance**를 주지만, RMSNorm은 **shift invariance 없음**. 그러나 residual stream에서 shift는 드물어 실전 문제 없음.

### 정리 6.5 — RMSNorm의 Backward Pass

$\partial \text{RMS}/\partial x_j = x_j/(D \cdot \text{RMS})$.

$\partial y_i/\partial x_j = (\gamma_i/\text{RMS})\left(\delta_{ij} - \frac{x_i x_j}{D \text{RMS}^2}\right)$.

LN backward보다 **항 수가 적음** — coupling 구조는 유사.

---

## 🔬 수학적 유도

### 정리 6.5 유도

$r = \text{RMS}(x) = (1/D \sum x_i^2 + \epsilon)^{1/2}$. $\partial r/\partial x_j = x_j/(D r)$.

$y_i = \gamma_i x_i / r$. Chain rule:

$\partial y_i/\partial x_j = \gamma_i [\delta_{ij}/r - x_i r^{-2} \partial r/\partial x_j]$

$= (\gamma_i/r)[\delta_{ij} - x_i x_j/(D r^2)]$

즉 Jacobian = $(\gamma_i/r)(I - xx^T/(D r^2))$.

LN의 Jacobian에는 **추가로 $-1/D$ 항** (centering 때문):

$\partial \hat{x}_i^{\text{LN}}/\partial x_j = (1/\sigma)[\delta_{ij} - 1/D - \hat{x}_i \hat{x}_j/D]$

RMSNorm은 **$-1/D$ 항 제거**. $\square$

### 왜 centering이 redundant한가

**경험적 관찰** (Zhang-Sennrich 2019 §3): Machine translation 실험에서 centering을 제거해도 **BLEU score가 거의 동일**. 이론적 설명:

- Residual block $x + F(x)$에서 $F(x)$가 random이면 $\mathbb{E}[x] = 0$.
- 점진 학습으로 $x$가 특정 mean을 가질 수 있지만, downstream의 $\gamma, \beta$가 이를 보정.
- LN의 centering은 **network의 다른 부분이 할 수 있는 일**을 미리 하는 것 — 잉여.

현대 LLM에서 이 관찰이 robust하게 입증됨.

---

## 💻 실험으로 효과 검증

### 실험 1 — RMSNorm PyTorch 구현

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        # x: (..., D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * x / rms

# 동작 확인
x = torch.randn(2, 16)
rms = RMSNorm(16)
y = rms(x)
print("RMS of y:", y.pow(2).mean(-1).sqrt())   # should be ≈ gamma magnitude
```

### 실험 2 — LN vs RMSNorm 속도 벤치마크

```python
import time

D = 4096
x = torch.randn(32, 512, D).cuda()

ln = nn.LayerNorm(D).cuda()
rms = RMSNorm(D).cuda()

# Warmup
for _ in range(10): ln(x); rms(x)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(500): ln(x)
torch.cuda.synchronize()
print(f"LN     : {time.time()-t0:.3f}s")

t0 = time.time()
for _ in range(500): rms(x)
torch.cuda.synchronize()
print(f"RMSNorm: {time.time()-t0:.3f}s")

# → RMSNorm이 20-40% 빠름 (GPU architecture에 따라)
```

### 실험 3 — 훈련 중 residual stream의 centering 확인

```python
# Llama-style Transformer 훈련 중 각 layer의 residual stream x의 mean을 측정
# → 일반적으로 |mean(x)| << std(x) (Zhang-Sennrich 관찰)
# 이는 centering이 잉여라는 가설 지지

# 간략 스케치 (실제 훈련된 모델에서):
def measure_centering(model, data):
    means = []
    hooks = []
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            m = output.mean(-1).abs().mean().item()
            means.append(m)
    for block in model.blocks:
        hooks.append(block.register_forward_hook(hook))
    _ = model(data)
    for h in hooks: h.remove()
    return means

# LLM에서 residual mean이 std보다 100배 작은 경우가 많음
```

### 실험 4 — SwiGLU FFN 구현

```python
class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.W2(torch.nn.functional.silu(self.W(x)) * self.V(x))

# 비교: 기본 GELU FFN
class GELU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

# 파라미터 수: SwiGLU는 두 projection (W, V) + W2 → 기본 FFN보다 1.5배
# 성능: 동일 파라미터 제약이면 SwiGLU가 약간 더 나은 perplexity (Shazeer 2020)
```

### 실험 5 — Llama-style block 전체 조립

```python
class LlamaBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=1376):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, d_ff)
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

# d_ff 선택: SwiGLU는 파라미터 많으므로 기존 GELU FFN (d_ff = 4*d_model)의 ~2/3로 축소
# Llama: d_ff = 2/3 * 4 * d_model ≈ 2.67 * d_model → 여기서는 1376 for d_model=512
```

---

## 🔗 실전 활용

### 현대 LLM의 RMSNorm 채택 현황 (2024 기준)

| Model | Normalization |
|-------|--------------|
| GPT-2, GPT-3 | LayerNorm (Pre-LN) |
| GPT-4 | (비공개, 추정 LN 또는 RMSNorm) |
| Llama 1, 2, 3 | **RMSNorm** |
| Mistral, Mixtral | **RMSNorm** |
| Qwen | **RMSNorm** |
| Gemma | **RMSNorm** |
| Claude (Anthropic) | (비공개) |
| PaLM (Google) | LayerNorm |

RMSNorm이 **2023년 이후 새 모델의 사실상 표준**. Legacy 모델은 LN 유지.

### 구현 tips

- **Epsilon**: RMSNorm의 $\epsilon$은 LN보다 보통 작게 (1e-6 ~ 1e-5). RMS가 std보다 작을 수 있어 numerical stability 덜 critical.
- **mixed precision (bfloat16)**: RMS 계산을 `float32`로 upcast 권장 (Llama 구현 관습).
- **torch.nn.RMSNorm** (PyTorch 2.4+): 내장 구현 (이전엔 custom 필요).

### 다른 최근 트렌드

- **SwiGLU** (Shazeer 2020) → Llama 표준.
- **GeLU** → 초기 Transformer (BERT, GPT-2).
- **GELU** → 정확한 정의는 Gaussian Error Linear Unit.
- **RoPE** (Rotary Position Embedding, Su 2021): absolute/sinusoidal의 개선.
- **MQA / GQA** (Multi-Query / Grouped-Query Attention): inference 효율화.

이 조합이 **Llama recipe**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Residual stream이 centered | 특수 task에서 non-centered일 수 있음 |
| Centering이 잉여 | Small model이나 특이 loss에서는 도움될 수도 |
| Scalar $\gamma$ 표현력 충분 | LayerScale처럼 더 풍부한 affine이 필요한 경우 |
| GPU 환경 최적화 | CPU에서는 속도 차이 미미 |

**주의**: "LN은 이제 구식"이라는 말은 **과장**. 기존 모델의 호환성, 일부 task에서 centering의 잔재적 이점, bf16 numerical stability 등 이유로 LN이 여전히 활발. RMSNorm이 더 최신 default지만 LN이 "틀렸다"는 아님.

---

## 📌 핵심 정리

$$\boxed{\text{RMSNorm}(x) = \gamma \cdot x / \sqrt{\text{mean}(x^2) + \epsilon} \quad (\text{centering 생략})}$$

| 개념 | 의미 |
|------|------|
| **RMSNorm** | LayerNorm에서 $\mu$ 제거, scale only |
| **효율** | ~25% faster, parameter 절반 ($\beta$ 없음) |
| **성능** | LN과 거의 동등, LLM에서 표준화 |
| **Llama stack** | Pre-RMSNorm + SwiGLU + RoPE + AdamW |
| **Ch3 마무리** | Normalization 계보의 현재 종착점 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $x = (1, 3, 5, 7)$의 RMSNorm 출력을 구하라 ($\epsilon = 0$, $\gamma = 1$).

<details>
<summary>힌트 및 해설</summary>

RMS $= \sqrt{(1 + 9 + 25 + 49)/4} = \sqrt{21} \approx 4.583$.

$y = x / \text{RMS} = (1, 3, 5, 7) / 4.583 = (0.218, 0.654, 1.091, 1.527)$.

LN과 비교 ($\mu = 4, \sigma = \sqrt{5}$): $\hat{x}_{\text{LN}} = (-1.34, -0.45, 0.45, 1.34)$.

**차이**: LN은 centered (sum = 0), RMSNorm은 모든 값이 양수(원 부호 유지). Residual stream에서 sign이 의미 있으면 RMSNorm이 이를 보존.

</details>

**문제 2** (심화): RMSNorm이 **shift invariance**를 잃은 것은 실전에서 문제가 되는가? $x + c$ (constant shift)에 대한 RMSNorm과 LN의 반응을 비교하라.

<details>
<summary>힌트 및 해설</summary>

- **LN**: $\text{LN}(x + c) = \text{LN}(x)$ — shift invariant.
- **RMSNorm**: $\text{RMSNorm}(x + c) \neq \text{RMSNorm}(x)$ in general. RMS$(x+c) = \sqrt{\text{mean}((x+c)^2)} = \sqrt{\text{mean}(x^2) + 2c\bar{x} + c^2}$.

**실전 영향**: Residual stream $x$에 외부에서 큰 $c$ shift가 들어오면 RMSNorm의 출력이 크게 변화. 하지만:
1. 실제 Transformer에서 큰 shift가 들어올 이유가 없음 (attention/FFN output은 random mean ≈ 0).
2. Positional encoding은 bounded magnitude → shift 미미.
3. Embedding layer는 학습되므로 shift를 명시적으로 학습하지 않음.

따라서 **실전에서 shift invariance 상실은 거의 영향 없음**. Zhang-Sennrich는 이를 empirically 검증.

예외: 특수 task (pre-quantized activation, unusual loss)에서 centering이 유용할 수 있음.

</details>

**문제 3** (이론-실전): Llama 3의 recipe — Pre-RMSNorm + SwiGLU + RoPE + AdamW. 각 component가 **어느 축**의 regularization/optimization을 담당하는가? 4축 관점(Prior/Ensemble/Landscape/Invariance)으로 분류하라.

<details>
<summary>힌트 및 해설</summary>

| Component | 역할 | 4축 분류 |
|-----------|------|----------|
| Pre-RMSNorm | Gradient scale 안정화, landscape smoothing | **Landscape** |
| SwiGLU | FFN의 표현력 증가 (regularization 아닌) | — |
| RoPE | Positional information (regularization 아닌) | — |
| AdamW (weight decay) | L2 regularization (decoupled) | **Prior** (Gaussian) |
| Warmup + Cosine LR | Optimization trajectory | **Landscape** (implicit) |
| Dropout (attention에 약간) | Ensemble approximation | **Ensemble** |
| 데이터 중복 제거 + Quality filter | Data-level regularization | **Invariance** (distribution) |

**관찰**:
- 현대 LLM recipe는 4축 중 3축(Prior, Landscape, Ensemble)이 중심.
- Invariance는 **data quality와 curation**으로 주로 처리 (vision과 다름 — vision은 augmentation).
- Landscape가 가장 중요 — gradient flow가 LLM 성공의 핵심.

이는 Ch7-04의 "통합 recipe"에서 상세히 다뤄지는 주제를 미리 본 것.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Fixup · SkipInit](./05-fixup-skipinit.md) | [📚 README로 돌아가기](../README.md) | [Chapter 4 → 01. Vicinal Risk Minimization ▶](../ch4-data-augmentation/01-vicinal-risk.md) |

</div>
