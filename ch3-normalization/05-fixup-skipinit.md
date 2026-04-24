# 05. BN 없이 깊은 네트워크 — Fixup, SkipInit

## 🎯 핵심 질문

- **Fixup 초기화** (Zhang et al. 2019)는 ResNet의 어떤 scale을 조정하여 BN 없이 훈련을 가능하게 하는가?
- $L^{-1/(2m-2)}$라는 초기화 공식은 어디서 나오는가?
- **SkipInit** (De & Smith 2020)의 "**learnable $\alpha_l = 0$**" 초기화는 어떤 의도인가?
- **NFNet** (Brock 2021)의 AGC(adaptive gradient clipping)이 BN의 마지막 역할을 대체한 이유는?

---

## 🔍 왜 "BN 없는 ResNet"인가

BN의 세 가지 비용:

1. **계산 overhead**: ~10-20% training time 증가.
2. **Memory**: Running stats + per-batch 통계 저장.
3. **Distributed training**: batch stats 동기화 overhead (SyncBN).
4. **Small batch 문제**: Ch3-04에서 본 부작용.

**Ideal**: ResNet의 성능을 유지하면서 BN 제거. 이 문서는 두 가지 접근을 본다.

- **Fixup**: 초기화만으로 gradient flow 안정화.
- **SkipInit**: residual branch를 0으로 시작해 점진적 학습.
- **NFNet**: 위 + Scaled Weight Standardization + AGC.

이 흐름은 "BN이 왜 효과적이었는가"를 **역설계**한다: Santurkar의 landscape smoothing을 초기화·가중치 제약으로 대체.

---

## 📐 수학적 선행 조건

- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): He initialization, Xavier initialization, residual connection
- Ch3-01, Ch3-02: BN의 정규화 효과
- 확률: forward activation variance 전파 공식

---

## 📖 직관적 이해

### ResNet 깊어지면 왜 BN이 필요한가

$L$-layer ResNet $h_L = h_0 + \sum_{\ell=1}^L F_\ell(h_{\ell-1})$. 초기화 후 각 $F_\ell(h)$가 unit-variance 정도이면 $h_L$의 variance는 **$L$배 증가**. 깊은 네트워크에서 activation/gradient blow-up.

BN은 매 layer마다 variance를 1로 리셋 → explosion 방지.

### Fixup의 아이디어

"**Residual branch의 scale을 처음부터 작게 만들자**". 각 block의 마지막 conv의 weight를 $L^{-1/(2m-2)}$로 scale ($m$은 block의 conv 개수, 보통 2).

**효과**: 초기 forward pass에서 $F_\ell(h) \approx 0$. 전체 output $h_L \approx h_0$. Gradient도 bounded. 훈련 중 $F_\ell$이 점진적으로 의미 있는 크기로 학습.

### SkipInit의 단순함

Fixup보다 단순: 각 residual block에 **learnable scalar $\alpha_l$**을 추가하고 **$\alpha_l = 0$으로 초기화**.

$$h_{\ell+1} = h_\ell + \alpha_l F_\ell(h_\ell)$$

초기에는 identity network와 같음. $\alpha_l$이 gradient descent로 점진 증가 → "block을 언제 활성화할지" 자체를 학습.

### 두 기법의 공통 아이디어

둘 다 **"residual branch를 초기에 거의 0"** 으로 만들어 gradient가 shortcut만 통과하게 함. 깊이에 따른 variance 폭발 방지.

이 관점에서 BN의 효과 중 "activation variance 조절"은 **초기화로 대체 가능**하다. 나머지 (실시간 landscape smoothing)은 NFNet의 AGC로 해결.

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — ResNet Residual Block

$m$-layer block (conv/BN/ReLU/conv):

$$F_\ell(h) = W_\ell^{(m)} \sigma \cdots \sigma(W_\ell^{(1)} h)$$

표준 ResNet block은 $m = 2$. Output: $h_{\ell+1} = h_\ell + F_\ell(h_\ell)$.

### 정의 5.2 — Fixup Initialization (Zhang, Dauphin, Ma 2019)

$L$-block ResNet에서 각 $F_\ell$의 **마지막** convolution을 **$L^{-1/(2m-2)}$로 scale**:

$$W_\ell^{(m), \text{Fixup}} = W_\ell^{(m), \text{He}} \cdot L^{-1/(2m-2)}$$

$m = 2$ (표준 block): $L^{-1/2}$로 scale.

또한:
- Block 내부 intermediate layer: He initialization.
- 마지막 output classifier layer: 0으로 초기화.
- Skip connection의 learnable bias, scaling 추가 ($b_l, g_l$).

### 정의 5.3 — SkipInit (De & Smith 2020)

각 residual block에 learnable scalar $\alpha_l \in \mathbb{R}$:

$$h_{\ell+1} = h_\ell + \alpha_l F_\ell(h_\ell)$$

초기화: $\alpha_l = 0 \ \forall \ell$. Block 내부 weight는 normal He initialization.

### 정리 5.4 — Fixup의 Forward Variance 안정성

$L$-layer Fixup ResNet에서 $h_0$이 unit variance이면:

$$\text{Var}(h_L) = (1 + O(1/L)) \cdot \text{Var}(h_0)$$

즉 variance가 $L \to \infty$에서도 **bounded** (정확히 $O(1)$). BN 없이 이를 달성.

### 정리 5.5 — SkipInit의 Initial Identity Property

초기화 $\alpha_l = 0$이므로:

$$h_L = h_0 + \sum_\ell 0 \cdot F_\ell(\cdot) = h_0$$

**초기 네트워크 = identity**. Training 초기에 gradient가 deep layer까지 **그대로** 전달 (no vanishing/exploding).

### 정리 5.6 — NFNet의 Scaled Weight Standardization

Brock et al. 2021는 **SWS**: weight를 매 forward에서 표준화:

$$\hat{W}_{ij} = \gamma \frac{W_{ij} - \mu_W}{\sigma_W \sqrt{\text{fan-in}}}$$

$\gamma$ scalar learnable. BN처럼 activation을 만지지 않고 **weight를 매 forward에서 재조정**. 이는 WeightNorm (Ch3-04)의 발전형.

**AGC** (Adaptive Gradient Clipping): gradient norm이 weight norm의 일정 비율을 넘으면 clip. Scale invariance를 gradient level에서 보장.

---

## 🔬 수학적 유도

### 정리 5.4 증명 스케치

$h_{\ell+1} = h_\ell + F_\ell(h_\ell)$, $h_\ell$의 variance $v_\ell$.

$F_\ell$의 output variance는 block 내 weight의 multiplicative effect. 표준 He init에서 $\text{Var}(F_\ell(h)) = v_\ell \cdot c$ for some constant $c = O(1)$.

Fixup scaling $L^{-1/(2m-2)}$: 마지막 conv의 weight가 $L^{-1/(2m-2)}$배 → output variance가 $L^{-1/(m-1)}$배로 scale.

$m = 2$ (표준): $\text{Var}(F_\ell) = O(v_\ell / L)$.

$v_{\ell+1} = v_\ell + O(v_\ell / L) = v_\ell (1 + O(1/L))$.

누적: $v_L = v_0 \prod_\ell (1 + O(1/L)) \leq v_0 \cdot e^{\sum O(1/L)} = v_0 \cdot e^{O(1)}$ — **bounded**. $\square$

### Fixup vs SkipInit의 차이

Fixup은 deterministic initialization scaling. SkipInit은 learnable — network가 스스로 $\alpha_l$ 조정.

경험적으로:
- Fixup: 더 정밀한 초기화, small architectures에서 최적.
- SkipInit: 훈련 과정 자체가 adaptive, deeper/wider 네트워크에 유연.

---

## 💻 실험으로 효과 검증

### 실험 1 — Fixup ResNet 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixupBlock(nn.Module):
    def __init__(self, in_c, out_c, L, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, padding=1, bias=False)
        # Fixup scaling on last conv (m=2): L^(-1/2)
        with torch.no_grad():
            self.conv2.weight.mul_(L ** -0.5)
        # Fixup bias and scale
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.skip = nn.Conv2d(in_c, out_c, 1, stride, bias=False) if in_c != out_c else nn.Identity()
    def forward(self, x):
        out = self.conv1(x + self.bias1a)
        out = F.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        return F.relu(out + self.skip(x))

# ResNet-100 Fixup (L=100 blocks)
class FixupResNet(nn.Module):
    def __init__(self, L=100, c=64, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.Sequential(*[FixupBlock(c, c, L) for _ in range(L)])
        self.classifier = nn.Linear(c, num_classes)
        nn.init.zeros_(self.classifier.weight)   # classifier zero init
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
```

### 실험 2 — SkipInit 구현과 비교

```python
class SkipInitBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, padding=1)
        self.alpha = nn.Parameter(torch.zeros(1))   # 0으로 초기화
        self.skip = nn.Conv2d(in_c, out_c, 1, stride) if in_c != out_c else nn.Identity()
    def forward(self, x):
        residual = F.relu(self.conv1(x))
        residual = self.conv2(residual)
        return self.skip(x) + self.alpha * residual

# 훈련 전후 alpha 추적
net = nn.Sequential(*[SkipInitBlock(64, 64) for _ in range(30)])
print("초기 alpha 값:", [b.alpha.item() for b in net[:5]])
# → 모두 0 (identity network)

# 훈련 후 alpha가 점진적으로 증가하는 양상 확인
```

### 실험 3 — Activation variance 추적 (Fixup vs BN vs nothing)

```python
class PlainResNet(nn.Module):
    """BN, Fixup 없음 — 비교용"""
    def __init__(self, L=50, c=64):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(L):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1)))
    def forward(self, x):
        x = self.stem(x)
        variances = [x.var().item()]
        for block in self.blocks:
            x = x + block(x)   # residual
            variances.append(x.var().item())
        return variances

# 같은 입력 분포에서 변화 추적
import matplotlib.pyplot as plt
x = torch.randn(4, 3, 32, 32)

plain = PlainResNet(L=50)
vars_plain = plain(x)
fixup = FixupResNet(L=50, num_classes=10)
# activation var 측정은 hook 필요 — 여기서는 스케치

plt.figure(figsize=(9, 4))
plt.plot(vars_plain, label='Plain (no BN/Fixup)')
plt.yscale('log')
plt.xlabel('layer'); plt.ylabel('activation variance')
plt.title('Deep residual network의 activation variance')
plt.legend(); plt.grid(alpha=0.3); plt.show()
# → Plain은 기하급수 증가, Fixup은 거의 일정
```

### 실험 4 — Large learning rate에서의 수렴 비교

```python
# 같은 CIFAR-10 task, lr = 1e-3
# Plain ResNet → 발산
# BN ResNet → 수렴
# Fixup ResNet → 수렴 (BN 없이!)
# SkipInit ResNet → 수렴, 초기 몇 epoch은 매우 느림 (alpha 학습)
```

---

## 🔗 실전 활용

### Fixup의 실전 사용

- CIFAR-10 ResNet 100+ layer까지 BN 없이 훈련 성공 (Zhang 2019).
- ImageNet ResNet-50에서 BN 거의 동등한 성능.
- **장점**: BN의 메모리/계산 overhead 제거.
- **단점**: 초기화 공식이 architecture-specific, 수정 시 재유도 필요.

### SkipInit의 실전 사용

- DeepMind의 "Normalizer-Free ResNets" (Brock 2021) 기초.
- Transformer의 "LayerScale" (CaiT, 2021)도 유사 철학: $\alpha_l \cdot F_\ell$로 residual branch 학습.
- **Llama 등 LLM**에서 유사한 "scaling residual" 트릭 사용.

### NFNet의 전체 recipe

1. **Scaled Weight Standardization**: BN 없이 weight로 normalization 효과.
2. **Adaptive Gradient Clipping**: gradient/weight ratio 제한.
3. **Large batch + Mixup + RandAugment** (Ch4).
4. **AdamW with cosine decay**.

결과: ImageNet top-1 86%+ with NFNet-F6 (EfficientNet-L2 수준, BN 없이).

### 언제 BN 대신 Fixup/SkipInit?

- Small compute budget에서 overhead 절감.
- 깊은 Transformer에 적용 (LayerScale).
- Generative model (BN이 해로운 경우).

### 언제 여전히 BN?

- 표준 CNN pipeline (pre-trained model 활용).
- Image classification on ImageNet with large batch.
- 새 architecture 개발 — BN이 "safe default".

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Fixup 초기화 formula (L^(-1/(2m-2))) | Block 구조 변경 시 재유도 필요 |
| SkipInit alpha = 0 | 초기 수렴 매우 느림 (alpha 학습 시간) |
| NFNet AGC의 hyperparameter | 정밀 튜닝 필요 (clip ratio ~0.01) |
| ResNet 구조 전제 | Non-residual network에는 직접 적용 어려움 |
| Image task 중심 | NLP에서는 LN이 여전히 표준 |

**주의**: "BN 완전 대체"는 아직 **작은 규모 pre-trained에서만** 확실. 초대형 모델(GPT-4 규모) BN 없이 훈련은 BN/LN/RMSNorm 중 **LN/RMSNorm이 여전히 필수**.

---

## 📌 핵심 정리

$$\boxed{\text{BN 없이 깊은 NN: Fixup (init) / SkipInit ($\alpha_l = 0$) / NFNet (SWS + AGC)}}$$

| 기법 | 접근 방식 |
|------|----------|
| **Fixup** | Residual branch 마지막 conv를 $L^{-1/(2m-2)}$로 scale |
| **SkipInit** | Learnable $\alpha_l$을 0으로 초기화, 점진 증가 |
| **NFNet** | Scaled Weight Standardization + AGC로 BN 완전 대체 |
| **통일 원리** | Landscape smoothing을 초기화·weight·gradient로 재구성 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $L = 50$, $m = 2$ 표준 ResNet block에서 Fixup scaling factor는? $L = 200$이면?

<details>
<summary>힌트 및 해설</summary>

$L^{-1/(2m-2)} = L^{-1/2}$ (for $m = 2$).

- $L = 50$: $50^{-0.5} \approx 0.1414$.
- $L = 200$: $200^{-0.5} \approx 0.0707$.

더 깊은 네트워크에서 **더 작게** scale. 즉 초기에 residual branch output이 **깊이에 비례해 더 작음**. 이것이 variance explosion 방지의 정량적 이유.

비교: BN은 모든 layer에서 variance를 1로 리셋 → 깊이 무관 scaling. Fixup은 "**초기화에서 지불하고 훈련 중 학습**" 접근.

</details>

**문제 2** (심화): SkipInit의 $\alpha_l$이 훈련 중 어떻게 변화하는가? Layer별로 **얼마나 "활성화"**되는지 실험적으로 확인하는 방법을 설명하라.

<details>
<summary>힌트 및 해설</summary>

훈련 중 각 `alpha_l.item()`을 log. 일반적 패턴:

- **Early layers** (입력 근처): $\alpha_l$ 빠르게 증가 → block이 일찍 활성화.
- **Deep layers** (출력 근처): $\alpha_l$ 천천히 증가 → 훈련 후반에야 유용.
- **Final $\alpha_l$ 분포**: 모든 block이 유의미하게 활성 ($\alpha_l \sim 0.1-1.0$).

```python
alpha_history = {i: [] for i in range(len(blocks))}
for epoch in ...:
    for batch in ...:
        train_step()
    for i, block in enumerate(blocks):
        alpha_history[i].append(block.alpha.item())

# Plot 각 layer의 alpha trajectory
for i, trace in alpha_history.items():
    plt.plot(trace, alpha=0.3, label=f'layer {i}' if i%10==0 else None)
```

"Learned depth selection"을 이 curve가 보여줌 — 깊은 네트워크가 실제로 모든 block을 활용하는지, 일부만인지 진단 가능.

DeepMind 2020은 이를 바탕으로 "Normalizer-Free" 모델 설계: 학습된 $\alpha_l$ 작은 block은 제거 가능.

</details>

**문제 3** (이론-실전): Transformer의 **LayerScale** (CaiT 2021)는 SkipInit의 아이디어를 Transformer에 적용. 수식 차이와 실전 효과를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**LayerScale**: Transformer block에서 attention과 FFN의 output에 learnable **diagonal** matrix $\text{diag}(\lambda)$ 적용:

$$h_{\ell+1} = h_\ell + \text{diag}(\lambda_\ell^{\text{attn}}) \cdot \text{Attn}(\text{LN}(h_\ell))$$
$$h_{\ell+1} \leftarrow h_{\ell+1} + \text{diag}(\lambda_\ell^{\text{ffn}}) \cdot \text{FFN}(\text{LN}(h_{\ell+1}))$$

$\lambda \in \mathbb{R}^D$ per-dimension scale, $\lambda_0 = \epsilon$ (매우 작은 값, e.g. 1e-5, 1e-4).

**SkipInit과의 차이**:
- SkipInit: scalar $\alpha_l$, 0으로 초기화.
- LayerScale: per-channel $\lambda_l$, small $\epsilon$으로 초기화 (gradient flow 유지).

**실전 효과** (CaiT Table 1):
- ViT-Large 깊이 36 layers: LayerScale로 ImageNet top-1 +1.4%.
- Deeper (48+) 모델에서도 안정 훈련.
- LLM (Llama-like)에서도 점진 확산.

**통합 관점**: SkipInit, LayerScale, Fixup 모두 "residual branch를 작게 시작" 원리의 다른 구현. Scale 단위(scalar vs per-channel)와 초기값(0 vs small $\epsilon$)이 다를 뿐.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. GN · IN · WN](./04-gn-in-wn.md) | [📚 README로 돌아가기](../README.md) | [06. RMSNorm · 현대 Transformer ▶](./06-rmsnorm-modern.md) |

</div>
