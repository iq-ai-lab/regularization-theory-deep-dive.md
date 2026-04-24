# 01. Batch Normalization (Ioffe & Szegedy 2015)

## 🎯 핵심 질문

- Batch Normalization의 **수식**은 정확히 무엇이며 왜 이 형태인가?
- 왜 **$\gamma, \beta$ 같은 affine 복원 파라미터**가 필수인가?
- **Train mode**와 **Eval mode**의 계산이 어떻게 다른가? Running mean/variance의 역할은?
- Forward와 backward pass의 chain rule은 어떻게 전개되는가?

---

## 🔍 왜 BN이 등장했는가 (Ioffe의 원 주장)

2015년까지 깊은 네트워크의 훈련 난점은 **internal covariate shift (ICS)**:
- Layer $\ell$의 input 분포가 훈련 중 계속 변함 (아래 layer의 weight 변화 때문).
- Layer $\ell$은 "움직이는 target"에 대해 훈련해야 함 → lr를 작게 쓰거나 careful initialization 필요.

Ioffe & Szegedy의 **해결책**: 각 layer의 input을 **정규화**하여 분포를 안정화. BN은 그 후 표준 도구가 되었으며, ResNet (He 2016), Inception (Szegedy 2015) 등 모든 주요 CNN에 탑재.

**그러나** — Ch3-02에서 보겠지만 **Santurkar 2018이 ICS 설명을 실험으로 반박**. BN의 실제 효과는 **loss landscape smoothing**. 이 문서는 먼저 BN의 **수식과 구현** 을 엄밀화하고, 다음 문서에서 신화 해체로 넘어간다.

---

## 📐 수학적 선행 조건

- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): forward/backward pass, chain rule
- 통계: sample mean $\mu_B$, sample variance $\sigma_B^2$, z-score 정규화
- 미분: $\partial/\partial x (x / \sqrt{\sigma^2 + \epsilon})$의 전개

---

## 📖 직관적 이해

### 기본 아이디어

각 mini-batch 내에서 feature별로 **zero mean, unit variance**로 정규화, 그 후 학습 가능한 affine $(\gamma, \beta)$로 복원:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

**왜 다시 affine?** 단순 정규화는 표현력을 제한한다 — activation의 특정 분포가 최적일 수도 있다. $\gamma, \beta$로 "적절한 분포"를 학습할 수 있게 한다.

### Train vs Eval 모드

- **Train**: 각 mini-batch의 $\mu_B, \sigma_B^2$을 계산해 정규화.
- **Eval**: 훈련 중 수집한 **running statistics** $\hat{\mu}, \hat{\sigma}^2$을 사용 (EMA).

**왜 분리?** Inference 시에는 batch 없을 수도, 혹은 batch size가 1일 수도. Single-sample inference가 일관되게 작동하려면 population-level 추정치 필요.

### Affine param의 자유도

$(\gamma, \beta)$가 없으면 BN은 "정규화된 값만"을 다음 layer로 보냄 — 모든 BN output이 zero mean, unit variance. 이는 signal의 magnitude/shift 정보를 강제로 잃는다.

- $\gamma$: 원하는 scale 학습 (predicted std).
- $\beta$: 원하는 shift 학습 (predicted mean).

극단: BN이 identity가 되려면 $\gamma = \sigma_B, \beta = \mu_B$ — 원래 분포로 완전 복원.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — BatchNorm Operation (Ioffe & Szegedy 2015, Alg. 1)

Mini-batch $B = \{x_1, \ldots, x_m\}$ ($x_i \in \mathbb{R}$, 한 feature dimension 기준), hyperparameter $\epsilon > 0$:

$$\begin{aligned}
\mu_B &= \frac{1}{m}\sum_{i=1}^m x_i \\
\sigma_B^2 &= \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2 \\
\hat{x}_i &= \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
y_i &= \gamma \hat{x}_i + \beta = \text{BN}_{\gamma, \beta}(x_i)
\end{aligned}$$

Multi-feature ($x_i \in \mathbb{R}^d$)에서는 **각 feature마다 독립**으로 적용 → $\gamma, \beta \in \mathbb{R}^d$.

### 정의 1.2 — 4D Conv Input의 BN

Conv feature map $x \in \mathbb{R}^{B \times C \times H \times W}$. **각 채널 $c$**에 대해:
- Statistics를 $(B, H, W)$ 전체에서 계산 (즉 $m = B \cdot H \cdot W$).
- $\gamma, \beta \in \mathbb{R}^C$ — 채널별로 하나씩.

### 정의 1.3 — Running Statistics

Momentum $\alpha \in (0, 1)$ (PyTorch default: $\alpha = 0.1$):

$$\hat{\mu}_{\text{run}} \leftarrow (1 - \alpha) \hat{\mu}_{\text{run}} + \alpha \mu_B$$
$$\hat{\sigma}^2_{\text{run}} \leftarrow (1 - \alpha) \hat{\sigma}^2_{\text{run}} + \alpha \sigma_B^2$$

훈련 중 모든 batch에 걸쳐 EMA로 수집. Eval 시 이것을 $\mu_B, \sigma_B^2$ 대신 사용.

### 정리 1.4 — Train/Eval 모드의 수식

$$y^{\text{train}}_i = \gamma \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta, \quad y^{\text{eval}}_i = \gamma \frac{x_i - \hat{\mu}_{\text{run}}}{\sqrt{\hat{\sigma}^2_{\text{run}} + \epsilon}} + \beta$$

**주의**: 두 모드는 **완전히 다른 함수**다. 같은 $x_i$를 넣어도 다른 출력. Batch size 1 inference에서는 eval mode가 필수 (train mode라면 $\sigma_B = 0$로 divide by 0).

### 정리 1.5 — Forward Pass의 Chain Rule (Backward)

각 $\hat{x}_i, \mu_B, \sigma_B^2$에 대한 $\partial L / \partial x_i$:

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}$$

그리고:
$$\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i, \quad \frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$

---

## 🔬 수학적 유도

### Backward Pass 완전 유도

**Given**: $\partial L / \partial y_i$ (upstream gradient).

$y_i = \gamma \hat{x}_i + \beta$이므로:

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma, \quad \frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i, \quad \frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$

$\hat{x}_i = (x_i - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$의 변수 의존성:
- $x_i$: 명시적.
- $\mu_B$: 모든 $x_j$에 의존.
- $\sigma_B^2$: 모든 $x_j$에 의존.

$$\frac{\partial L}{\partial \sigma_B^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot \left(-\frac{1}{2}\right)(\sigma_B^2 + \epsilon)^{-3/2}$$

$$\frac{\partial L}{\partial \mu_B} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{-2 \sum_i (x_i - \mu_B)}{m}$$

마지막 항은 $-2\sum_i(x_i - \mu_B)/m = 0$ (mean 정의). 따라서:

$$\frac{\partial L}{\partial \mu_B} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}}$$

종합:

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m} \quad \square$$

### 정규화와 gradient rescaling의 효과

$\hat{x} = (x - \mu_B)/\sigma$에서 scale invariance: $x \to a x$로 바꿔도 $\mu_B \to a\mu_B$, $\sigma_B \to a\sigma_B$, $\hat{x}$는 **불변**. 따라서 **BN이 있는 layer는 input scale에 rescaling invariant**.

이 성질이 learning rate의 robustness를 제공 — weight update가 layer activation scale에 덜 민감.

---

## 💻 실험으로 효과 검증

### 실험 1 — PyTorch BatchNorm의 구조 확인

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=4)
# trainable parameters: gamma, beta
print("gamma (weight):", bn.weight.data)   # 초기값 1.0
print("beta  (bias) :", bn.bias.data)      # 초기값 0.0
# buffer: running_mean, running_var
print("running_mean :", bn.running_mean)
print("running_var  :", bn.running_var)
print("num_batches_tracked:", bn.num_batches_tracked)
```

### 실험 2 — 수동 구현 vs PyTorch

```python
def manual_bn(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta

torch.manual_seed(0)
x = torch.randn(32, 4)
bn = nn.BatchNorm1d(4); bn.train()
with torch.no_grad():
    y_torch = bn(x)
y_manual = manual_bn(x, bn.weight, bn.bias)

print("max |diff|:", (y_torch - y_manual).abs().max().item())
# → 거의 0 (수치 오차 수준)
```

### 실험 3 — Train vs Eval mode 출력 차이

```python
bn = nn.BatchNorm1d(4)
x = torch.randn(10, 4)

# 훈련 (running stats 수집)
bn.train()
for _ in range(50):
    _ = bn(torch.randn(10, 4))

# 같은 input을 train/eval로 돌리면 다른 출력
bn.train()
y_train = bn(x)
bn.eval()
y_eval = bn(x)
print("train output [0]:", y_train[0])
print("eval  output [0]:", y_eval[0])
print("max |diff|:", (y_train - y_eval).abs().max().item())
# → 꽤 큰 차이 — 같은 input 서로 다른 출력
```

### 실험 4 — Scale Invariance 검증

```python
bn = nn.BatchNorm1d(4); bn.train()
x = torch.randn(16, 4)
x_scaled = x * 100.0       # scale 100배

y = bn(x)
# 같은 BN 모듈 인스턴스는 running stats를 공유하므로 새 인스턴스로
bn2 = nn.BatchNorm1d(4); bn2.train()
y_scaled = bn2(x_scaled)

print("max |y - y_scaled| :", (y - y_scaled).abs().max().item())
# → 매우 작음 (scale invariance — 단 초기 gamma=1 덕분)
```

### 실험 5 — Ch3-02 예고: BN 있는/없는 네트워크의 loss landscape 거칠기

```python
class SimpleNet(nn.Module):
    def __init__(self, use_bn=True):
        super().__init__()
        layers = []
        in_d = 100
        for h in [64, 64, 64, 64]:
            layers.append(nn.Linear(in_d, h))
            if use_bn: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            in_d = h
        layers.append(nn.Linear(64, 10))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# 같은 지점에서 gradient Lipschitz 추정
# (실제 Santurkar 2018 측정은 Ch3-02에서 상세화)
```

---

## 🔗 실전 활용

### 언제 BN을 쓰는가

- **CNN on ImageNet/CIFAR**: 거의 필수. ResNet, EfficientNet 등 모두 탑재.
- **Large batch size (≥32)**: BN의 mini-batch 통계가 population 근사로 유효.
- **Transfer learning**: ImageNet pre-trained BN의 running stats를 그대로 재사용.

### 언제 BN을 피하는가

- **Small batch (≤8)**: batch statistics 불안정 → **Group Norm** (Ch3-04) 권장.
- **RNN/sequence models**: time step 간 batch 구조가 복잡 → **Layer Norm** (Ch3-03).
- **GAN 훈련**: generator의 분포가 불안정 → Layer/Instance Norm 선호.
- **Inference batch size 1**: running stats 사용 가능하지만 domain shift 위험.

### 흔한 실수

1. **Train mode로 inference**: batch size 1에서 $\sigma_B = 0$ → NaN 출력.
2. **Pretrained BN fine-tuning**: 새 domain에서 running stats가 안 맞음 → `model.eval()` 해야 할 수도.
3. **BN + Dropout 순서**: BN 다음 Dropout이 permissible하지만, Dropout → BN은 dropout noise가 BN stats에 편입되어 왜곡. "Disharmony" 문제 (Li 2019).

### Affine param의 흥미로운 활용

**Feature-wise Linear Modulation (FiLM)** (Perez 2018): $\gamma, \beta$를 conditioning input으로 동적 생성 → conditional normalization. StyleGAN의 AdaIN도 이 원리.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Batch statistics가 population 근사 | 작은 batch에서는 노이즈 많음 |
| Training/Eval 일관성 | 두 모드 mismatch가 domain shift처럼 작용 가능 |
| Feature 독립적 정규화 | Feature 간 correlation 있으면 정확한 decorrelation 필요 (다른 기법) |
| 기울기 수치 안정 | $\epsilon$ 너무 작으면 NaN, 너무 크면 정규화 효과 감소 |
| Channel별 $\gamma, \beta$ | Instance/Group Norm과 다른 affine 구조 |

**중요**: BN의 "ICS 완화" 주장은 Ch3-02에서 논파. **수식 자체는 정확하지만 설명은 수정 필요**.

---

## 📌 핵심 정리

$$\boxed{\text{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta}$$

| 개념 | 의미 |
|------|------|
| **$\mu_B, \sigma_B^2$** | 현 mini-batch의 feature별 mean/variance |
| **$\gamma, \beta$** | 학습 가능한 scale/shift — 표현력 복원 |
| **Running stats** | Eval 모드용 population 추정, EMA로 수집 |
| **Train/Eval 분리** | 두 모드 서로 다른 함수 — 핵심 구분 |
| **다음 질문** | ICS 주장은 왜 틀렸는가? → Ch3-02 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Batch size 4, feature 3인 input $x$가 다음과 같을 때 BN의 $\hat{x}$를 계산하라 ($\epsilon = 0$ 무시, $\gamma = 1, \beta = 0$).

$x = \begin{bmatrix} 1 & 0 & 2 \\ 3 & 2 & 4 \\ 5 & 4 & 6 \\ 7 & 6 & 8 \end{bmatrix}$

<details>
<summary>힌트 및 해설</summary>

Feature별 (column별) 통계:
- Col 0: mean=4, var= ((1-4)²+(3-4)²+(5-4)²+(7-4)²)/4 = (9+1+1+9)/4 = 5, std=√5 ≈ 2.236
- Col 1: mean=3, var=5, std=√5
- Col 2: mean=5, var=5, std=√5

$\hat{x}$ = $(x - \mu) / \sigma$:

$\hat{x}_{0,0} = (1-4)/\sqrt{5} \approx -1.34$  
$\hat{x}_{1,0} = (3-4)/\sqrt{5} \approx -0.45$  
$\hat{x}_{2,0} = (5-4)/\sqrt{5} \approx 0.45$  
$\hat{x}_{3,0} = (7-4)/\sqrt{5} \approx 1.34$

(다른 column은 대칭으로 같은 값.)

확인: 각 column의 평균 = 0, 분산 = 1 — BN의 보장.

</details>

**문제 2** (심화): $\gamma, \beta$ 없이 BN을 쓰면 무엇을 잃는가? Identity function이 되는 조건을 살펴라.

<details>
<summary>힌트 및 해설</summary>

$\gamma, \beta$ 없으면 $y = \hat{x}$. 이는 **정규화된 분포**만 다음 layer로 전달.

**잃는 표현력**:
1. Feature의 **natural scale**: 어떤 feature는 큰 값이 유용할 수 있음 (e.g. 극단값 detection).
2. Feature의 **mean shift**: ReLU 전에 $\text{bias}$가 필요할 수 있음.
3. **Identity function**: $\hat{x} = \gamma \cdot x + \beta$로 복원 불가 — BN은 identity를 표현할 수 없음 (scale 정보 상실).

**$\gamma = \sigma_B, \beta = \mu_B$로 Identity 복원**: 이론적으로 가능하지만 $\sigma_B, \mu_B$는 batch-dependent이므로 static $\gamma, \beta$로는 exact identity 불가. 대신 network가 **데이터 분포에 맞게 optimal $\gamma, \beta$를 학습**.

Ioffe-Szegedy의 표현: "If $\gamma = \sqrt{\text{Var}(x)}$ and $\beta = \mathbb{E}[x]$, then $y = x$" — 하지만 이는 **population** variance, running stats는 추정치. $\gamma, \beta$를 학습하게 두면 네트워크가 "가장 유용한 분포"를 고른다.

</details>

**문제 3** (이론-실전): PyTorch의 `model.eval()`을 잊으면 batch size 1 inference에서 어떤 문제가 생기는가?

<details>
<summary>힌트 및 해설</summary>

Batch size 1이면 mini-batch의 $\sigma_B^2 = 0$. 정규화 $(x - \mu_B)/\sqrt{0 + \epsilon}$에서 $\sqrt{\epsilon}$으로 나누는 꼴:

- 극단적으로 작은 denominator → **output이 엄청 크거나 NaN**.
- $\epsilon$이 작으면 (1e-5) 결과가 $\sim 10^{2.5}$ 배 확대.

또한 running stats가 inference input에 의해 **업데이트** 되어 (train mode에서) 오염될 수 있음 (단 `torch.no_grad()` 써도 stats 업데이트는 일어남). 특히 test set의 분포가 train과 다르면 running stats가 천천히 오염.

**올바른 관습**:
- Inference: `model.eval()` + `torch.no_grad()` 필수.
- 배포 전 `torch.save(model.state_dict(), ...)`로 running stats 고정.
- ONNX export 전에 `eval()` 모드.

Docker container 배포 시 `model.eval()`을 프로덕션 코드에 명시.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Chapter 2 → 05. Stochastic Depth](../ch2-dropout/05-dropout-dropconnect-stochdepth.md) | [📚 README로 돌아가기](../README.md) | [02. Santurkar 2018의 BN 신화 반박 ▶](./02-santurkar-refutation.md) |

</div>
