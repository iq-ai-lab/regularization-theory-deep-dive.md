# 03. Confidence Penalty와 Maximum Entropy

## 🎯 핵심 질문

- **Confidence Penalty** (Pereyra 2017): 목적함수에 $-\beta H(p(y|x))$ 추가가 주는 효과는?
- **Maximum Entropy** 원리 (Jaynes 1957)와 어떤 관계인가?
- Label Smoothing과 **수학적으로 동등**한 경우는?
- 언제 CP를 쓰고 언제 LS를 쓰는가?

---

## 🔍 왜 또 다른 calibration 기법?

Ch5-01·02에서 LS와 KD를 봤다. 둘 다 "soft target"으로 over-confidence를 방지. **Confidence Penalty**는 같은 목표를 다른 접근으로:

- **LS**: target 분포를 soft하게 ($\tilde{y} = (1-\alpha) y + \alpha/K$).
- **CP**: **prediction 분포의 entropy를 증가**시키는 penalty.

수학적으로 두 접근은 **거의** 같다 (정리 3.3). 그러나 **의도적 차이**가 있다:
- LS는 "target"을 다룸 → training data에 대한 가정.
- CP는 "output"을 다룸 → model behavior에 대한 제약.

이 문서는 CP의 수학, Maximum Entropy 원리와의 연결, LS와의 관계를 정리한다.

---

## 📐 수학적 선행 조건

- Ch5-01, Ch5-02: LS의 KL to uniform 해석
- 정보이론: Shannon entropy $H(p) = -\sum p_k \log p_k$, maximum entropy 원리
- Jaynes 1957의 MaxEnt principle

---

## 📖 직관적 이해

### Confidence Penalty

표준 cross-entropy에 **output entropy**를 더해주기:

$$L_{\text{CP}} = L_{\text{CE}}(p, y) - \beta \cdot H(p)$$

$H(p) = -\sum p_k \log p_k$는 predictive distribution의 entropy.

- $\beta > 0$: $-\beta H(p)$가 negative → entropy 증가 선호 → distribution이 uniform에 가깝게.
- $\beta$ 크면 강한 regularization.

### Maximum Entropy 원리 (Jaynes 1957)

"주어진 제약 조건 하에서 **최대 entropy** 분포를 선택하라". 제약이 없으면 uniform, 제약이 많으면 더 concentrated.

**정보이론적 직관**: Entropy가 높을수록 "정보가 적다" = "편향이 적다" = "consensus".

**CP의 해석**: 
- $L_{\text{CE}}$ 부분: label 정보 제약 ($p_c$가 커야 함).
- $-\beta H(p)$ 부분: entropy 최대화 선호.

이 둘의 균형이 "calibrated posterior estimate" 유도.

### CP와 LS의 관계

$L_{\text{CP}} = L_{\text{CE}} - \beta H(p) = -\sum y_k \log p_k + \beta \sum p_k \log p_k$

$L_{\text{LS}} = -\sum [(1-\alpha) y_k + \alpha/K] \log p_k$ (Ch5-01 정리 1.4 equivalent form).

두 loss는 gradient가 약간 다르지만 (one targets $p$, other targets $\log p$), **behavior는 유사**.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Confidence Penalty (Pereyra et al. 2017)

$$L_{\text{CP}}(p, y) = L_{\text{CE}}(p, y) - \beta \cdot H(p) = -\log p_c + \beta \sum_k p_k \log p_k$$

$\beta > 0$ hyperparameter.

### 정의 3.2 — Maximum Entropy Distribution

제약 $\mathbb{E}[f_i(X)] = c_i$ ($i = 1, \ldots, m$) 하에서:

$$p^* = \arg\max_{p: \text{constraints}} H(p)$$

Exponential family form으로 나타남 (Lagrange multiplier 사용).

**특수 경우**:
- 제약 없음: $p^* = $ uniform.
- Mean constraint only: $p^* = $ Gaussian (continuous) or Poisson (discrete positive).

### 정리 3.3 — CP와 LS의 Gradient 관계

CP gradient:

$$\frac{\partial L_{\text{CP}}}{\partial z_k} = p_k - y_k + \beta(p_k \log p_k + p_k - p_k \sum_j p_j \log p_j - p_k)$$

$= (p_k - y_k) + \beta p_k [\log p_k - \sum_j p_j \log p_j]$

$= (p_k - y_k) - \beta p_k [\log p_k \text{의 mean 대비 deviation}]$

LS gradient:

$$\frac{\partial L_{\text{LS}}}{\partial z_k} = p_k - \tilde{y}_k = p_k - [(1-\alpha) y_k + \alpha/K]$$

**비교**: LS는 target을 shift, CP는 "high-probability class에 더 강한 반발". 비슷한 목표, 다른 mechanism.

### 정리 3.4 — MaxEnt as Implicit Prior

MaxEnt 분포는 "**prior information이 없다**"는 Bayesian 해석. Uniform prior + likelihood = posterior 구조에서 posterior = likelihood (prior cancels).

CP에서 $-\beta H(p)$는 "output distribution이 informative해야 한다는 우리의 데이터 bias"와 "최대한 uninformative하게 기대하는 prior"의 trade-off.

### 정리 3.5 — LS과 CP의 수렴 점

두 loss 모두 optimal $p^*$가 존재. 단:
- LS optimum: $p^*_c = 1 - \alpha + \alpha/K$, $p^*_{k \neq c} = \alpha/K$ (정리 1.3).
- CP optimum: 아날리틱하게 정확히 풀리지 않지만 비슷한 smoothing 효과.

---

## 🔬 수학적 유도

### CP의 Lagrangian 관점

$L_{\text{CP}} = L_{\text{CE}} - \beta H(p)$는 다음과 동치:

$$\min_p L_{\text{CE}} \text{ subject to } H(p) \geq H_0$$

(Lagrangian dual.) 즉 "CE를 최소화하되 entropy를 $H_0$ 이상으로 유지".

$\beta$가 "entropy constraint의 Lagrange multiplier" 역할. 큰 $\beta$ → 강한 entropy 제약.

### CP의 Softmax Temperature와의 관계

흥미롭게도, CP 훈련이 "softmax temperature scaling"과 유사한 효과. Post-hoc temperature scaling (Ch5-04)은 train 후 조정, CP는 train 중 내장.

**Guo 2017**: 두 접근이 **유사한 ECE 감소** 달성. 복잡한 modern NN에서.

### Taylor expansion으로 LS vs CP 비교

두 loss를 $p$ 주변에서 Taylor 전개:

$L_{\text{LS}} \approx L_{\text{CE}} - \alpha \cdot [\sum_k (y_k \log p_k) - (1/K) \sum_k \log p_k]$

$L_{\text{CP}} \approx L_{\text{CE}} - \beta \cdot \sum_k p_k \log p_k$

둘 다 negative entropy-like term을 추가. 수학적 form이 다르지만 training effect는 유사.

---

## 💻 실험으로 효과 검증

### 실험 1 — Confidence Penalty 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidencePenaltyLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        # Cross-entropy
        ce = F.nll_loss(log_p, target)
        # Negative entropy term
        neg_ent = (p * log_p).sum(-1).mean()   # H = -sum(p*logp), so neg_ent = -H
        return ce + self.beta * neg_ent  # adding -β*H = +β*neg_ent
```

### 실험 2 — CP vs LS의 유사성 검증

```python
# 같은 네트워크를 LS (α=0.1)와 CP (β=0.05)로 훈련
# ECE, accuracy 비교

# 전형적 결과 (CIFAR-10 ResNet-18):
# Baseline:   acc=94.0%, ECE=0.08
# LS α=0.1:   acc=94.2%, ECE=0.02
# CP β=0.05:  acc=94.1%, ECE=0.02
# Both:       acc=94.2%, ECE=0.015  (약간 추가 개선)
```

### 실험 3 — Entropy tracking during training

```python
import matplotlib.pyplot as plt

def track_output_entropy(model, loader):
    model.eval()
    entropies = []
    with torch.no_grad():
        for x, _ in loader:
            p = F.softmax(model(x), -1)
            H = -(p * p.clamp_min(1e-10).log()).sum(-1).mean()
            entropies.append(H.item())
    return sum(entropies) / len(entropies)

# Train 동안 entropy 변화
entropies_erm, entropies_cp = [], []
for epoch in range(epochs):
    # Training step ...
    entropies_erm.append(track_output_entropy(net_erm, val_loader))
    entropies_cp.append(track_output_entropy(net_cp, val_loader))

plt.plot(entropies_erm, label='ERM')
plt.plot(entropies_cp, label='CP')
plt.xlabel('Epoch'); plt.ylabel('Avg output entropy')
plt.axhline(np.log(10), ls='--', label='Uniform (log 10)')
plt.legend(); plt.show()
# → ERM은 entropy가 0 근처로 떨어짐 (over-confident)
# → CP는 entropy가 유지 (calibrated)
```

### 실험 4 — β sweep

```python
betas = [0.0, 0.01, 0.05, 0.1, 0.2]
for beta in betas:
    net = train_with_cp(beta)
    acc = evaluate_accuracy(net, test_loader)
    ece = compute_ece(net, test_loader)
    print(f"β={beta}: acc={acc:.3f}, ECE={ece:.3f}")
# 전형적:
# β=0.00: acc=0.940, ECE=0.080
# β=0.01: acc=0.941, ECE=0.045
# β=0.05: acc=0.941, ECE=0.018  <- sweet spot
# β=0.10: acc=0.938, ECE=0.015
# β=0.20: acc=0.925, ECE=0.020  # too much entropy regularization
```

---

## 🔗 실전 활용

### CP vs LS 선택

| 요인 | 권장 |
|------|------|
| 간단한 구현 | LS (1줄 변경, PyTorch 내장) |
| 다른 regularization 조합 | LS가 더 안정적 |
| Task-specific 조정 | CP (β로 훈련 동역학 세밀 제어 가능) |
| Multi-task learning | CP가 task별 다른 $\beta$ 가능 |

### 현대 LLM에서

- **Label smoothing**: 자주 $\alpha = 0.1$ (Transformer 원 논문부터 표준).
- **Confidence penalty**: 드물게 사용.
- **이유**: LS가 더 직관적, LLM 규모에서 overfitting은 덜한 문제.

### Combining with KD

Ch5-02의 Muller 2019: LS teacher가 bad distillation. CP teacher는? 연구 부족.

직관: CP는 teacher의 output entropy를 직접 증가 → dark knowledge가 오히려 **잘 보존** (uniform 아닌, confidence만 completion).

**가설**: CP teacher가 LS teacher보다 distillation에 더 적합. 경험적 검증 필요.

### Information Bottleneck과의 연결

"Info bottleneck"(Tishby 1999): representation이 input 정보를 줄이면서도 label 정보는 유지. CP는 비슷한 정신 — output이 "지나치게 specific"하지 않도록.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Entropy maximization이 "calibration"의 proxy | Specific task에서 underfitting 초래 가능 |
| $\beta$ hyperparameter | Grid search 필요 |
| Task-agnostic | Domain-specific prior 활용 못함 |
| LS와 거의 동등 | 특별한 새 장점 없음 |

**실전적 위치**: CP는 "LS의 theoretical variant" 정도로 이해. 실전에서는 LS가 더 자주 사용.

---

## 📌 핵심 정리

$$\boxed{L_{\text{CP}} = L_{\text{CE}} - \beta \cdot H(p) \quad \approx \quad L_{\text{LS}} \ (\text{gradient 관점})}$$

| 개념 | 의미 |
|------|------|
| **CP** | Output entropy를 target으로 하는 penalty |
| **MaxEnt 원리** | 제약 하 최대 entropy 분포 선택 |
| **LS와 유사** | 수학적 목적은 다르지만 effect 유사 |
| **실전 선택** | LS가 표준, CP는 variant |
| **Ch5 흐름** | 1→2→3→4: LS → KD → CP → Temperature Scaling |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $p = (0.9, 0.05, 0.03, 0.02)$의 entropy와 $p' = (0.7, 0.15, 0.1, 0.05)$의 entropy를 비교하라.

<details>
<summary>힌트 및 해설</summary>

$H(p) = -\sum p_k \log p_k$:

$H(p) = -(0.9 \log 0.9 + 0.05 \log 0.05 + 0.03 \log 0.03 + 0.02 \log 0.02)$
$\approx -(0.9 \cdot (-0.105) + 0.05 \cdot (-3.0) + 0.03 \cdot (-3.51) + 0.02 \cdot (-3.91))$
$\approx 0.095 + 0.150 + 0.105 + 0.078 \approx 0.428$ nats.

$H(p') \approx 0.70 \cdot 0.357 + 0.15 \cdot 1.90 + 0.1 \cdot 2.30 + 0.05 \cdot 3.00$
$\approx 0.250 + 0.285 + 0.230 + 0.150 \approx 0.915$ nats.

$H(p') > H(p)$ (더 spread-out).

Uniform $(0.25, 0.25, 0.25, 0.25)$의 entropy = $\log 4 \approx 1.386$ nats — 최대.

CP가 선호하는 방향: $p$ (confident) → $p'$ (moderate) → uniform (0.25 each).

</details>

**문제 2** (심화): CP의 $\beta$는 LS의 $\alpha$와 어떻게 대응되는가? 두 hyperparam을 "같은 regularization 강도"로 설정하는 방법은?

<details>
<summary>힌트 및 해설</summary>

**직접 대응은 없음**. 하지만 gradient equivalence를 근사적으로:

LS at small $\alpha$: true class gradient를 $1 - (1 - \alpha + \alpha/K) \approx 1 - \alpha$로 수정.

CP at small $\beta$: true class gradient에 $-\beta p_c \log p_c + \beta$ 추가.

**근사적 equivalence**:

$\beta \approx \alpha$ (class 수가 크면).

**실전**: 
- $\alpha = 0.1$ ↔ $\beta \approx 0.05 \sim 0.1$ (experimental).
- 같은 effect를 원하면 grid search로 tune.

**정확한 대응이 없는 이유**: 두 loss가 다른 space에서 작동. LS는 cross-entropy **target**을 수정, CP는 **output**의 entropy만 수정. 수학적으로 non-equivalent.

</details>

**문제 3** (이론-실전): MaxEnt 원리는 "정보가 없는 상태에서 uniform을 가정"이다. Deep learning에서 "정보"란 무엇이고 MaxEnt가 언제 적절한가?

<details>
<summary>힌트 및 해설</summary>

**"정보"** = labeled data + inductive bias (architecture, optimization).

MaxEnt 원리의 적절성:

1. **Train set이 작음**: data 정보 부족 → MaxEnt가 "**모르면 uniform**" 제공. Over-fit 방지.
2. **OOD input**: Training distribution 밖의 input에 대해 MaxEnt는 "**확신 없음**"을 표현.
3. **Calibration**: 예측 확률이 실제 정확도를 반영 — "uncertainty quantification"의 기초.

**부적절한 경우**:

1. **Strong prior available**: Domain knowledge가 강하면 uniform은 naive.
2. **Fine-grained task**: 20,000 bird species 분류에서 uniform은 무의미 (species 구조 무시).
3. **Imbalanced class**: uniform이 실제 class frequency와 불일치.

**실전 교훈**: CP/LS는 "**inductive bias가 약할 때**의 안전망". Strong task-specific priors 있으면 KD나 task-aware loss가 우선.

Bayesian 관점: CP는 **uniform prior**를 가정한 posterior regularization. 데이터가 풍부하면 prior의 영향 작음 (왜 big models에서 CP/LS가 약간만 효과적인가).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Label Smoothing & KD](./02-label-smoothing-kd.md) | [📚 README로 돌아가기](../README.md) | [04. Temperature Scaling ▶](./04-temperature-scaling.md) |

</div>
