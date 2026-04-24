# 04. Temperature Scaling (Guo et al. 2017)

## 🎯 핵심 질문

- **Temperature Scaling**은 훈련 후 **post-hoc** calibration — 왜 이것이 유용한가?
- 단일 scalar $T$로 accuracy를 유지하면서 ECE를 개선하는 메커니즘은?
- $T$ 최적값은 어떻게 학습하는가?
- **Platt scaling, isotonic regression**과의 비교는?

---

## 🔍 왜 "post-hoc" calibration이 필요한가

Ch5-01, 02, 03의 기법들은 **훈련 중** calibration 개선:
- Label Smoothing, KD, Confidence Penalty — loss에 조합.

**문제**: 
- 이미 **훈련 완료된 model**을 재훈련하기 어려울 때.
- Large model fine-tuning은 expensive.
- Production pre-trained model (ImageNet ResNet-50 등) 재훈련 불가능.

**해결**: **Post-hoc** calibration — 훈련 후 small parametric adjustment.

**Temperature Scaling** (Guo 2017): **단일 scalar** $T$로 모든 logit을 나눈다.

$$p^{\text{cal}} = \text{softmax}(z / T)$$

- $T = 1$: 원래 softmax (변화 없음).
- $T > 1$: softer distribution (less confident).
- $T < 1$: sharper distribution (more confident).

**Modern NN은 over-confident** → $T > 1$이 대부분 최적.

**놀라운 사실**: 단순한 1-parameter 조정으로 **state-of-the-art calibration** 달성 (Guo 2017 main finding).

---

## 📐 수학적 선행 조건

- Ch5-01: ECE 정의
- Softmax의 temperature 해석
- Validation set, NLL minimization

---

## 📖 직관적 이해

### Over-Confident NN

Guo 2017의 관찰: modern NN (ResNet, DenseNet)은 **매우 over-confident**.

- LeNet (old): reasonably calibrated.
- ResNet-50 on CIFAR-100: **극단적 over-confidence** — 99% confidence 예측이 80% accuracy.

원인 (hypothesis):
1. **Depth**: 깊은 네트워크가 sharp logits.
2. **Large model + overfitting**: NLL minimization이 $p_c \to 1$ 압박.
3. **Batch Normalization**: 정확한 이유 불명, 경험적 관찰.

### Temperature Scaling의 Simple Fix

모든 logit을 $T > 1$로 나눠 부드럽게:

$$p_k^{\text{cal}} = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

**핵심 특징**:
- **Argmax 불변**: $\arg\max_k p_k^{\text{cal}} = \arg\max_k p_k$. Top-1 accuracy 보존.
- **Rank 불변**: 모든 class의 rank 유지.
- **Confidence만 조정**: 최대 확률이 감소, 모든 확률 softer.

### 최적 $T$ 학습

Validation set의 **Negative Log Likelihood**:

$$T^* = \arg\min_T \text{NLL}(T) = \arg\min_T \sum_i -\log p^{(T)}_{c_i}(x_i)$$

단일 scalar 최적화 — LBFGS나 simple grid search로 초.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Temperature Scaling

Trained model $f$의 logit $z = f(x)$. Temperature-scaled softmax:

$$p^{(T)}_k(x) = \frac{\exp(z_k(x) / T)}{\sum_j \exp(z_j(x) / T)}$$

$T = 1$은 no scaling. Post-hoc calibration: $T$를 validation NLL 최소화로 학습.

### 정리 4.2 — Accuracy Preservation

**모든 $T > 0$**에 대해:

$$\arg\max_k p^{(T)}_k(x) = \arg\max_k p^{(1)}_k(x)$$

즉 **top-1 prediction 불변**. 이는 $\exp$의 monotonicity와 symmetric scaling에서 나옴.

### 정리 4.3 — NLL 최적화의 Convexity

$$\text{NLL}(T) = \sum_i \log \sum_k \exp((z_k(x_i) - z_{c_i}(x_i))/T)$$

$T$에 대해 convex (log-sum-exp은 convex, linear in $1/T$).

**함의**: Unique global minimum, gradient descent로 안정 최적화.

### 정리 4.4 — ECE 감소 보장 (Local)

Validation set에서 $T^*$로 NLL을 최소화하면 ECE도 local minimum 근처. 그러나:
- **NLL과 ECE는 직접 대응 안 됨** — NLL 최소화가 항상 ECE 최소 아님.
- 실전에서 두 metric이 대부분 일치.

### 정의 4.5 — Platt Scaling (이진 분류)

$$p^{\text{Platt}}(x) = \sigma(a \cdot z(x) + b)$$

$z$는 model의 output logit (scalar for binary). $(a, b)$를 validation로 학습. Temperature Scaling의 이진 version + bias shift.

### 정의 4.6 — Isotonic Regression

Non-parametric calibration: $[0, 1]$의 monotonic function $g: \hat{p} \to p^{\text{cal}}$ 학습.

Validation에서 confidence bin별로 actual accuracy와 fitting.

**Temperature Scaling vs Isotonic**:
- TS: 1 parameter, parametric.
- Isotonic: $O(n)$ parameters, non-parametric, flexible.

### 정리 4.7 — Temperature Scaling과 Label Smoothing의 관계

Temperature scaling은 **"logit 전체를 scale"**, Label smoothing은 **"target을 smooth"**. 두 기법 모두 gradient가 target logit을 "무한히 밀지 못하게" 함:

- LS: target이 $1 - \alpha + \alpha/K < 1$이라 gradient가 finite point에서 멈춤.
- TS: gradient가 $T$로 나뉘어 magnitude 감소.

---

## 🔬 수학적 유도

### 정리 4.2 증명

$p^{(T)}_k = e^{z_k/T}/Z(T)$. $T > 0$이면 $e^{\cdot/T}$는 monotonically increasing in $z_k$. 따라서 $\arg\max_k e^{z_k/T} = \arg\max_k z_k$, 그리고 denominator가 $k$-independent이므로 $\arg\max_k p^{(T)}_k = \arg\max_k z_k = \arg\max_k p^{(1)}_k$. $\square$

### 정리 4.3 증명

$$\text{NLL}(T) = -\sum_i \log p^{(T)}_{c_i}(x_i) = \sum_i \left[-z_{c_i}/T + \log \sum_k e^{z_k/T}\right]$$

$u = 1/T$로 substitution:

$= \sum_i [-u \cdot z_{c_i} + \log \sum_k e^{u \cdot z_k}]$

첫 항: $u$의 linear function.

둘째 항: $u \cdot z$의 log-sum-exp = convex in $u$ (log-sum-exp은 convex).

Convex sum = convex. 따라서 NLL이 $u = 1/T$의 convex function.

$T$에 대한 convexity: $u = 1/T$는 $T > 0$에서 strictly decreasing bijection. 그러나 convex/convex composition은 일반적으로 convex 아님 — 엄밀한 argument는 direct 2nd derivative 계산 혹은 $u$-space 최적화 권장.

**실전**: $u = 1/T$ 공간에서 $\log T = -\log u$로 변환 → 명확한 convex 최적화.

### TS의 Gradient

$\frac{\partial \text{NLL}}{\partial T}$ 계산:

$\frac{\partial p^{(T)}_{c_i}}{\partial T}$는 $p^{(T)}_{c_i}$ 정의와 chain rule로. 결과:

$\frac{\partial \text{NLL}}{\partial T} = -\frac{1}{T^2}\sum_i [z_{c_i} - \sum_k p^{(T)}_k z_k]$

$= -\frac{1}{T^2}\sum_i [z_{c_i} - \mathbb{E}_{k \sim p^{(T)}_k}[z_k]]$

이는 "true class logit vs expected logit"의 차이의 합. 직관: $T^*$에서 이 차이가 zero.

---

## 💻 실험으로 효과 검증

### 실험 1 — Temperature Scaling 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)  # initial T > 1
    def forward(self, logits):
        return logits / self.T

def fit_temperature(model, val_loader):
    """Validation NLL로 T 최적화 (LBFGS)."""
    ts = TemperatureScaling().cuda()
    optimizer = torch.optim.LBFGS([ts.T], lr=0.01, max_iter=50)
    
    # 모든 validation logit 수집
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            all_logits.append(model(x.cuda()))
            all_labels.append(y.cuda())
    all_logits = torch.cat(all_logits); all_labels = torch.cat(all_labels)
    
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(ts(all_logits), all_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return ts.T.item()

T_opt = fit_temperature(model, val_loader)
print(f"Optimal T: {T_opt:.4f}")
# ResNet-50 ImageNet에서 T_opt ≈ 1.2 ~ 1.4
```

### 실험 2 — ECE 비교 (pre- vs post-scaling)

```python
def compute_ece_with_T(model, loader, T=1.0, n_bins=15):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.cuda()) / T
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs); all_labels.append(y)
    all_probs = torch.cat(all_probs); all_labels = torch.cat(all_labels)
    
    confs, preds = all_probs.max(-1)
    accs = (preds == all_labels).float()
    
    ece = 0
    bins = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1])
        if mask.sum() > 0:
            ece += mask.float().mean().item() * abs(confs[mask].mean() - accs[mask].mean()).item()
    return ece

ece_before = compute_ece_with_T(model, test_loader, T=1.0)
ece_after = compute_ece_with_T(model, test_loader, T=T_opt)
print(f"ECE before: {ece_before:.4f}")
print(f"ECE after : {ece_after:.4f}")
# 전형적: ECE before 0.07, ECE after 0.01
```

### 실험 3 — NLL vs T curve

```python
import numpy as np
import matplotlib.pyplot as plt

Ts = np.linspace(0.5, 3.0, 50)
nlls = []
for T in Ts:
    with torch.no_grad():
        scaled_logits = all_logits / T
        nll = F.cross_entropy(scaled_logits, all_labels).item()
    nlls.append(nll)

plt.plot(Ts, nlls)
plt.axvline(T_opt, ls='--', c='r', label=f'T*={T_opt:.2f}')
plt.xlabel('T'); plt.ylabel('Validation NLL')
plt.title('Temperature Scaling — NLL is convex in T')
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

### 실험 4 — Accuracy is preserved

```python
def get_accuracy(model, loader, T=1.0):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.cuda()) / T
            preds = logits.argmax(-1)
            correct += (preds == y.cuda()).sum().item()
            total += y.size(0)
    return correct / total

acc_before = get_accuracy(model, test_loader, T=1.0)
acc_after = get_accuracy(model, test_loader, T=T_opt)
print(f"Accuracy before: {acc_before:.4f}")
print(f"Accuracy after : {acc_after:.4f}")
# → 정확히 같음 (top-1 argmax 불변)
```

### 실험 5 — Platt vs Isotonic vs TS 비교 (binary)

```python
# Binary classification task
# 같은 model에 다른 calibration 기법 적용

# Temperature Scaling: NLL-minimizing T
# Platt Scaling: σ(a*z + b), validation으로 a, b 학습
# Isotonic: bin-based monotonic 매핑

# 전형적 결과 (AUROC 불변, Brier score 개선):
# Uncalibrated: Brier=0.25, ECE=0.12
# TS:            Brier=0.22, ECE=0.03
# Platt:         Brier=0.22, ECE=0.03
# Isotonic:      Brier=0.21, ECE=0.02  (약간 더 나은 calibration, more params)
```

---

## 🔗 실전 활용

### Temperature Scaling의 실전 워크플로우

```python
# 1. 훈련 완료된 model 사용 (training time에는 calibration 무시)
trained_model = load_pretrained()

# 2. Validation set에서 T 최적화
T = fit_temperature(trained_model, val_loader)
print(f"Learned T: {T:.3f}")

# 3. Inference 시 T 적용
def calibrated_predict(x):
    logits = trained_model(x) / T
    return F.softmax(logits, dim=-1)
```

### 언제 TS가 유용한가

- **Production model**: 재훈련 불가능 — TS로 post-hoc 개선.
- **Ensemble calibration**: 여러 model의 ensemble output도 TS로 fine-tune.
- **Uncertainty quantification**: Risk-sensitive app (medical, autonomous driving).

### 언제 TS가 불충분한가

- **Class-specific mis-calibration**: 특정 class만 over-confident인 경우 — TS는 전체 scale만 바꿈. 이 때 **Matrix Scaling** (parameter matrix $W$ for logit transformation) 또는 **Dirichlet Calibration** (Kull 2019) 사용.
- **Distribution shift**: Val과 test distribution이 크게 다르면 TS도 불충분.

### TS의 robustness

Temperature scaling의 장점은 **단순성과 robustness**:
- **Overfitting 없음**: 1 parameter → validation set small해도 안전.
- **Architecture-agnostic**: CNN, Transformer 모두 적용 가능.
- **Composable**: Dropout, Mixup, LS 등과 함께 사용 가능.

### 현대 LLM에서의 응용

Large LM의 token probability calibration에도 TS 적용. CLIP 같은 foundation model에서도 post-hoc T 조정 일반화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 모든 class에 같은 $T$ | Class-imbalanced에서는 class-dependent 필요 |
| NLL 최적화가 ECE 최소 근사 | 직접 대응 아님 |
| Val과 test가 같은 distribution | Distribution shift 상황에서 약함 |
| Logit magnitude만 조정 | Logit direction 잘못되면 교정 불가 |
| 훈련된 model이 "거의 맞음" | 심각한 miscalibration은 TS로 복구 불가 |

**핵심 통찰**: TS는 "model이 대체로 맞는데 over-confident일 뿐"인 상황에서 효과적. Major accuracy issue가 있으면 model 자체 개선 필요.

---

## 📌 핵심 정리

$$\boxed{T^* = \arg\min_T \sum_i -\log \text{softmax}(z(x_i)/T)_{c_i}, \quad p^{\text{cal}} = \text{softmax}(z/T^*)}$$

| 개념 | 의미 |
|------|------|
| **Temperature Scaling** | 단일 scalar $T$로 logits 나눔 |
| **Post-hoc** | 훈련 후 val set으로 $T$ 학습, 재훈련 없음 |
| **Accuracy preserved** | Top-1 argmax 불변 (정리 4.2) |
| **Guo 2017 결과** | Modern NN의 ECE를 0.07 → 0.01 수준으로 개선 |
| **Ch5 마무리** | Train-time (LS, KD, CP) + Post-hoc (TS) 조합 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Logit $z = (2, 1, 0.5)$를 $T = 1$과 $T = 2$로 softmax하여 비교하라.

<details>
<summary>힌트 및 해설</summary>

**$T = 1$**: $\exp(z) = (e^2, e^1, e^{0.5}) \approx (7.39, 2.72, 1.65)$. Sum $= 11.76$. $p \approx (0.629, 0.231, 0.140)$.

**$T = 2$**: $z/2 = (1, 0.5, 0.25)$. $\exp = (2.72, 1.65, 1.28)$. Sum $= 5.65$. $p \approx (0.481, 0.292, 0.227)$.

**관찰**: 
- Top-1 prediction 같음: class 0.
- Confidence 감소: 0.629 → 0.481.
- Distribution이 uniform에 가까워짐 (entropy 증가).

이것이 TS의 핵심 — "confidence만 조정, prediction 유지".

</details>

**문제 2** (심화): ResNet-50 ImageNet validation에서 $T^* = 1.3$이 학습되었다. Test set에서 이 $T$가 여전히 effective한가? Validation-test gap 상황을 분석하라.

<details>
<summary>힌트 및 해설</summary>

**일반적**: 같은 distribution이면 $T^*$가 test에도 effective.

**Val-test gap 유발 요인**:
1. **Val set이 작음**: $T^*$의 overfitting. Solution: larger val set, cross-validation.
2. **Distribution shift**: Val과 test 다른 subpopulation — e.g. val에 서양 얼굴, test에 아시아 얼굴. Solution: domain-specific $T$.
3. **Class imbalance**: Val과 test의 class ratio 다름. Solution: stratified val.

**Check**: Test ECE도 val ECE와 비슷한지. 크게 차이 나면 val set이 representative하지 않음.

**실전**: $T$는 매우 간단하므로 overfitting은 드물지만, **많은 hyperparameter (Matrix Scaling 등)** 는 val에 overfit 쉬움. TS의 장점은 "overfitting-proof".

</details>

**문제 3** (이론-실전): Label Smoothing으로 훈련된 model에 추가로 Temperature Scaling을 적용하면 어떻게 되는가? 두 기법은 redundant인가 complementary인가?

<details>
<summary>힌트 및 해설</summary>

**Empirical**: 두 기법이 **대체로 complementary**. LS + TS가 LS alone보다 ECE 약간 더 감소.

**이유**:
- **LS**: Training time — target의 structure 수정 → 훈련 dynamics 변화.
- **TS**: Post-hoc — 이미 훈련된 logits의 scale 조정.

LS가 gradient를 수정해서 optimal $T = 1$에 가깝게 훈련. 그래도 약간의 residual over-confidence 남아있어 TS가 추가로 개선.

**경험적 관찰** (Guo 2017 Table 3):
- LS alone: ECE 0.015.
- LS + TS: ECE 0.008.

**다만 주의**: LS 강도 $\alpha$와 TS 강도 $T$가 "상호 보완"되어야. 둘 다 크면 under-confident → NLL 증가.

**현대 모델 recipe**: LS + TS 조합이 일반적. 특히 medical / autonomous driving 등 uncertainty critical.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Confidence Penalty](./03-confidence-penalty.md) | [📚 README로 돌아가기](../README.md) | [Chapter 6 → 01. Early Stopping = L2 ▶](../ch6-early-stopping-implicit/01-early-stopping-as-l2.md) |

</div>
