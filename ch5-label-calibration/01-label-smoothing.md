# 01. Label Smoothing (Szegedy et al. 2016)

## 🎯 핵심 질문

- One-hot 라벨을 **$(1-\alpha)y + \alpha/K$** 로 대체하면 cross-entropy gradient는 어떻게 변하는가?
- 왜 이것이 **over-confidence를 방지**하는가?
- **Expected Calibration Error (ECE)** 는 어떻게 정의하고 측정하는가?
- $\alpha = 0.1$에서 top-1 accuracy는 유지되면서 calibration이 개선되는 이유는?

---

## 🔍 왜 label smoothing이 필요한가

Supervised classification의 표준 loss는 softmax cross-entropy:

$$L = -\sum_k y_k \log p_k = -\log p_{\text{true}}$$

One-hot label $y$에서 **오직 true class의 log-probability만** 중요. Gradient는 true class logit $z_{\text{true}}$를 **무한히 밀어붙임**:

- True class: $\partial L / \partial z_{\text{true}} = -(1 - p_{\text{true}})$ → $p_{\text{true}} \to 1$까지 밀어.
- False classes: $\partial L / \partial z_k = p_k$ → $p_k \to 0$까지 밀어.

결과: **모델이 over-confident** — $p_{\text{true}} = 0.99+$의 예측이 **실제 accuracy보다 높음**. Test set의 95% accuracy 모델이 예측 확률 99%를 준다면 이는 **poorly calibrated**.

**Label Smoothing** (Szegedy 2016): one-hot을 soft target으로 대체하여 gradient가 logit을 **무한히 밀지 않도록**. 결과: calibration 개선, 가끔 accuracy 개선.

---

## 📐 수학적 선행 조건

- Cross-entropy loss $L = -\sum_k y_k \log p_k$
- Softmax $p_k = \exp(z_k)/\sum_j \exp(z_j)$
- 정보이론: entropy $H(p) = -\sum_k p_k \log p_k$, cross-entropy, KL divergence

---

## 📖 직관적 이해

### Label Smoothing 공식

$K$-class classification, true class $c$. One-hot $y_k = \mathbb{1}[k = c]$.

**Smoothed label** ($\alpha \in (0, 1)$):

$$\tilde{y}_k = (1 - \alpha) y_k + \alpha / K = \begin{cases} 1 - \alpha + \alpha/K & k = c \\ \alpha/K & k \neq c \end{cases}$$

$\alpha = 0.1$: true class 확률 0.91, 다른 class 각 $0.1/K$.

### Cross-Entropy Gradient의 변화

One-hot: $\partial L/\partial p_c = -1/p_c$ → $p_c \to 1$일 때 gradient $\to -1$ (bounded이지만 target 항상 1).

Smoothed: target $1 - \alpha + \alpha/K$. Gradient $\partial L/\partial p_c = -(1 - \alpha + \alpha/K)/p_c$. Target이 1이 아니라 $\sim 0.91$에서 멈추므로 $p_c$가 0.91 초과하면 **gradient가 반대 방향**으로 밀어 (negative penalty on $p_c > 0.91$).

즉 **훈련된 모델의 $p_c$가 0.91 정도에 머무름**. Over-confidence 차단.

### Expected Calibration Error (ECE)

"예측 확률이 실제 정확도와 얼마나 일치하는가":

$$\text{ECE} = \sum_b \frac{|B_b|}{n} \big|\text{avg\_conf}(B_b) - \text{accuracy}(B_b)\big|$$

$B_b$는 confidence bin (e.g. 10 equal-width bins). ECE = 0이면 **perfectly calibrated** (예측 확률 $p$이 실제 정확도).

ResNet-50 ImageNet: ECE ~5-8% (untrained calibration). Label Smoothing으로 1-2%까지 감소.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Label Smoothing

$K$-class classification. $\alpha \in [0, 1)$. True class $c$에 대한 smoothed target:

$$\tilde{y}_k = \begin{cases} 1 - \alpha + \alpha/K & k = c \\ \alpha/K & k \neq c \end{cases}$$

**Label Smoothing loss**:

$$L_{\text{LS}}(p, y) = -\sum_k \tilde{y}_k \log p_k$$

### 정의 1.2 — Expected Calibration Error

모델의 prediction $p(x) = \text{softmax}(f(x))$. $M$ bins $B_1, \ldots, B_M$ ($[0, 1]$을 균등 분할):

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

- $\text{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} \max_k p_k(x_i)$ (average predicted confidence).
- $\text{acc}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} \mathbb{1}[\hat{y}_i = y_i]$ (actual accuracy).

### 정리 1.3 — LS 훈련의 Optimal Logit (Szegedy 2016)

**Optimal logit** under LS (gradient=0):

$$z_k^* = \begin{cases} \log\frac{1 - \alpha + \alpha/K}{\alpha/K} = \log\frac{K - (K-1)\alpha}{\alpha} & k = c \\ \text{const} & k \neq c \end{cases}$$

즉 **true class logit이 다른 logits보다 $\log(K/\alpha - K + 1)$만큼 크다** — 유한. One-hot에서는 $+\infty$.

$\alpha = 0.1, K = 1000$: $z_c^* - z_k^* = \log(10,000 - 900) \approx 9.12$ — finite margin.

### 정리 1.4 — LS와 Confidence Penalty의 관계

$L_{\text{LS}}$는 equivalent하게:

$$L_{\text{LS}}(p, y) = L_{\text{CE}}(p, y) + \alpha \cdot \text{KL}(\text{Uniform} \| p)$$

**Cross-entropy + uniform distribution과의 KL**. Uniform에 가깝게 가도록 push → entropy 증가.

### 정리 1.5 — 경험적 calibration 개선 (Müller 2019)

Label Smoothing은 대부분 case에서 ECE 감소시키지만:

- Knowledge distillation과 함께 사용 시 student의 **calibration이 오히려 나빠질 수 있음**.
- $\alpha$ 너무 크면 accuracy 저하.

---

## 🔬 수학적 유도

### 정리 1.3 — Optimal Logit 유도

$L_{\text{LS}} = -\sum_k \tilde{y}_k \log p_k$. $p_k = e^{z_k}/\sum_j e^{z_j}$.

$\partial L/\partial z_k = p_k - \tilde{y}_k$ (softmax의 gradient 표준 결과).

Optimality: $p_k^* = \tilde{y}_k$ → $p_c^* = 1 - \alpha + \alpha/K$, $p_{k \neq c}^* = \alpha/K$.

Logit 관점: $p_k = e^{z_k}/Z$, $Z = \sum_j e^{z_j}$. Taking log:

$\log p_c^* - \log p_k^* = z_c - z_k$

$\log\frac{1 - \alpha + \alpha/K}{\alpha/K} = z_c - z_k$

$z_c - z_k = \log\frac{(K - 1)(1 - \alpha) + 1}{(1 - \alpha + \alpha/K) \cdot \alpha / K} \approx \log \frac{K - K\alpha + 1}{\alpha}$ (simplification).

Approximation $\alpha$ small: $z_c - z_k \approx \log(K(1-\alpha)/\alpha + 1/\alpha) \approx \log(K/\alpha)$. $\square$

### 정리 1.4 증명

$L_{\text{LS}}(p, y) = -\sum_k \tilde{y}_k \log p_k = -\sum_k [(1-\alpha) y_k + \alpha/K] \log p_k$

$= -(1-\alpha) \sum_k y_k \log p_k - \frac{\alpha}{K} \sum_k \log p_k$

$= (1-\alpha) L_{\text{CE}} + \frac{\alpha}{K}\sum_k [-\log p_k]$

$\frac{\alpha}{K}\sum_k (-\log p_k) = \alpha \cdot \mathbb{E}_{k \sim \text{Unif}}[-\log p_k] = \alpha [\text{KL}(\text{Unif} \| p) + H(\text{Unif})]$

$H(\text{Unif}) = \log K$는 constant. 따라서:

$L_{\text{LS}} = (1-\alpha) L_{\text{CE}} + \alpha \cdot \text{KL}(\text{Unif} \| p) + \alpha \log K$

$w$ 독립 상수 제거하면:

$L_{\text{LS}} \propto (1-\alpha) L_{\text{CE}} + \alpha \cdot \text{KL}(\text{Unif} \| p) \quad \square$

**해석**: Cross-entropy + "output 분포를 uniform에 가깝게". 후자가 **entropy regularization**.

---

## 💻 실험으로 효과 검증

### 실험 1 — Label Smoothing loss 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.K = num_classes
    def forward(self, logits, target):
        # target: class indices (not one-hot)
        log_p = F.log_softmax(logits, dim=-1)
        # Smoothed target
        smooth_target = torch.full_like(log_p, self.alpha / self.K)
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.alpha + self.alpha / self.K)
        loss = -(smooth_target * log_p).sum(dim=-1).mean()
        return loss

# PyTorch 1.10+ 내장:
# nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 실험 2 — Training with vs without LS (CIFAR-10 ResNet-18)

```python
import torch
import torchvision
import torch.nn as nn

def train_model(use_ls=False, alpha=0.1, epochs=50):
    net = torchvision.models.resnet18(num_classes=10)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=alpha if use_ls else 0.0)
    
    for epoch in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward(); opt.step()
    return net

net_erm = train_model(False)
net_ls = train_model(True, alpha=0.1)
```

### 실험 3 — ECE 측정 및 비교

```python
def compute_ece(logits, labels, n_bins=15):
    probs = F.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()
    
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        in_bin = (confs > bins[i]) & (confs <= bins[i+1])
        if in_bin.sum() > 0:
            avg_conf = confs[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += in_bin.float().mean() * (avg_conf - avg_acc).abs()
    return ece.item()

# Test 데이터에서 비교
with torch.no_grad():
    all_logits_erm = torch.cat([net_erm(x) for x, _ in test_loader])
    all_logits_ls = torch.cat([net_ls(x) for x, _ in test_loader])

ece_erm = compute_ece(all_logits_erm, test_labels)
ece_ls = compute_ece(all_logits_ls, test_labels)
print(f"ECE (ERM):         {ece_erm:.4f}")
print(f"ECE (LS α=0.1):    {ece_ls:.4f}")
# 전형적 결과:
#   ECE (ERM):     0.08 (8%)
#   ECE (LS):      0.02 (2%) — 4배 개선
```

### 실험 4 — Reliability diagram 시각화

```python
import matplotlib.pyplot as plt

def plot_reliability(logits, labels, n_bins=15, title=''):
    probs = F.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()
    
    bins = torch.linspace(0, 1, n_bins + 1)
    acc_per_bin, conf_per_bin = [], []
    for i in range(n_bins):
        in_bin = (confs > bins[i]) & (confs <= bins[i+1])
        if in_bin.sum() > 0:
            acc_per_bin.append(accuracies[in_bin].mean().item())
            conf_per_bin.append(confs[in_bin].mean().item())
        else:
            acc_per_bin.append(0); conf_per_bin.append(0)
    
    plt.bar(bins[:-1].numpy(), acc_per_bin, width=1/n_bins, align='edge',
            label='Accuracy', alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.title(title)
    plt.xlabel('Confidence'); plt.ylabel('Accuracy')
    plt.legend()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(axes[0]); plot_reliability(all_logits_erm, test_labels, title='ERM')
plt.sca(axes[1]); plot_reliability(all_logits_ls, test_labels, title='Label Smoothing')
plt.tight_layout(); plt.show()
```

**관찰**: ERM은 high-confidence bin에서 **accuracy가 confidence보다 낮음** (over-confident). LS는 더 일관.

### 실험 5 — $\alpha$ sweep

```python
alphas = [0.0, 0.05, 0.1, 0.2, 0.4]
for alpha in alphas:
    net = train_model(True, alpha=alpha)
    with torch.no_grad():
        all_logits = torch.cat([net(x) for x, _ in test_loader])
    acc = (all_logits.argmax(-1) == test_labels).float().mean().item()
    ece = compute_ece(all_logits, test_labels)
    print(f"α={alpha:.2f}: acc={acc:.4f}, ECE={ece:.4f}")
# 전형적 결과:
# α=0.00: acc=0.940, ECE=0.082
# α=0.05: acc=0.940, ECE=0.045
# α=0.10: acc=0.942, ECE=0.021  # sweet spot
# α=0.20: acc=0.939, ECE=0.015
# α=0.40: acc=0.920, ECE=0.025  # too much smoothing
```

---

## 🔗 실전 활용

### 표준 $\alpha$ 선택

| Dataset / Task | 권장 $\alpha$ |
|-------------|-------------|
| ImageNet | 0.1 |
| CIFAR-100 | 0.1 |
| CIFAR-10 | 0.05 ~ 0.1 |
| Machine translation | 0.1 |
| Speech recognition | 0.1 ~ 0.2 |
| Binary classification | LS 덜 유용 (2 class만 있어 uniform이 [0.5, 0.5]) |

### 다른 regularization과 조합

- **Mixup**: Mixup도 soft label을 주기 때문에 LS와 redundant. 둘 다 쓰면 $\alpha_{\text{LS}}$ 작게.
- **Weight decay**: 서로 다른 regularization, 독립적으로 적용 가능.
- **Temperature scaling (Ch5-04)**: LS와 다른 mechanism — 훈련 후 post-hoc calibration. 함께 사용 가능.

### 주의사항 — Knowledge Distillation과의 상호작용

**Müller 2019의 중요한 발견**: Teacher를 label smoothing으로 훈련하면 student distillation이 오히려 **악화**.

이유: LS가 logit의 **class 간 structure**(semantic similarity)를 **wash out**시켜 dark knowledge를 손실.

**실전**: Distillation에서 teacher 훈련은 **LS 없이**. Student는 distillation loss (soft target) 자체가 regularization 역할.

### LLM에서의 Label Smoothing

Autoregressive language model의 next-token prediction에서 LS는:
- Transformer 원 논문(Vaswani 2017)은 $\alpha = 0.1$ 사용.
- 현대 LLM (GPT, Llama)에서는 LS 자주 사용 안 함 — 대신 데이터 scale로 calibration 관리.

### 변형

- **Online Label Smoothing** (Zhang 2020): 훈련 중 model의 output을 참고하여 dynamic smoothing.
- **Adversarial Label Smoothing** (Müller 2019 Appendix): adversarial 방향으로 smoothing.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Uniform smoothing target | Class 간 semantic similarity 무시 (hard KD가 더 정확) |
| Single $\alpha$ for all classes | Class imbalance 있으면 class-dependent smoothing이 더 나음 |
| $\alpha$ static | 훈련 중 동적 조정이 유용할 수 있음 |
| Well-tuned for classification | Regression에서는 N/A |
| ECE 평가 bin 수 | Bin 선택에 따라 ECE 수치 달라짐 |

**주의**: LS가 "항상 좋다"는 신화. 특수 상황(KD teacher, fine-grained classification)에서는 harmful일 수 있음.

---

## 📌 핵심 정리

$$\boxed{\tilde{y}_k = (1-\alpha)y_k + \alpha/K, \quad L_{\text{LS}} = (1-\alpha)L_{\text{CE}} + \alpha \cdot \text{KL}(\text{Unif}\|p)}$$

| 개념 | 의미 |
|------|------|
| **Label smoothing** | One-hot을 soft target으로, target class에 $1 - \alpha + \alpha/K$ |
| **Optimal logit margin** | $z_c^* - z_k^* = \log(K/\alpha)$ (유한, one-hot은 $\infty$) |
| **ECE** | Expected Calibration Error — 실전 calibration 지표 |
| **Entropy reg 해석** | LS = CE + $\alpha$ KL to uniform |
| **다음 질문** | Hinton의 KD와 어떻게 연결? → Ch5-02 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K = 1000$ (ImageNet), $\alpha = 0.1$일 때 smoothed label의 true class 확률과 false class 확률은?

<details>
<summary>힌트 및 해설</summary>

- True class: $1 - 0.1 + 0.1/1000 = 0.9001$.
- False class (각): $0.1/1000 = 0.0001$.

Sum check: $0.9001 + 999 \times 0.0001 = 0.9001 + 0.0999 = 1.0$ ✓.

이 수치는 model의 optimal output이 $(0.9001, 0.0001, \ldots)$에 가까워지도록 유도. 그러면 top-1 confidence ≈ 0.9, top-1 accuracy ≈ 0.9일 때 잘 calibrated.

</details>

**문제 2** (심화): 정리 1.3의 optimal logit margin $z_c^* - z_k^* = \log(K/\alpha)$를 사용하여, $\alpha = 0.1$일 때 ImageNet ($K = 1000$)과 CIFAR-10 ($K = 10$)의 margin을 비교하라.

<details>
<summary>힌트 및 해설</summary>

- ImageNet: $\log(1000/0.1) = \log(10000) \approx 9.21$.
- CIFAR-10: $\log(10/0.1) = \log(100) \approx 4.61$.

**ImageNet margin이 2배**. 이는 LS의 "regularization 강도"가 task에 따라 다름을 의미.

**실용적 함의**:
- ImageNet에서 $\alpha = 0.1$이 더 "약한" regularization (큰 margin 허용) → accuracy loss 거의 없음.
- CIFAR-10에서 같은 $\alpha$는 "강한" regularization → margin이 작아 어려운 example은 학습 느림.

**이것이 왜** 큰 class 수 (ImageNet 1000) 에서 LS가 **더 안정적**인가의 이유. Binary classification ($K = 2$)에서는 margin이 $\log(2/0.1) \approx 3$으로 아주 작아 LS 효과 미미.

</details>

**문제 3** (이론-실전): Label smoothing이 왜 Knowledge Distillation의 teacher 훈련에서 **harmful**한가? "Dark knowledge" 관점으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Dark knowledge** (Hinton 2015): Teacher의 soft target $p_k$가 담는 **class 간 유사도 정보**. 예: "3" 이미지에 대해 softmax가 $p_3 = 0.95, p_8 = 0.03, p_5 = 0.01, \ldots$이면 "3이 8과 유사함"이라는 정보.

**Label Smoothing effect**: Teacher를 LS로 훈련하면 output logits의 **relative distances**가 왜곡:

- LS 없음: $p_k$가 진짜 class confusion에 비례.
- LS $\alpha = 0.1$: 모든 false class가 $\alpha/K$로 수렴 → confusion 정보 **smoothed out**.

**Distillation 후 student가 잃는 것**: "3과 8의 유사도" 같은 fine-grained structure. Muller 2019 Figure 1: LS teacher의 logit space에서 class들이 **tighter cluster**로 모이고 inter-class structure가 감소.

**실전 권장**:
- Teacher 훈련: **LS 없이**.
- Student 훈련: KD의 soft target 자체가 regularization. 추가 LS는 duplicate.

이는 "regularization 기법들은 독립적이지 않다 — 조합 효과를 고려해야 한다"는 중요한 교훈. Ch5-02에서 KD 상세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Chapter 4 → 05. Contrastive Learning](../ch4-data-augmentation/05-contrastive.md) | [📚 README로 돌아가기](../README.md) | [02. Label Smoothing과 KD ▶](./02-label-smoothing-kd.md) |

</div>
