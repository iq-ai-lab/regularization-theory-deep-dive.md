# 02. Label Smoothing과 Knowledge Distillation

## 🎯 핵심 질문

- **Knowledge Distillation** (Hinton 2015)의 "dark knowledge"는 무엇이며 왜 중요한가?
- Teacher의 softmax에 **temperature $T$**를 적용하는 이유는?
- Label Smoothing은 KD의 어떤 extreme case로 볼 수 있는가?
- Muller 2019가 발견한 "LS teacher는 bad teacher"의 원인은?

---

## 🔍 왜 KD와 LS를 함께 보는가

Ch5-01에서 LS를 봤다. 같은 "soft target" 아이디어가 **Knowledge Distillation**에도 존재:

- LS: uniform mixture $(1-\alpha)\text{one-hot} + \alpha \cdot \text{Uniform}$.
- KD: teacher의 softmax output을 soft target으로.

**공통점**: 모두 "hard target을 soft target으로 대체".

**차이점**: LS는 **uniform** soft (task-agnostic), KD는 **teacher-informed** soft (학습된 class similarity).

이 문서는 KD의 메커니즘, LS와의 수학적 관계, 그리고 **둘 함께 쓰면 왜 문제인가**를 정리.

---

## 📐 수학적 선행 조건

- Ch5-01: Label smoothing, cross-entropy 구조
- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): KL divergence
- 정보이론: KL, cross-entropy

---

## 📖 직관적 이해

### Knowledge Distillation (Hinton 2015)

**설정**: Large "teacher" model $T$ → small "student" model $S$. Student를 teacher의 soft output에 fit.

**Softmax with Temperature**:

$$p^{(T)}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

$T = 1$: 원래 softmax. $T > 1$: 더 부드러운 분포 (uniform에 가깝게). $T \to \infty$: uniform.

**KD Loss**:

$$L_{\text{KD}} = T^2 \cdot \text{KL}(p^{(T)}_{\text{teacher}} \| p^{(T)}_{\text{student}}) + \lambda \cdot L_{\text{CE}}(p^{(1)}_{\text{student}}, y)$$

- 첫 항: teacher의 soft output에 student을 fit.
- 둘째 항: ground truth에 fit (standard cross-entropy).
- $T^2$ scaling: gradient scale 맞춤.

### Dark Knowledge

Teacher의 soft output이 담는 "hidden 정보":

예: 이미지가 "3"이고 teacher output = $(0.95, 0.03, 0.01, 0.005, 0.001, \ldots)$.
- Hard label: "3" — 다른 class 무시.
- Teacher soft: "3이 주로 맞지만 **8과 약간 혼동**됨" ← dark knowledge.

Student가 이 soft target을 fit하면 "3과 8의 관계"를 학습 → 더 일반적인 feature.

### LS는 KD의 Extreme Case

Teacher가 "아무것도 모르는" extreme에서는 teacher output = **uniform** (모든 class에 $1/K$). 이 uniform을 soft target으로 쓰면:

$p^{(T)}_{\text{teacher}} = \text{Uniform}$

$L_{\text{KD}} = T^2 \cdot \text{KL}(\text{Uniform} \| p^{(T)}_{\text{student}}) + L_{\text{CE}}$

이는 **Label Smoothing과 수학적으로 동일** (정리 1.4의 LS = CE + KL to uniform).

즉 **LS = "no-knowledge teacher"로부터의 KD**.

### Muller 2019의 발견

**Teacher를 LS로 훈련하면 student KD 성능 저하**.

이유 (직관):
- LS teacher의 output은 "true class + uniform"에 가까움.
- Class 간 semantic similarity 정보 (dark knowledge)가 **wash out**.
- Student가 이 소실된 정보를 학습할 수 없음.

이는 "regularization 기법들의 상호작용"의 대표적 예.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Tempered Softmax

Logit $z$, temperature $T > 0$:

$$p^{(T)}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

$T = 1$: standard softmax. $T = \infty$: uniform. $T = 0^+$: one-hot at argmax.

### 정의 2.2 — KD Loss (Hinton 2015)

Teacher model $T$, Student model $S$. Both have logits $z^{T}, z^{S}$.

$$L_{\text{KD}}(z^S, z^T, y) = \alpha \cdot T^2 \cdot \text{KL}(p^{(T)}_{\text{teacher}} \| p^{(T)}_{\text{student}}) + (1-\alpha) \cdot L_{\text{CE}}(p^{(1)}_{\text{student}}, y)$$

Hyperparams: $T$ (temperature, 보통 2~10), $\alpha$ (soft vs hard target weight).

### 정리 2.3 — LS = KD with Uniform Teacher

Teacher가 constant uniform output $(1/K, \ldots, 1/K)$을 출력한다면:

$$L_{\text{KD}}(z^S, \text{uniform}, y) = \alpha \cdot T^2 \cdot \text{KL}(\text{Unif} \| p^{(T)}) + (1-\alpha) L_{\text{CE}}$$

이는 $T$-scaled LS (정리 1.4)와 정확히 동치.

### 정리 2.4 — KD's Gradient w.r.t. Student Logits

$$\frac{\partial L_{\text{KD}}}{\partial z^S_k} = \alpha T \cdot (p^{(T)}_{\text{student}, k} - p^{(T)}_{\text{teacher}, k}) + (1-\alpha)(p^{(1)}_{\text{student}, k} - y_k)$$

Teacher's soft target이 student logits를 "**teacher의 posterior**" 방향으로 끌어당김.

### 정리 2.5 — KD as Soft Label Learning (Urban 2017)

Hard label $y$ 없이 teacher soft target만 써도 student가 학습 가능. 즉 **KD alone** (labeled data 없이 teacher prediction만 사용)으로 student 훈련 가능. 이는 semi-supervised / self-training의 기반.

---

## 🔬 수학적 유도

### Temperature의 역할

$T$ 커질수록 $p^{(T)}$가 uniform에 가까워짐. Mathematically:

$p^{(T)}_k = \frac{e^{z_k/T}}{\sum e^{z_j/T}}$. $T \to \infty$면 모든 $z_j/T \to 0$, 모든 $e^{z_j/T} \to 1$, softmax → uniform.

**Dark knowledge가 amplified**: $z_k$ 간 작은 차이가 $T$로 나뉘면 softmax output이 더 "덜 선명" → class 간 ratio가 보존.

예: $z = (10, 9.8, 5)$, $T = 1$: $p \approx (0.55, 0.45, 0.001)$ (3rd class info 거의 소실).  
$T = 5$: $p \approx (0.34, 0.33, 0.33)$ — 3 classes가 더 비슷하게 보임.

### KD Loss의 Equivalent form

$L_{\text{KD}} = \alpha T^2 \cdot \text{KL}(p^T_T \| p^T_S) + (1-\alpha) L_{\text{CE}}$

$\text{KL}(p \| q) = \sum p_k \log(p_k / q_k) = -\sum p_k \log q_k + H(p)$

Teacher는 $w$-independent → $H(p^T_T)$ drop. $L_{\text{KL}}$ term = $-\sum_k p^T_{T,k} \log p^T_{S,k}$ + const = **cross-entropy with soft target**.

즉 **KD loss = cross-entropy with tempered teacher soft target + hard cross-entropy**.

### $T^2$ Scaling의 이유

Gradient $\partial p^{(T)}/\partial z \propto 1/T$. 따라서 $L_{\text{KD}}$의 gradient는 $T$에 따라 scaling.

$T^2$ factor는 gradient magnitude를 **$T=1$일 때와 비슷**하게 유지 → $T$를 바꿔도 training dynamics 안정.

---

## 💻 실험으로 효과 검증

### 실험 1 — KD Loss 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
    def forward(self, student_logits, teacher_logits, target):
        # Soft loss
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * self.T**2
        # Hard loss
        hard_loss = F.cross_entropy(student_logits, target)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# 사용
kd = KDLoss(T=4, alpha=0.7)
loss = kd(student(x), teacher(x).detach(), y)
```

### 실험 2 — Teacher 훈련: LS 유무에 따른 student KD 성능

```python
# Step 1: Teacher 훈련 (LS 유무 두 버전)
teacher_noLS = train_teacher(alpha_ls=0.0)   # no label smoothing
teacher_LS = train_teacher(alpha_ls=0.1)      # with label smoothing

# Step 2: Teacher1, Teacher2를 사용하여 Student 훈련
for teacher_name, teacher in [('noLS', teacher_noLS), ('LS', teacher_LS)]:
    student = SmallNet()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for epoch in range(50):
        for x, y in loader:
            opt.zero_grad()
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = kd_loss(s_logits, t_logits, y)
            loss.backward(); opt.step()
    
    test_acc = evaluate(student, test_loader)
    print(f"Student from {teacher_name} teacher: {test_acc:.4f}")

# 전형적 결과 (Muller 2019):
# Student from noLS teacher: 94.3%
# Student from LS teacher:   92.8%
# → 1.5% 저하 — dark knowledge 손실로 인한 degradation
```

### 실험 3 — Teacher softmax의 "class similarity matrix" 시각화

```python
# Teacher의 output을 similarity matrix로
# 각 pair (i, j)에서 teacher가 class i 이미지에 class j를 confuse하는 정도

def class_confusion_matrix(model, loader, num_classes=10):
    confusion = torch.zeros(num_classes, num_classes)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            probs = F.softmax(model(x), dim=-1)
            for c in range(num_classes):
                mask = (y == c)
                if mask.sum() > 0:
                    confusion[c] += probs[mask].sum(dim=0)
    return confusion / confusion.sum(dim=-1, keepdim=True)

conf_noLS = class_confusion_matrix(teacher_noLS, test_loader)
conf_LS = class_confusion_matrix(teacher_LS, test_loader)

# Diagonal을 제외한 off-diagonal variance 비교
# LS teacher: off-diagonal이 더 uniform (structure 소실)
# noLS teacher: off-diagonal에 class-pair specific structure
```

### 실험 4 — Temperature $T$ sweep

```python
Ts = [1, 2, 4, 8, 16]
for T in Ts:
    kd = KDLoss(T=T, alpha=0.7)
    student = SmallNet()
    # 훈련 ...
    acc = evaluate(student, test_loader)
    print(f"T={T}: student test acc = {acc:.4f}")
# 전형적 결과:
# T=1: 91.2% (insufficient soft info)
# T=2: 93.0%
# T=4: 93.8%  <-- sweet spot
# T=8: 93.5%
# T=16: 92.5% (too uniform, information diluted)
```

---

## 🔗 실전 활용

### KD의 주요 응용

1. **Model compression**: Large BERT → DistilBERT (Sanh 2019).
2. **Speech**: Large Conformer → on-device STT.
3. **Vision**: ResNet-152 → ResNet-18.
4. **Privacy-preserving learning**: Teacher soft label이 raw data 덜 공개.

### Teacher-Student 선택

- **Same architecture, different size**: 일반적 (e.g. ResNet-50 → ResNet-18).
- **Different architecture**: CNN teacher → Transformer student (feature matching 어려움).
- **Self-distillation**: student가 자기 자신의 soft output으로 re-train (Furlanello 2018).
- **Online distillation**: Teacher와 Student 동시 훈련 (DML, Zhang 2018).

### KD Variant

- **Feature-level distillation**: Logit뿐 아니라 intermediate activation도 match (Romero 2014 FitNets).
- **Relation-level**: Teacher의 representation 공간 structure를 student가 mimic.
- **Data-free KD**: Original data 없이 synthetic data로 (Chen 2019).

### LLM에서의 KD

- **Zephyr, Orca**: GPT-4 soft output으로 small model fine-tune.
- **Model distillation as a service**: Anthropic, OpenAI의 distillation pipeline.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Teacher가 student보다 good | "Dumb teacher effect": teacher 너무 큰 gap이면 student 따라잡기 어려움 |
| Soft target이 dark knowledge 담음 | LS teacher에서는 dark knowledge 손실 (Muller 2019) |
| Same domain | Teacher/student가 다른 domain이면 distillation 비효율 |
| Single teacher | Multi-teacher ensemble이 더 정확할 수 있음 |
| Logit-level matching | Feature-level matching이 representation quality에 중요할 수도 |

---

## 📌 핵심 정리

$$\boxed{L_{\text{KD}} = \alpha T^2 \text{KL}(p_T^{(T)} \| p_S^{(T)}) + (1-\alpha) L_{\text{CE}}}$$

| 개념 | 의미 |
|------|------|
| **Dark knowledge** | Teacher softmax의 class 간 유사도 정보 |
| **Temperature** | $T > 1$로 soft target 강화 |
| **LS = KD + uniform teacher** | 두 기법의 수학적 통일 |
| **Muller 2019** | LS teacher는 bad distillation teacher |
| **$T^2$ scaling** | Gradient magnitude 보존 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Teacher logit $z = (10, 8, 3, 1)$에 대해 $T = 1$과 $T = 5$에서의 softmax를 계산하라.

<details>
<summary>힌트 및 해설</summary>

**$T = 1$**: $\exp(z) = (e^{10}, e^8, e^3, e^1) \approx (22026, 2981, 20.1, 2.72)$. Normalize: $p \approx (0.88, 0.12, 0.001, 0.0001)$.

**$T = 5$**: $z/T = (2, 1.6, 0.6, 0.2)$. $\exp = (7.39, 4.95, 1.82, 1.22)$. Sum $\approx 15.4$. $p \approx (0.48, 0.32, 0.12, 0.08)$.

**관찰**: $T=1$은 거의 one-hot. $T=5$는 훨씬 "soft" — 다른 class도 의미 있는 확률. Student가 이를 통해 "class 2와 class 3이 class 4보다 class 1에 더 가깝다" 같은 구조 학습.

</details>

**문제 2** (심화): KD loss에서 $T \to 1$ 극한을 취하면 어떻게 되는가? $T \to \infty$ 극한은?

<details>
<summary>힌트 및 해설</summary>

**$T \to 1$**: Standard softmax. Teacher의 output은 one-hot에 가까움. KL term이 거의 hard cross-entropy와 같음 → KD가 **hard label training과 동등**. Dark knowledge 없음.

**$T \to \infty$**: $p^{(T)} \to$ uniform. KL $\to 0$ (both close to uniform). Soft loss가 0에 수렴 → student가 hard label만 학습. 역시 KD 효과 없음.

**Sweet spot $T \in [2, 8]$**: 다른 class의 relative probability를 의미 있게 전달 + gradient magnitude 안정.

**실전 tip**: $T$는 **task와 class 수**에 의존. Class 수 적으면 ($K = 10$) $T = 4$ 정도, 많으면 ($K = 1000$) $T = 8$ 이상 필요.

</details>

**문제 3** (이론-실전): Self-distillation ("student = teacher = same architecture, same data")에서 학생이 선생보다 더 좋아질 수 있는가? 현재 연구의 답은?

<details>
<summary>힌트 및 해설</summary>

**답**: **종종 가능** (Furlanello 2018 "Born Again Neural Networks").

**관찰**:
- Iteration 1: teacher (hard label 훈련) → student_1 (teacher soft) → 성능 약간 상승.
- Iteration 2: student_1 → student_2 → 다시 약간 상승.
- 3-5 iteration 후 plateau.

**왜 가능한가 (여러 설명)**:

1. **Implicit regularization**: Soft label이 LS 같은 regularization 역할. Student_1이 generalize.
2. **Dark knowledge amplification**: 여러 iteration으로 feature가 refined.
3. **"Knowledge ensemble" in logit space**: Teacher의 local minima에서 soft target이 smooth cover.

**한계**:
- Iterative improvement는 10-20 iteration 후 saturate.
- Original teacher 대비 1-2% 개선이 전형적 (dramatic 아님).
- 훈련 cost가 비례 증가.

**최근 연구** (Allen-Zhu & Li 2020): 이론적 설명 — "multi-view data"에서는 self-distillation이 diversification 통해 더 좋은 feature 학습.

**실전**: Self-distillation보다 ensemble이 대부분 더 effective. 하지만 inference cost가 문제면 self-distillation으로 single model 개선.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Label Smoothing](./01-label-smoothing.md) | [📚 README로 돌아가기](../README.md) | [03. Confidence Penalty · MaxEnt ▶](./03-confidence-penalty.md) |

</div>
