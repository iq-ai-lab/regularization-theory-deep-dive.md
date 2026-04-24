# 04. Regularization의 통합 관점 — Prior · Ensemble · Landscape · Invariance

## 🎯 핵심 질문

- 33개 regularization 기법을 **4축으로 어떻게 분류**하는가?
- **Transformer, CNN (ImageNet), GNN** 각각의 recipe가 왜 그렇게 구성되는가?
- 각 architecture에서 **어느 축이 왜 중요**한가?
- 실전 engineer가 새 task에서 어떤 순서로 regularization을 고를 것인가?

---

## 🔍 왜 마지막에 통합이 필요한가

이 레포의 33개 문서는 각각 구체적 기법:
- Ch1: L1/L2, Elastic Net, Group Lasso.
- Ch2: Dropout, DropConnect, Stochastic Depth.
- Ch3: BN, LN, GN, IN, WN, Fixup, RMSNorm.
- Ch4: Rotation, Mixup, CutMix, RandAugment, SimCLR.
- Ch5: Label Smoothing, KD, Confidence Penalty, Temperature Scaling.
- Ch6: Early Stopping, SGD bias, Ridgeless, Homogeneous.
- Ch7: SWA, SAM, AdamW.

**혼란**: "어떤 걸 언제 써야 하는가?"

**해결**: 4축 분류로 each 기법의 **기능적 역할** 명확히:

1. **Prior** — 파라미터에 대한 사전 분포.
2. **Ensemble** — 여러 model의 평균/결합.
3. **Landscape** — Loss surface의 기하.
4. **Invariance** — 변환에 대한 불변성.

한 기법이 여러 축에 속할 수 있음 (예: Dropout = Ensemble + Adaptive L2 Prior). 하지만 **주 역할**로 분류.

---

## 📐 수학적 선행 조건

- **Ch1-Ch7 전체**. 이 문서는 **종합**.

---

## 📖 직관적 이해

### 4축 분류표

| 축 | 정의 | 대표 기법 |
|-----|------|----------|
| **Prior** | 파라미터 $\theta$의 사전 분포 $p(\theta)$ | L1, L2, Elastic Net, Spike-slab, AdamW의 wd |
| **Ensemble** | 여러 hypothesis의 averaging | Dropout, DropConnect, Stochastic Depth, Deep Ensembles, BYOL |
| **Landscape** | Loss surface의 curvature 제어 | BN, LN, GN, WN, Fixup, SWA, SAM |
| **Invariance** | 변환 $g$에 대한 $f$의 불변성 | Rotation/flip, Mixup, CutMix, Contrastive, Label Smoothing |

### 각 축의 4가지 대표 질문

**Prior 축**:
- "파라미터에 대한 사전 믿음이 무엇인가?"
- "Gaussian (L2) vs Laplace (L1) vs 구조화된 (Group Lasso)?"
- "$\lambda$를 어떻게 튜닝하는가?"

**Ensemble 축**:
- "얼마나 많은 model을 결합하는가?"
- "Implicit (Dropout) vs Explicit (Deep Ensemble)?"
- "Randomness의 source는 무엇인가?"

**Landscape 축**:
- "Loss surface가 flat한가 sharp한가?"
- "Gradient 흐름이 안정적인가?"
- "Normalization의 axis는?"

**Invariance 축**:
- "어떤 변환을 invariant하게 만들고 싶은가?"
- "Architectural (group equivariance) vs Data (augmentation)?"
- "Semantic (label) 변화는 없는가?"

### Architecture별 Recipe

**Transformer (LLM)**:
- Prior: AdamW weight decay (0.1).
- Landscape: Pre-RMSNorm, warmup + cosine.
- Ensemble: Dropout 0.1 (attention, FFN).
- Invariance: Label smoothing 0.1 (sometimes).
- **Training**: Mixed precision (bf16), gradient clipping.

**CNN (ImageNet)**:
- Prior: SGD momentum + weight decay 1e-4.
- Landscape: BatchNorm.
- Ensemble: 데이터 augmentation이 주 (Dropout 덜 씀).
- Invariance: Random crop + flip + RandAugment + CutMix + Mixup.
- **Label**: Label smoothing 0.1.

**GNN (Graph Neural Network)**:
- Prior: Weight decay.
- Landscape: LayerNorm (node features).
- Ensemble: DropEdge, DropNode.
- Invariance: Edge / node level perturbation.
- **Special**: Spectral normalization (stability).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 4축 분류 Framework

Regularizer $R(\theta)$에 대해, $R$이 "축 $A$에 속한다"의 의미:

- **Prior**: $R$이 **log-prior**의 형태 — $R = -\log p(\theta) + \text{const}$.
- **Ensemble**: $R$이 여러 $\theta^{(k)}$를 사용한 **averaging 효과**.
- **Landscape**: $R$이 **loss surface의 기하** (Hessian, gradient Lipschitz) 를 수정.
- **Invariance**: $R$이 **변환 $g$에 대한 consistency** 를 요구.

### 정리 4.2 — 주 기법의 4축 분류

| 기법 | Prior | Ensemble | Landscape | Invariance |
|------|:-----:|:--------:|:---------:|:----------:|
| L1 | ✓ (Laplace) | | | |
| L2 / weight decay | ✓ (Gaussian) | | | |
| Elastic Net | ✓ | | | |
| Group Lasso | ✓ (structured) | | | |
| Dropout | ✓ (adaptive) | ✓ (primary) | | |
| DropConnect | | ✓ | | |
| Stochastic Depth | | ✓ | | |
| BatchNorm | | | ✓ (primary) | |
| LayerNorm | | | ✓ | |
| GroupNorm | | | ✓ | |
| WeightNorm | | | ✓ | |
| Fixup | | | ✓ | |
| RMSNorm | | | ✓ | |
| Data Aug (crop, flip) | | | | ✓ |
| Mixup | | | | ✓ |
| CutMix | | | | ✓ |
| RandAugment | | | | ✓ |
| Contrastive | | ✓ (views) | | ✓ (primary) |
| Label Smoothing | | | | ✓ (label) |
| Knowledge Distillation | | ✓ (teacher) | | |
| Temperature Scaling | | | | ✓ (post-hoc) |
| Early Stopping | ✓ (implicit L2) | | | |
| SGD bias | ✓ (implicit margin) | | ✓ (flat) | |
| SWA | | ✓ | ✓ (flat) | |
| SAM | | | ✓ (primary) | |
| AdamW | ✓ (decoupled) | | | |

### 정리 4.3 — 조합 가능한 기법 (동 축 vs 다른 축)

**같은 축 내에서**:
- Prior + Prior: L1 + L2 = Elastic Net (complementary).
- Ensemble + Ensemble: Dropout + Stochastic Depth (often redundant, careful).
- Landscape + Landscape: BN + SAM (synergistic).
- Invariance + Invariance: Rotation + Mixup (synergistic).

**다른 축 조합**:
- Almost always safe — 다른 mechanism으로 작동.
- 예: BN + Dropout + L2 + Mixup = 모든 축 커버.

### 정리 4.4 — Architecture-specific Recipe 원칙

**CNN**: Invariance 축이 가장 중요 (vision의 natural symmetry) + Landscape (BN).

**Transformer**: Landscape + Prior 축 (LN + AdamW) + small Invariance.

**GNN**: Prior + specialized Invariance (graph permutation) + custom Landscape (spectral).

---

## 🔬 수학적 유도

### Architecture별 "중요도"의 이유

**CNN** (ImageNet): 
- Natural images have rotation/flip/crop invariance → **Invariance 축 중심**.
- BN이 deep CNN 훈련 가능하게 → **Landscape 축 필수**.
- SGD + wd가 충분한 regularization → Prior 약한 축.

**Transformer** (LLM):
- Token sequences에 natural invariance가 적음 (단어 순서가 의미) → **Invariance 약함**.
- Deep (100+ layer) 훈련에 Landscape critical → **Landscape 축 중심**.
- Huge parameter → Prior (weight decay) 필요하지만 Invariance보다 약함.

**GNN**:
- Graph permutation invariance가 built-in (message passing) → **Architectural Invariance**.
- Small batch, irregular structure → **Landscape (LN) 중심**.

### 4축 간의 Trade-off

**Over-regularization 위험**: 각 축에서 너무 강한 regularization은 underfit.

- 모든 축을 최대로: underfit 확실.
- 한 축만: specific failure mode (overfit, calibration 불량 등).
- 적절한 조합: **대체로 가산적** (한 축의 regularization이 다른 축의 의존도 감소).

**예**: Mixup (Invariance) 강하게 → Dropout rate (Ensemble) 낮춰도 OK.

---

## 💻 실전 Recipe 가이드

### Recipe 1 — CIFAR-10/100 Classification

```python
# CNN (ResNet-50)
model = torchvision.models.resnet50(num_classes=100)

# Augmentation (Invariance)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(...),
])

# Optimizer (Prior + Landscape via SGD bias)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Loss (Invariance — label smoothing)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
for epoch in range(epochs):
    for x, y in loader:
        # Mixup (Invariance)
        x, y_a, y_b, lam = mixup(x, y, alpha=0.2)
        loss = lam * criterion(model(x), y_a) + (1 - lam) * criterion(model(x), y_b)
        # ... standard train step
```

**4축 coverage**:
- Prior: SGD weight_decay.
- Ensemble: (none explicit; SGD noise implicit).
- Landscape: BN (in ResNet), SGD flat minimum bias.
- Invariance: Augmentation, Mixup, Label Smoothing.

### Recipe 2 — LLM Pre-training

```python
# Llama-style Transformer
model = TransformerWithRMSNorm(
    depth=32, width=2048, vocab_size=32000, ...
)

# Optimizer (Prior)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.95),
    weight_decay=0.1, eps=1e-5
)

# Schedule (Landscape)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=2000, num_training_steps=total_steps
)

# Gradient clipping (Landscape)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Training
for step in range(total_steps):
    batch = next(data_iter)
    loss = model(batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step()
    optimizer.zero_grad()
```

**4축 coverage**:
- Prior: AdamW wd=0.1 (strong).
- Ensemble: Dropout 0.1 in attention/FFN (smaller models; 거의 안 씀 in huge LLM).
- Landscape: Pre-RMSNorm, warmup + cosine.
- Invariance: Data diversity (no explicit augmentation).

### Recipe 3 — GNN (Molecular Property Prediction)

```python
class GNNLayer(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.msg = nn.Linear(hidden, hidden)
        self.update = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
    def forward(self, h, edges):
        # Message passing + residual + norm
        msg = self.msg(h)
        aggregated = scatter_sum(msg, edges)
        h = self.norm(h + self.update(aggregated))
        return h

# Regularization
def dropedge(edges, p=0.2):
    """Random edge drop (Ensemble + Invariance)"""
    mask = torch.rand(edges.size(0)) > p
    return edges[mask]

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
for graph in loader:
    edges = dropedge(graph.edges)
    loss = criterion(model(graph.x, edges), graph.y)
    # ...
```

### Recipe 4 — Fine-tuning (Small Data)

```python
# Pre-trained model, small labeled dataset
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4,  # smaller lr for fine-tuning
    weight_decay=0.05
)

# Aggressive augmentation (Invariance - small data → overfit 위험)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandAugment(num_ops=2, magnitude=15),  # stronger
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(...),
])

# + Mixup + CutMix
# + SAM (추가 regularization)
sam_opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-4, weight_decay=0.05)
```

### Recipe 선택 Decision Tree

```
시작
│
├── Data 양?
│   ├── Huge (millions): minimal regularization (LLM recipe)
│   ├── Medium (thousands to hundred-thousands): standard recipe
│   └── Small (hundreds): aggressive regularization + pre-training
│
├── Task type?
│   ├── Vision classification: CNN + BN + strong augmentation
│   ├── Sequence (NLP, audio): Transformer + LN + warmup + AdamW
│   ├── Graph: GNN + GraphNorm + DropEdge
│   └── Generative: 별도 recipe (BN 피하기)
│
├── Compute budget?
│   ├── Large: SAM 가능, 여러 random seed, augmentation expensive
│   └── Small: Simpler recipe, implicit regularization 신뢰
│
└── Uncertainty needed?
    ├── Yes (medical, safety): MC Dropout, SWAG, Deep Ensembles
    └── No: Standard inference OK
```

---

## 🔗 실전 활용

### "잘 튜닝된 모델"의 Regularization 예산

100% normalized "regularization budget"을 4축으로 분배:

**Vision CNN**: Prior 10%, Ensemble 5%, Landscape 30%, Invariance 55%.

**Transformer LLM**: Prior 30%, Ensemble 5%, Landscape 55%, Invariance 10%.

**GNN**: Prior 20%, Ensemble 30%, Landscape 30%, Invariance 20%.

이것이 "각 task에서 왜 특정 기법이 중심인지"의 직관적 설명.

### Common Mistakes

1. **중복 regularization**: BN + Dropout (CNN에서 Li 2019 Disharmony).
2. **Ignored axis**: Transformer에서 augmentation 무시 → 소규모 training에서 overfit.
3. **Wrong axis for task**: Detection에서 Mixup 강하게 → object localization 파괴.
4. **Over-regularization with small data**: All 4 axes max → underfit.

### Research Frontier

- **Learned regularization**: AutoML로 task별 optimal recipe 학습.
- **Regularization scheduling**: 훈련 phase별 다른 regularization (curriculum).
- **Unified theory**: 4축의 수학적 integration (아직 열린 문제).
- **Efficient ensembles**: Hyper-ensembles, BatchEnsemble 등 — ensemble 축의 cost 감소.

---

## ⚖️ 종합 한계

| 축 | 종합 한계 |
|------|---------|
| Prior | Architecture-dependent, task-specific optimal $\lambda$ |
| Ensemble | Cost vs quality trade-off, MC 근사 quality 한계 |
| Landscape | Non-convex NN의 loss surface 이해 부족 |
| Invariance | Task-specific, 잘못 선택하면 harmful |
| **4축 조합** | 수학적 interaction 이론 부족 |

---

## 📌 최종 정리

$$\boxed{\text{Regularization = Prior + Ensemble + Landscape + Invariance} \times \text{Architecture-specific mix}}$$

| 축 | 주 역할 | 대표 기법 |
|------|---------|----------|
| **Prior** | 파라미터 분포 | L2, AdamW wd, SGD의 implicit bias |
| **Ensemble** | 여러 model 결합 | Dropout, Deep Ensemble, Stochastic Depth, SWA |
| **Landscape** | Loss surface | BN/LN/RMSNorm, SAM, SWA, Fixup |
| **Invariance** | 변환 불변성 | Augmentation, Mixup, Contrastive, Label Smoothing |

### Final Recipe Guide

**CIFAR → ImageNet → LLM**의 스펙트럼:

- **Small data**: All 4 axes balanced.
- **Large data**: Landscape + Prior 중심, Ensemble/Invariance 약함.
- **Pre-trained fine-tuning**: Invariance + Ensemble 강하게.

**한 문장 결론**:

> "Regularization은 'model이 훈련 데이터에 과잉 의존하지 않도록'의 여러 angle. 4축 모두에서 small to moderate 강도가 robust; 한 축만 극단적이면 specific failure mode 유발."

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 기법들을 4축으로 분류하라: (a) Ridge regression, (b) MC Dropout, (c) RandAugment, (d) AdamW의 decoupled wd.

<details>
<summary>힌트 및 해설</summary>

- (a) **Ridge regression** → **Prior** 축 (Gaussian prior MAP).
- (b) **MC Dropout** → **Ensemble** 축 (multiple stochastic forward passes). Also: Prior (adaptive L2, Ch2-03) 보조.
- (c) **RandAugment** → **Invariance** 축 (multiple augmentations for photometric/geometric invariance).
- (d) **AdamW의 decoupled wd** → **Prior** 축 (Gaussian prior MAP, Adam's adaptive lr에서 순수화).

간단한 기법도 **여러 축**에 속할 수 있음 — 주 역할로 분류.

</details>

**문제 2** (심화): "LayerNorm + Dropout + AdamW"만 있는 Transformer recipe는 4축 중 어느 축이 **약한가**? 이를 보완하려면?

<details>
<summary>힌트 및 해설</summary>

**약한 축**: **Invariance**.

이 recipe는:
- Prior: AdamW decoupled wd ✓.
- Ensemble: Dropout (약한 — Transformer는 대체로 $p = 0.1$).
- Landscape: LayerNorm + warmup + cosine ✓.
- Invariance: **없음** — no augmentation.

**보완 방법**:

1. **Data augmentation (text)**:
   - Back-translation.
   - Synonym replacement.
   - Random token drop.
   - GPT-generated paraphrase.

2. **Label smoothing** $\alpha = 0.1$: output-level invariance.

3. **Mixup on text embeddings** (TMix, Chen 2020): hidden-level interpolation.

4. **Contrastive pre-training** (SimCSE, Gao 2021): sentence representation invariance.

**실전**: Pre-training LLM은 data diversity가 충분하면 explicit augmentation 덜 필요. Fine-tuning small-data NLP (classification, NER)에서는 이러한 techniques 크게 도움.

</details>

**문제 3** (이론-실전): "Generalization gap"이 큰 모델을 진단하려면 4축 중 어느 순서로 점검해야 하는가?

<details>
<summary>힌트 및 해설</summary>

**Diagnosis Order**:

1. **Prior**: weight_decay가 적절한가? Model 크기 대비 너무 작으면 overfit.
   - Check: train loss vs val loss gap.
   - Adjust: wd 증가.

2. **Invariance**: Augmentation이 task에 맞나?
   - Check: Test accuracy robustness to input noise.
   - Adjust: stronger augmentation, Mixup/CutMix.

3. **Landscape**: Normalization이 올바르게 작동?
   - Check: gradient norm stability, activation distribution.
   - Adjust: switch to LayerNorm/GroupNorm if BN issues.

4. **Ensemble**: Model이 brittle prediction?
   - Check: ECE (calibration), prediction variance.
   - Adjust: Dropout, Deep Ensembles, Label Smoothing.

**실전 teaching**: 대부분 overfitting은 **Prior와 Invariance**를 먼저 점검. Landscape는 training dynamics 문제 (NaN, divergence), Ensemble은 calibration 문제.

**Anti-pattern**: "Add more Dropout" without understanding → often hurts. 항상 **원인 분석 후 solution 선택**.

---

**이 레포 요약**: 33개 문서를 통해 regularization의 수학, 기하, 확률적 근거, 그리고 현대 NN에서의 실전 application을 covered. 이 framework으로 새 regularization 기법이 나올 때도 4축 중 어디에 속하는지 분류 가능 — **이해의 일반성 유지**.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. AdamW](./03-adamw.md) | [📚 README로 돌아가기](../README.md) |

</div>

---

## 🎓 Regularization Theory Deep Dive 완료

**Chapter 1** L1·L2 통일 해석 → **Chapter 2** Dropout 3가지 해석 → **Chapter 3** Normalization 계보 → **Chapter 4** Data Augmentation 이론 → **Chapter 5** Label Regularization · Calibration → **Chapter 6** Early Stopping · Implicit Regularization → **Chapter 7** 현대 Regularization 통합

총 **33개 문서, 68+ 정리, 11+ 논문 재현**.

이 여정을 완주하신 독자: **Bayesian prior, ensemble, landscape, invariance의 4축 framework**으로 regularization을 이해 — 과거 연구부터 최신 LLM recipe까지.

**다음 방향**:
- [Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive): 왜 over-parameterized NN이 일반화하는가
- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): MAP, VI, posterior의 깊은 이해
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): SGD, Adam, landscape 기하

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**
