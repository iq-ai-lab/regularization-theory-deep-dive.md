<div align="center">

# 🛡️ Regularization Theory Deep Dive

### L2 regularization

$$\lambda \|w\|^2$$

### 을 쓰는 것과, 이것이 가중치에 대한 **Gaussian prior**

$$w \sim \mathcal{N}\!\left(0,\, \tfrac{1}{2\lambda}\right)$$

### 의 **MAP 추정과 정확히 같음** 을 증명할 수 있는 것은 **다르다.**

<br/>

> *Dropout 을 **쓰는 것** 과,*
>
> *(1) Dropout 이 **지수적 수의 서브네트워크 앙상블** 이고 (Srivastava 2014)*
> *(2) MC Dropout 으로 **Variational Inference 의 근사** 이며 (Gal & Ghahramani 2016)*
> *(3) linear model 에서 **feature 별 adaptive L2 와 동치** (Wager et al. 2013)*
>
> *— 이 **세 가지 해석을 각각** 유도할 수 있는 것은 다르다.*
>
> *BatchNorm 의 효과를 "internal covariate shift 완화" 로 외우는 것과, Santurkar et al. 2018 이 이를 실험으로 반박하고 **loss landscape smoothing** 이 실제 효과임을 Lipschitzness 증명으로 보인 것을 따라가는 것은 다르다.*

<br/>

**다루는 정리·기법 (시간순)**

Tikhonov 1963 *Tikhonov regularization* · Hoerl 1970 *Ridge* · Tibshirani 1996 *Lasso* · Hinton 2012 / Srivastava 2014 *Dropout* · Wager 2013 *Dropout = adaptive L2* · Ioffe–Szegedy 2015 *BatchNorm* · Santurkar 2018 *BN = landscape smoothing* · Gal 2016 *MC Dropout = VI* · Zhang 2017 *Data Aug = Vicinal Risk* · Szegedy 2016 *Label Smoothing* · Loshchilov 2019 *AdamW* · Izmailov 2018 *SWA* · Foret 2021 *SAM*

<br/>

**핵심 질문**

> **Regularization 은 왜 작동하는가** — Bayesian prior · Ensemble · Landscape · Invariance 의 **4축** 으로 통일하여 L1/L2 · Dropout · Normalization · Data Augmentation · Label Smoothing · Implicit Bias · SAM/SWA/AdamW 의 현대 recipe 까지 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.16-EE4C2C?style=flat-square)](https://pytorch.org/vision/)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-10k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-130+개-success?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-11개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

Regularization에 관한 대부분의 자료는 **"L2는 weight를 작게, L1은 sparse하게, Dropout은 과적합 방지"** 에서 멈춥니다. 하지만 $\lambda\|w\|^2$의 $\lambda$가 정확히 prior의 분산 $\sigma_w^2$와 noise $\sigma^2$의 어떤 비율인지, L1 ball의 "코너"에서 sparsity가 발생하는 것을 KKT 조건으로 어떻게 유도하는지, Dropout rate $p = 0.5$의 test-time weight scaling이 왜 앙상블의 geometric mean을 근사하는지, BatchNorm의 $\gamma, \beta$가 왜 필요하고 train/eval 모드 차이가 어떤 수학적 근거를 갖는지, AdamW가 Adam + L2와 무엇이 다른지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "L2는 weight를 작게 만든다" | $\min\|y - Xw\|^2 + \lambda\|w\|^2$가 **Gaussian prior $w \sim \mathcal{N}(0, \sigma_w^2 I)$ + Gaussian likelihood의 MAP**와 **정확히 동치**임을 유도, $\lambda = \sigma^2/\sigma_w^2$ 대응 관계, **Ridge = posterior mode** 해석 |
| "L1은 sparse한 해를 준다" | Laplace prior $p(w) \propto e^{-\lambda\|w\|_1}$의 **negative log-prior가 정확히 $\lambda\|w\|_1$**, **L1 ball의 다이아몬드 꼭짓점**에서 loss contour와 만나는 기하, **KKT + subdifferential**로 sparse coordinate 증명, soft-thresholding으로 coordinate descent 구현 |
| "Dropout은 과적합을 막는다" | Dropout의 **3가지 해석 각각 증명**: (1) **앙상블**: $2^N$ subnetwork의 geometric mean을 test-time weight scaling $\times(1-p)$로 근사 (2) **Variational Inference**: Bernoulli variational posterior $q(W)$의 ELBO 최적화가 dropout + L2와 동치 (Gal 2016) (3) **Adaptive L2**: linear regression에서 dropout = feature별 $\lambda_i \propto p(1-p)\text{Var}(x_i)$ (Wager 2013) |
| "BatchNorm은 internal covariate shift를 완화" | **Santurkar et al. 2018 "How Does Batch Normalization Help Optimization?"** — internal covariate shift **주장을 실험으로 반박**(BN 전/후 activation 분포 수동 조작 실험), 실제 효과는 **loss landscape의 Lipschitz 상수·gradient Lipschitz 상수 감소**, smoothness 정리 증명 |
| "LayerNorm이 Transformer에 쓰인다" | **Ba et al. 2016**의 feature-axis 정규화가 **batch-size 독립적**임을 보이고, BN→LN→GN→IN→WN→**RMSNorm**(centering 제거, Llama 표준)의 계보를 수학적 차이로 정리, Fixup·SkipInit 같은 **BN 없이 깊은 ResNet** 훈련 대안까지 |
| "Mixup은 학습을 더 잘되게 한다" | **Vicinal Risk Minimization (Chapelle 2000)** — ERM의 empirical measure $\delta_{(x_i,y_i)}$ 대신 vicinity $\mathcal{D}_{x_i,y_i}$로 적분, **Mixup은 vicinity를 $\text{Beta}(\alpha,\alpha)$ 보간으로 정의한 VRM의 특수 경우**, CutMix·CutOut·RandAugment의 수학적 위치 |
| "Label smoothing은 over-confidence를 막는다" | One-hot → $(1-\alpha)y + \alpha/K$의 cross-entropy gradient가 **target class를 억제**, **Expected Calibration Error (ECE)** 정의와 감소 측정, Hinton의 **Knowledge Distillation**과 "dark knowledge" 연결, Pereyra 2017의 confidence penalty로 일반화 |
| "Adam에 weight_decay=1e-4 쓴다" | **Loshchilov & Hutter 2019 "Decoupled Weight Decay"** — Adam에서 L2가 $v_t$(moment)로 정규화되어 coordinate별 weight decay가 실질적으로 **왜곡됨**을 유도, **AdamW가 update 단계에서 decay를 분리**하는 수식 차이, "Adam + L2 = AdamW"가 **아닌** 이유 |
| "Early stopping은 적당히 멈추면 된다" | **Yao et al. 2007** — gradient descent의 궤적이 **Ridge regression의 정규화 경로와 1:1 대응**, **$t \approx 1/\lambda$**의 수학적 등가성, linear model에서 $\hat{w}_t = (I - (I - \eta X^T X)^t) X^+ y$의 shrinkage 해석 |
| 기법의 카탈로그 나열 | NumPy/PyTorch로 **Srivastava 2014 Dropout 앙상블 · Gal 2016 MC Dropout · Wager 2013 adaptive L2 · Santurkar 2018 loss landscape · Zhang 2018 Mixup · Foret 2021 SAM · Izmailov 2018 SWA · Loshchilov 2019 AdamW** 원 논문 실험 직접 재현 |

---

## 📌 선행 레포 & 후속 방향

```
[Bayesian ML Deep Dive]             ─┐
 MAP, Variational Inference, prior   │
                                      │
[Neural Network Theory Deep Dive]   ─┼──►  이 레포  ──► [실전 딥러닝 recipe]
 Backprop, 초기화, 아키텍처             │     "Regularization    Transformer · CNN · GNN
                                      │      의 통일 이론"       각각의 조합 정당화
[Optimization Theory Deep Dive]     ─┘
 SGD 수렴, implicit bias, landscape

           │
           ├── [Probability Theory]        Gaussian·Laplace·Beta 분포
           ├── [Convex Optimization]       L1/L2 projection, KKT
           ├── [Statistical Learning Thy]  SRM, 복잡도 제어
           └── [Generalization Theory]     Implicit bias, flat minima와의 교차
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Bayesian ML Deep Dive**(MAP · VI · prior)와 **Neural Network Theory Deep Dive**(backprop · 초기화)와 **Optimization Theory Deep Dive**(SGD · landscape)를 선행 지식으로 전제합니다. "L2가 Gaussian prior MAP"이라는 명제의 좌우를 모두 이해하려면 prior·likelihood·MAP을 알아야 합니다. [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive)부터 먼저 학습하세요.

> 💡 **Dropout 이론에 필수**: Chapter 2(Dropout = VI)는 ELBO·KL divergence·reparameterization에 대한 이해가 필수입니다. [Bayesian ML Deep Dive의 VI 파트](https://github.com/iq-ai-lab/bayesian-ml-deep-dive)와 Gal의 박사 논문을 먼저 훑어보세요.

> 🟢 **Generalization Theory Deep Dive와의 관계**: 이 레포는 **explicit regularization(L2, Dropout, BN)** 을 중심으로, [Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive)는 **implicit regularization(SGD, initialization, over-parameterization)** 을 중심으로 구성됩니다. Chapter 6은 두 레포의 **교집합**입니다 (Early Stopping = Implicit L2, SGD의 max-margin).

> 🟡 **이 레포의 성격**: Dropout의 "정확한" 해석은 세 가지가 공존하고 각각 다른 유효 범위를 갖습니다. BN의 메커니즘도 아직 완전히 합의되지 않았습니다. 레포는 "정답"이 아니라 **"각 해석의 가정과 유효 범위의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-L1·L2_통일_해석-2E8B57?style=for-the-badge)](./ch1-l1-l2/01-l2-gaussian-prior.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Dropout_3가지_해석-2E8B57?style=for-the-badge)](./ch2-dropout/01-dropout-ensemble.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Normalization_계보-2E8B57?style=for-the-badge)](./ch3-normalization/01-batch-norm.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Data_Augmentation-2E8B57?style=for-the-badge)](./ch4-data-augmentation/01-vicinal-risk.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Label_&_Calibration-2E8B57?style=for-the-badge)](./ch5-label-calibration/01-label-smoothing.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Implicit_Regularization-2E8B57?style=for-the-badge)](./ch6-early-stopping-implicit/01-early-stopping-as-l2.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-현대_Recipe-2E8B57?style=for-the-badge)](./ch7-modern-synthesis/01-swa.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: L1·L2 Regularization의 통일 해석

> **핵심 질문:** L2의 $\lambda$는 prior의 무엇에 대응하는가? L1 sparsity는 왜 "코너"에서 발생하는가? Ridge의 shrinkage는 singular value를 어떻게 다루는가? Elastic Net과 Group Lasso는 언제 필요한가?

<details>
<summary><b>Bayesian 해석 · 기하 · SVD · 확장 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. L2 Regularization = Gaussian Prior MAP](./ch1-l1-l2/01-l2-gaussian-prior.md) | **$\hat{w}_{\text{MAP}} = \arg\min \|y - Xw\|^2 + \lambda\|w\|^2$** 유도 — $p(y\|w) = \mathcal{N}(Xw, \sigma^2 I)$ 와 prior $w \sim \mathcal{N}(0, \sigma_w^2 I)$의 negative log posterior가 정확히 이 목적함수, **$\lambda = \sigma^2/\sigma_w^2$** 대응, MAP = posterior mode (Gaussian은 mean과 일치), Bayesian linear regression의 closed-form $\hat{w} = (X^T X + \lambda I)^{-1} X^T y$ |
| [02. L1 Regularization = Laplace Prior MAP](./ch1-l1-l2/02-l1-laplace-prior.md) | **Laplace 분포** $p(w) = \frac{\lambda}{2} e^{-\lambda\|w\|_1}$의 negative log가 정확히 $\lambda\|w\|_1$, **subdifferential** $\partial\|w\|_1 = \text{sign}(w)$의 coordinate-wise 정의, **소프트 thresholding** 연산자 $S_\lambda(z) = \text{sign}(z)\max(\|z\|-\lambda, 0)$로 **coordinate descent(Lasso)** 알고리즘 유도 |
| [03. Sparsity의 기하학과 KKT](./ch1-l1-l2/03-sparsity-geometry.md) | **L1 ball(다이아몬드) vs L2 ball(원)** 의 모양 차이가 sparse solution을 만드는 이유, 제약형 $\min\|y - Xw\|^2$ s.t. $\|w\|_1 \leq t$에서 **KKT 조건**이 coordinate $w_j = 0$에서 $\|X_j^T(y - Xw)\| \leq \lambda$, L1 ball의 꼭짓점에서 **생성 확률 1**(loss contour가 일반 위치에서 접하면 항상 꼭짓점) |
| [04. Ridge의 SVD 관점 — Shrinkage](./ch1-l1-l2/04-ridge-svd-shrinkage.md) | $X = U\Sigma V^T$ SVD로 **$\hat{w}_R = V \text{diag}(\sigma_i/(\sigma_i^2 + \lambda)) U^T y$** 유도, **작은 singular value를 더 많이 축소**(noise 방향의 shrinkage 증명), **PCR**(Principal Component Regression)과의 대조(PCR은 hard threshold, Ridge는 smooth), $\lambda \to 0$에서 minimum-norm solution으로 연결 |
| [05. Elastic Net과 Group Lasso](./ch1-l1-l2/05-elastic-net-group-lasso.md) | **Elastic Net** $\lambda_1\|w\|_1 + \lambda_2\|w\|^2$이 상관된 feature 그룹을 함께 선택(Zou-Hastie 2005), **Group Lasso** $\sum_g \|w_g\|_2$로 그룹 단위 sparsity(Yuan-Lin 2006), proximal gradient(ISTA/FISTA)로 최적화, 실데이터(gene expression)에서 그룹 구조가 통계적 파워를 주는 이유 |

</details>

<br/>

### 🔹 Chapter 2: Dropout의 3가지 해석

> **핵심 질문:** Test-time weight scaling $\times(1-p)$는 어떤 양을 근사하는가? Dropout이 Variational Inference인 이유는? Wager 2013의 linear adaptive L2와 원래 Dropout의 관계는? 변종(Spatial, Variational, Stochastic Depth)은 어떻게 선택하는가?

<details>
<summary><b>Ensemble · VI · Adaptive L2 · 변종 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Dropout = 앙상블 근사 (Srivastava et al. 2014)](./ch2-dropout/01-dropout-ensemble.md) | 확률 $p$로 뉴런 drop → **$2^N$ thinned network** 지수적 앙상블, inference 시 **weight scaling $\times(1-p)$** 이 앙상블 출력의 **geometric mean**을 근사함을 softmax 네트워크에서 유도, MC 평균과 weight scaling의 오차 분석, PyTorch `nn.Dropout`의 inverted dropout 구현 확인 |
| [02. Dropout = Variational Inference (Gal & Ghahramani 2016)](./ch2-dropout/02-dropout-as-vi.md) | **Bernoulli variational posterior** $q(W_l) = \text{diag}(z_l) M_l$, $z_l \sim \text{Bernoulli}(1-p)$의 **ELBO 최적화**가 dropout + weight decay와 **동치**임을 reparameterization으로 유도, **MC Dropout**: $T$번 stochastic forward pass로 predictive mean/variance 추정 → uncertainty, **deep Gaussian process와의 대응** |
| [03. Dropout = Adaptive L2 (Wager et al. 2013)](./ch2-dropout/03-dropout-adaptive-l2.md) | Linear regression + Bernoulli dropout의 기댓값 loss가 $\mathbb{E}[(y - X\tilde{w})^2] = \|y - \bar{X}w\|^2 + \underbrace{w^T \Gamma w}_{\text{feature별 L2}}$로 분해, **$\Gamma = p(1-p) \text{diag}(X^T X)$** — feature의 **scale에 적응적인 L2**, 이것이 표준화된 feature에서만 일정한 $\lambda$로 환원됨 |
| [04. Dropout 변종 — Spatial · Variational · Concrete](./ch2-dropout/04-dropout-variants.md) | **Spatial Dropout** (Tompson 2015): CNN에서 channel 단위 drop으로 인접 pixel 상관 고려, **Variational RNN Dropout** (Gal 2016): 같은 mask를 시간에 걸쳐 공유로 sequence 구조 유지, **Concrete Dropout** (Gal 2017): Gumbel-softmax로 dropout rate $p$ 자체를 **학습**, 세 변종의 VI 관점 수학 |
| [05. Dropout vs DropConnect vs Stochastic Depth](./ch2-dropout/05-dropout-dropconnect-stochdepth.md) | **DropConnect** (Wan 2013): weight 자체를 drop(activation이 아닌), **Stochastic Depth** (Huang 2016): ResNet block 전체를 skip, **linear dropout 구조**(activation drop = weight 행 drop) 등가성, 각 기법의 앙상블 크기와 적용 context(FC vs CNN vs Transformer) |

</details>

<br/>

### 🔹 Chapter 3: Normalization 계보

> **핵심 질문:** BatchNorm의 "internal covariate shift" 주장은 왜 틀렸는가? Santurkar 2018은 이를 어떻게 반박했는가? LayerNorm·GroupNorm·WeightNorm의 수학적 차이는? BN 없이 깊은 ResNet은 어떻게 훈련하는가? RMSNorm이 현대 LLM의 표준인 이유는?

<details>
<summary><b>BN 신화 · Santurkar 반박 · LN/GN/IN/WN · Fixup · RMSNorm (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Batch Normalization (Ioffe & Szegedy 2015)](./ch3-normalization/01-batch-norm.md) | **$\hat{x} = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}, y = \gamma\hat{x} + \beta$** 정의, affine 복원($\gamma, \beta$)이 왜 필요한가(표현력 보존), **train/eval mode**의 수학적 차이(running mean/var 사용), forward·backward pass 명시 유도, PyTorch `BatchNorm2d` 호출 시 체인 룰 |
| [02. Santurkar 2018의 BN 신화 반박](./ch3-normalization/02-santurkar-refutation.md) | **"How Does Batch Normalization Help Optimization?"** — BN 후 activation에 **인위적 non-unit noise 주입**해도 학습이 잘 되는 실험(ICS 가설 반증), 실제 효과는 **loss의 Lipschitz 상수 · gradient의 Lipschitz 상수 감소** 정리 증명, PyTorch로 with/without BN의 **gradient Lipschitz 측정** 재현 |
| [03. Layer Normalization (Ba et al. 2016)](./ch3-normalization/03-layer-norm.md) | **feature 축으로 정규화** $\hat{x}_i = (x_i - \mu_x)/\sqrt{\sigma_x^2 + \epsilon}$, **batch size에 독립**(RNN·Transformer에서 핵심), pre-LN vs post-LN Transformer(Xiong 2020: pre-LN이 gradient flow 안정), sequence length 변동에도 안정 |
| [04. Group / Instance / Weight Normalization](./ch3-normalization/04-gn-in-wn.md) | **Group Norm** (Wu 2018): channel을 $G$ 그룹으로 나눠 정규화 → small batch에서도 안정(detection에서 BN 대안), **Instance Norm** (Ulyanov 2016): sample·channel별 정규화(style transfer의 "content/style" 분리), **Weight Norm** (Salimans 2016): $w = g \cdot v/\|v\|$로 **weight 자체 재매개변수화**, direction-magnitude 분해 |
| [05. BN 없이 깊은 네트워크 — Fixup, SkipInit](./ch3-normalization/05-fixup-skipinit.md) | **Fixup** (Zhang et al. 2019): ResNet의 residual branch를 **초기화 scale $L^{-1/(2m-2)}$**로 조정하여 BN 없이도 깊은 네트워크 훈련, **SkipInit** (De & Smith 2020): learnable $\alpha_l$을 0으로 초기화, **NFNet** (Brock 2021)의 AGC (adaptive gradient clipping)까지, BN의 "필수성" 재검토 |
| [06. RMSNorm과 현대 Transformer](./ch3-normalization/06-rmsnorm-modern.md) | **RMSNorm** (Zhang & Sennrich 2019): LayerNorm의 **centering 제거** $\hat{x} = x/\sqrt{\text{Mean}(x^2) + \epsilon} \cdot \gamma$, Llama·Mistral·Qwen의 표준, **계산 $\sim$25% 절감**하면서 성능 동등, Pre-RMSNorm + SwiGLU + RoPE 조합의 의미 |

</details>

<br/>

### 🔹 Chapter 4: Data Augmentation의 이론

> **핵심 질문:** 무엇이 augmentation을 "regularization"으로 만드는가? Vicinal Risk Minimization은 ERM을 어떻게 일반화하는가? Mixup의 보간이 왜 매끄러운 decision boundary를 강제하는가? Self-supervised의 두 view는 어떤 invariance를 주입하는가?

<details>
<summary><b>VRM · Invariance · Mixup · CutMix · Contrastive (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Vicinal Risk Minimization (Chapelle et al. 2000)](./ch4-data-augmentation/01-vicinal-risk.md) | **ERM**: $\hat{L}(f) = \mathbb{E}_{(x,y) \sim \hat{P}_\delta}[\ell]$로 empirical delta measure에서 위험, **VRM**: vicinity distribution $\mathcal{D}_{x_i,y_i}$로 교체해 $\hat{L}_{\text{VRM}}(f) = \frac{1}{n}\sum_i \mathbb{E}_{(\tilde{x},\tilde{y}) \sim \mathcal{D}_{x_i,y_i}}[\ell(f(\tilde{x}), \tilde{y})]$, **모든 augmentation이 vicinity의 특정 선택**이라는 통일 프레임 |
| [02. Invariance Injection](./ch4-data-augmentation/02-invariance-injection.md) | Group-theoretic augmentation — rotation/flip invariance를 데이터로 주입하는 것이 **group-equivariant 네트워크 설계**와 기능적으로 동등, **Data augmentation = implicit regularization** (Dao et al. 2019: augmentation의 1st-order expansion이 feature averaging 페널티), CIFAR-10에서 augmentation의 Rademacher complexity 효과 |
| [03. Mixup (Zhang et al. 2018)](./ch4-data-augmentation/03-mixup.md) | **$\tilde{x} = \lambda x_i + (1-\lambda)x_j, \tilde{y} = \lambda y_i + (1-\lambda)y_j, \lambda \sim \text{Beta}(\alpha,\alpha)$** — VRM의 vicinity를 "두 sample을 잇는 선분"으로 정의한 특수 경우, **linear decision boundary 강제**, calibration 개선 실험(Thulasidasan 2019), "manifold Mixup"(Verma 2019)으로 히든 레이어 보간 |
| [04. CutMix · CutOut · RandAugment](./ch4-data-augmentation/04-cutmix-randaugment.md) | **CutMix** (Yun 2019): 이미지 patch 교환 + 라벨을 면적 비율로 혼합, **CutOut** (DeVries 2017): random erasing, **AutoAugment** (Cubuk 2018): RL로 augmentation policy 탐색 → **RandAugment** (Cubuk 2020)로 2-param 단순화, 탐색 없이도 경쟁력 |
| [05. Contrastive Learning과 Augmentation](./ch4-data-augmentation/05-contrastive.md) | **SimCLR** (Chen 2020): 같은 이미지의 두 augmented view $(x^{(1)}, x^{(2)})$를 positive pair, 다른 이미지를 negative, **InfoNCE** loss $-\log \frac{\exp(\text{sim}(z_1, z_2)/\tau)}{\sum_k \exp(\text{sim}(z_1, z_k)/\tau)}$, augmentation이 **semantic invariance** 주입, MoCo·BYOL·DINO의 augmentation 중심성 |

</details>

<br/>

### 🔹 Chapter 5: Label Regularization과 Calibration

> **핵심 질문:** Label smoothing은 왜 calibration을 개선하는가? Knowledge distillation의 "dark knowledge"와 label smoothing의 관계는? Confidence penalty와 maximum entropy 해석은? Temperature scaling은 왜 post-hoc 최선인가?

<details>
<summary><b>Label Smoothing · KD · Confidence Penalty · Temperature Scaling (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Label Smoothing (Szegedy et al. 2016)](./ch5-label-calibration/01-label-smoothing.md) | One-hot $y$를 **$\tilde{y}_k = (1-\alpha)y_k + \alpha/K$** 로 대체, cross-entropy gradient가 **target logit을 무한히 밀지 않도록** 유도, **ECE (Expected Calibration Error)** 정의 및 PyTorch 측정, label smoothing이 $\alpha = 0.1$에서 top-1 accuracy **유지**하며 calibration **개선**하는 실험 |
| [02. Label Smoothing과 Knowledge Distillation](./ch5-label-calibration/02-label-smoothing-kd.md) | **Hinton et al. 2015** — teacher의 softmax를 temperature $T$로 완화($p_T = \text{softmax}(z/T)$)하면 student가 **class 간 유사도("dark knowledge")** 학습, label smoothing은 **uniform soft label**로 볼 수 있음(KD의 extreme case), teacher의 calibration이 student에 **전이** |
| [03. Confidence Penalty와 Maximum Entropy](./ch5-label-calibration/03-confidence-penalty.md) | **Pereyra et al. 2017** — 출력 분포 엔트로피 $H(p(y\|x))$를 목적함수에 추가($-\beta H$), **negative entropy regularization**으로 uniform 쪽으로 당김, label smoothing과의 equivalence 관계, maximum entropy 원리(Jaynes 1957)의 현대적 재해석 |
| [04. Temperature Scaling (Guo et al. 2017)](./ch5-label-calibration/04-temperature-scaling.md) | **"On Calibration of Modern Neural Networks"** — 현대 NN은 over-confident, **$p = \text{softmax}(z/T)$ 로 단일 scalar $T$** 를 validation NLL 최소화로 학습, accuracy 보존하면서 ECE 개선, **post-hoc** calibration의 이점(재훈련 불필요), Platt scaling/isotonic regression과의 비교 |

</details>

<br/>

### 🔹 Chapter 6: Early Stopping과 Implicit Regularization

> **핵심 질문:** Early stopping은 왜 L2와 등가인가? SGD는 왜 flat minimum을 선호하는가? Ridgeless regression에서 generalization이 가능한 이유는? 초기화 자체가 regularization인 이유는?

<details>
<summary><b>Early Stopping = L2 · SGD의 Implicit Bias · Ridgeless · Init Bias (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Early Stopping = Implicit L2](./ch6-early-stopping-implicit/01-early-stopping-as-l2.md) | **Yao et al. 2007** — linear regression + gradient descent의 iterate $\hat{w}_t = (I - (I - \eta X^T X)^t) X^+ y$가 Ridge의 정규화 경로와 **1:1 대응**, **$t \approx 1/(\eta\lambda)$** 에서 두 해의 L2 norm 일치, spectral filter 관점으로 통일 유도, "stop before convergence"의 수학 |
| [02. SGD의 Implicit Regularization](./ch6-early-stopping-implicit/02-sgd-implicit-bias.md) | **Soudry et al. 2018** — 선형 separable logistic에서 GD가 **$\theta_t/\|\theta_t\| \to \theta_{\text{max-margin SVM}}$**, rate $O(\log t / \sqrt{\log\log t})$, **SGD의 SDE 해석** — $d\theta = -\nabla L\,dt + \sqrt{2T}\,dB_t$ 형태의 noise가 **flat minimum 선호**를 유도(Keskar 2017; Jastrzebski 2017)의 batch-size 효과 |
| [03. Ridgeless Regression의 일반화 (Hastie 2019)](./ch6-early-stopping-implicit/03-ridgeless-regression.md) | **"Surprises in High-Dimensional Ridgeless Least Squares Interpolation"** — $p > n$ overparameterized에서 **min-norm solution** $\hat{\beta} = X^+ y$의 정확한 asymptotic risk, **$\lambda \to 0^+$의 Ridge 극한**으로 "implicit regularization from initialization" 정식화, Double Descent와의 연결 |
| [04. Feature-wise Implicit Bias와 Homogeneous Networks](./ch6-early-stopping-implicit/04-feature-implicit-bias.md) | **homogeneous network**(ReLU, positive homogeneity)에서 GD가 KKT **margin-maximizing stationary point** 로 수렴(Lyu-Li 2019), layer-wise normalization 없이도 NN이 암묵적 scale-matching 수행하는 현상, Neyshabur 2015의 path-norm 관점과의 교차 |

</details>

<br/>

### 🔹 Chapter 7: 현대 Regularization과 종합

> **핵심 질문:** SWA는 왜 flat minimum을 찾는가? SAM은 sharpness를 어떻게 명시적으로 penalize하는가? AdamW는 Adam + L2와 무엇이 다른가? Transformer·CNN·GNN의 표준 recipe는 왜 그렇게 구성되는가?

<details>
<summary><b>SWA · SAM · AdamW · 4축 통합 recipe (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Stochastic Weight Averaging (Izmailov et al. 2018)](./ch7-modern-synthesis/01-swa.md) | **$\bar{\theta} = \frac{1}{T}\sum_{t=t_0}^{t_0+T} \theta_t$** 로 훈련 후반부 iterate 평균, **flat region**으로 이동하는 직관(loss surface 곡률과 평균화의 관계), **SWAG** (Maddox 2019): SWA에 Gaussian posterior 학습 → **Bayesian model averaging**으로 uncertainty, CIFAR-100에서 SWA vs SGD 비교 재현 |
| [02. Sharpness-Aware Minimization (Foret et al. 2021)](./ch7-modern-synthesis/02-sam.md) | **$\min_\theta \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$** 의 minimax 목적함수, 1차 근사 $\epsilon^*(\theta) = \rho \nabla L / \|\nabla L\|$로 **two-step gradient** 구현, **flat minimum의 명시적 탐색**, ASAM (Kwon 2021)의 adaptive $\rho$, cost 2배지만 generalization 향상 |
| [03. Weight Decay vs L2 in Adaptive Methods — AdamW](./ch7-modern-synthesis/03-adamw.md) | **Loshchilov & Hutter 2019** — **Adam에서 L2**를 loss에 더하면 gradient가 $v_t$로 스케일링되어 **coordinate별 weight decay가 학습률로 왜곡**, AdamW는 **$\theta_{t+1} = \theta_t - \eta \hat{m}_t/\sqrt{\hat{v}_t} - \eta \lambda \theta_t$** 로 decay를 **update 단계에서 분리**, 수식 대조와 ImageNet에서의 성능 재현 |
| [04. Regularization의 통합 관점 — Prior·Ensemble·Landscape·Invariance](./ch7-modern-synthesis/04-unified-recipe.md) | 모든 기법을 **4축**으로 재분류: (1) Prior (L1/L2/Spike-slab) (2) Ensemble (Dropout/DropConnect/Stochastic Depth) (3) Landscape (BN/LN/SAM/SWA) (4) Invariance (Mixup/LabelSmoothing), **실전 recipe**: Transformer(Pre-RMSNorm + AdamW + Warmup + Dropout + LabelSmoothing), CNN ImageNet(BN + SGD-momentum + WeightDecay + CutMix), GNN(LayerNorm + DropEdge + SpectralNorm) |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현**을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$로 종결되는 엄밀한 증명 또는 `results/` 하의 플롯을 확인할 수 있습니다. (전체 130+개 정리 중 핵심 19개 발췌)

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **L2 = Gaussian Prior MAP** | Gaussian likelihood + Gaussian prior → MAP 목적함수가 정확히 $\|y - Xw\|^2 + \lambda\|w\|^2$, $\lambda = \sigma^2/\sigma_w^2$ | [Ch1-01](./ch1-l1-l2/01-l2-gaussian-prior.md) |
| **L1 = Laplace Prior MAP** | Laplace prior의 negative log가 $\lambda\|w\|_1$, subdifferential로 coordinate-wise 최적성 | [Ch1-02](./ch1-l1-l2/02-l1-laplace-prior.md) |
| **Sparsity의 KKT 증명** | L1 ball의 꼭짓점에서 generic loss contour가 접한다는 기하, KKT로 sparse coordinate 유도 | [Ch1-03](./ch1-l1-l2/03-sparsity-geometry.md) |
| **Ridge의 SVD Shrinkage** | $\hat{w}_R = V\,\text{diag}(\sigma_i/(\sigma_i^2+\lambda))\,U^T y$ — 작은 singular value 방향 강한 축소 | [Ch1-04](./ch1-l1-l2/04-ridge-svd-shrinkage.md) |
| **Dropout = Geometric Mean Ensemble** | Test-time weight scaling $\times(1-p)$이 $2^N$ 서브네트워크 출력의 geometric mean 근사 | [Ch2-01](./ch2-dropout/01-dropout-ensemble.md) |
| **Dropout = VI** | Bernoulli variational posterior의 ELBO 최적화가 dropout + L2와 동치 (Gal 2016) | [Ch2-02](./ch2-dropout/02-dropout-as-vi.md) |
| **Dropout = Adaptive L2** | Linear regression dropout이 $\Gamma = p(1-p)\text{diag}(X^T X)$의 feature별 L2 (Wager 2013) | [Ch2-03](./ch2-dropout/03-dropout-adaptive-l2.md) |
| **Santurkar 2018 BN Smoothness** | BN이 loss · gradient의 Lipschitz 상수를 감소시킴 — ICS는 정당화가 아님 | [Ch3-02](./ch3-normalization/02-santurkar-refutation.md) |
| **Fixup 초기화로 BN-free ResNet** | Residual branch scale $L^{-1/(2m-2)}$로 깊은 NN 훈련 가능 (Zhang 2019) | [Ch3-05](./ch3-normalization/05-fixup-skipinit.md) |
| **VRM의 정식화** | Empirical delta measure → vicinity distribution으로 교체 (Chapelle 2000) | [Ch4-01](./ch4-data-augmentation/01-vicinal-risk.md) |
| **Mixup = VRM 특수 경우** | $\text{Beta}(\alpha,\alpha)$ 보간으로 vicinity를 정의한 VRM, linear decision boundary 강제 | [Ch4-03](./ch4-data-augmentation/03-mixup.md) |
| **Label Smoothing Gradient 억제** | Target logit을 무한히 밀지 않아 over-confidence 방지, ECE 개선 실측 | [Ch5-01](./ch5-label-calibration/01-label-smoothing.md) |
| **Temperature Scaling의 NLL 최소화** | 단일 scalar $T$를 validation NLL로 학습 → accuracy 보존 + ECE 감소 (Guo 2017) | [Ch5-04](./ch5-label-calibration/04-temperature-scaling.md) |
| **Early Stopping = Ridge (Yao 2007)** | GD iterate와 Ridge 해의 spectral filter 대응, $t \approx 1/(\eta\lambda)$ | [Ch6-01](./ch6-early-stopping-implicit/01-early-stopping-as-l2.md) |
| **Soudry 2018 Max-Margin 수렴** | Separable logistic에서 GD가 max-margin SVM 해로 수렴 | [Ch6-02](./ch6-early-stopping-implicit/02-sgd-implicit-bias.md) |
| **SWA Flat Minimum** | 훈련 후반 iterate 평균이 flat region으로 이동, SWAG의 Bayesian 해석 | [Ch7-01](./ch7-modern-synthesis/01-swa.md) |
| **SAM Minimax** | $\min_\theta \max_{\|\epsilon\|\leq\rho} L(\theta+\epsilon)$의 1차 근사 two-step gradient | [Ch7-02](./ch7-modern-synthesis/02-sam.md) |
| **AdamW Decoupled Decay** | Adam + L2의 $v_t$ 정규화로 인한 왜곡 vs AdamW의 분리된 $-\eta\lambda\theta_t$ | [Ch7-03](./ch7-modern-synthesis/03-adamw.md) |
| **4축 통합 분류** | Prior / Ensemble / Landscape / Invariance로 모든 기법 재배열, 실전 recipe 도출 | [Ch7-04](./ch7-modern-synthesis/04-unified-recipe.md) |

> 💡 **챕터별 문서·정리 수**: Ch1(5문서, 29정리) · Ch2(5문서, 23정리) · Ch3(6문서, 21정리) · Ch4(5문서, 12정리) · Ch5(4문서, 17정리) · Ch6(4문서, 18정리) · Ch7(4문서, 13정리) — 합계 **33문서 + 130+ 정리·증명·실험**, 약 **13,000+ 라인** 분량 (실험 노트북 포함 시 16k+).

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.0
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.0         # Lasso·Ridge baseline (Ch1)
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 seaborn==0.13.0 \
            tqdm==4.66.0 torch==2.1.0 torchvision==0.16.0 \
            scikit-learn==1.3.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — L1 vs L2의 sparsity 비교 (Ch1-03)
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n, p = 30, 20
X = np.random.randn(n, p)
beta_true = np.zeros(p); beta_true[:5] = [2, -1, 0.5, -0.5, 1]
y = X @ beta_true + 0.1 * np.random.randn(n)

def lasso_cd(X, y, lam, max_iter=1000):
    """Coordinate descent with soft thresholding."""
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        for j in range(X.shape[1]):
            r_j = y - X @ w + X[:, j] * w[j]
            z_j = X[:, j] @ r_j / n
            w[j] = np.sign(z_j) * max(abs(z_j) - lam, 0)
    return w

def ridge(X, y, lam):
    n = X.shape[0]
    return np.linalg.solve(X.T @ X + lam * n * np.eye(X.shape[1]), X.T @ y)

lams = np.logspace(-3, 1, 40)
lasso_coefs = np.array([lasso_cd(X, y, lam) for lam in lams])
ridge_coefs = np.array([ridge(X, y, lam) for lam in lams])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in range(p):
    axes[0].semilogx(lams, lasso_coefs[:, i])
    axes[1].semilogx(lams, ridge_coefs[:, i])
axes[0].set_title('L1 (Lasso): 코너에서 sparse — 정확히 0으로')
axes[1].set_title('L2 (Ridge): smooth shrinkage')
for ax in axes: ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('coef')
plt.tight_layout(); plt.show()
# → Lasso는 λ 증가에 따라 coefficient가 정확히 0이 되고, Ridge는 0에 점근만

# ─────────────────────────────────────────────
# Dropout = 앙상블 확인 — MC Dropout으로 uncertainty 재현 (Ch2-01, Ch2-02)
# ─────────────────────────────────────────────
import torch
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, d_in=1, d_h=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h, 1))
    def forward(self, x): return self.net(x)

# 훈련 (간략 생략). inference 시 train() 모드 유지하여 dropout 활성화
net = MLPWithDropout().eval()   # dropout OFF — 단일 예측
net.train()                     # dropout ON  — MC Dropout 여러 번 샘플

# T번 forward로 predictive mean/variance 추정 → Bayesian uncertainty 근사
x_eval = torch.linspace(-2, 2, 200).unsqueeze(1)
with torch.no_grad():
    preds = torch.stack([net(x_eval) for _ in range(100)])
mean, std = preds.mean(0), preds.std(0)
# mean ± 2·std로 95% 신뢰구간 → Gal 2016의 Variational Inference 근사

# ─────────────────────────────────────────────
# Santurkar 2018 재현 — with/without BN의 gradient Lipschitz (Ch3-02)
# ─────────────────────────────────────────────
# 같은 네트워크를 BN 유/무로 훈련, 각 step에서 η를 다르게 흔들며
# ‖∇L(θ + δ) - ∇L(θ)‖ / ‖δ‖ 측정 → BN 버전이 더 smooth

# ─────────────────────────────────────────────
# Mixup 재현 (Ch4-03)
# ─────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]
# Linear interpolation이 smoother decision boundary를 강제 → calibration 개선
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 regularization이 작동하는가 (신화 vs 실제)** | ICS 신화 같은 잘못된 믿음 식별·논파 |
| 3 | 📐 **수학적 선행 조건** | Bayes ML · NN Theory · Opt Theory 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해 — 여러 해석 병렬 제시** | Bayesian / Ensemble / Landscape / Invariance 중 가능한 모든 축으로 |
| 5 | ✏️ **엄밀한 정의** | MAP · ELBO · VRM · ECE의 측도/최적화적 정의 |
| 6 | 🔬 **정리와 증명** | L1 sparsity · Dropout = Adaptive L2 · BN Smoothness 등 |
| 7 | 💻 **실험으로 효과 검증** | with/without regularization의 weight 분포·activation 분포·landscape 비교 |
| 8 | 🔗 **실전 활용** | 언제 어느 regularization을 쓸 것인가 (Transformer/CNN/GNN) |
| 9 | ⚖️ **가정과 한계** | 각 기법이 실패하거나 해로운 경우 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현·논문 비평 |

> 📚 **연습문제 총 99개**: 33문서 × 문서당 3문제(기초/심화/논문 비평), 모든 문제에 `<details>` 펼침 해설 포함. MAP 유도부터 VI ELBO 재유도, AdamW vs Adam 수식 대조, SAM의 1차 근사 유도까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 280줄(증명·코드·연습문제 포함) 기준 **약 40분~1시간**. 전체 33문서는 약 **22~32시간** 상당 (증명 재구성·실험 재현 포함 시 40시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Regularization을 쓰지만 왜 되는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 9~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01  L2 = Gaussian Prior MAP
       Ch1-02  L1 = Laplace Prior MAP
Day 2  Ch1-03  Sparsity 기하·KKT
       Ch1-04  Ridge의 SVD Shrinkage
Day 3  Ch2-01  Dropout = 앙상블
       Ch2-02  Dropout = VI (Gal 2016)
Day 4  Ch3-01  BatchNorm 정의
       Ch3-02  Santurkar 2018 반박 — "신화 논파"
Day 5  Ch4-01  VRM
       Ch4-03  Mixup
Day 6  Ch5-01  Label Smoothing
       Ch5-04  Temperature Scaling
Day 7  Ch7-03  AdamW — Adam + L2와의 차이
       Ch7-04  4축 통합 관점
```

</details>

<details>
<summary><b>🟡 "Bayesian 관점으로 regularization을 완전 통일한다" — Bayesian 심화 (2주, 약 18~22시간)</b></summary>

<br/>

```
1주차
  Day 1-2  Ch1-01,02  L2/L1의 MAP 유도 — Gaussian/Laplace prior 완전 전개
  Day 3    Ch1-03     KKT로 sparsity 기하 유도
  Day 4-5  Ch2-02     Dropout = VI — ELBO + reparameterization 완전 유도
  Day 6-7  Ch2-03     Dropout = Adaptive L2 (Wager 2013) 선형 모델 증명

2주차
  Day 1    Ch4-01     VRM의 측도론적 정식화
  Day 2-3  Ch5-02     Label Smoothing과 Knowledge Distillation (Hinton)
  Day 4    Ch5-03     Maximum Entropy 원리와 confidence penalty
  Day 5-6  Ch7-01     SWA → SWAG: Bayesian model averaging으로 연결
  Day 7    Ch7-04     4축 통합 — 모든 기법을 Bayesian prior로 해석
```

</details>

<details>
<summary><b>🔴 "Explicit + Implicit regularization의 현대 recipe 완전 정복" — 전체 정복 (10주, 약 22~32시간 + 실험 재현 12~18시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — L1·L2의 통일 해석
         → Gaussian/Laplace MAP 손 유도
         → Lasso coordinate descent 구현
         → Ridge SVD shrinkage 실측

2주차   Chapter 2 전체 — Dropout의 3가지 해석
         → Srivastava 2014 앙상블 근사 재현
         → Gal 2016 MC Dropout으로 uncertainty
         → Wager 2013 linear adaptive L2 증명

3주차   Chapter 3 (1~3) — BatchNorm·Santurkar·LayerNorm
         → BN 없이/있이 loss landscape Lipschitz 측정
         → Transformer pre-LN vs post-LN 비교

4주차   Chapter 3 (4~6) — GN·IN·WN·Fixup·RMSNorm
         → CIFAR에서 GN vs BN small-batch 비교
         → Fixup으로 100-layer BN-free ResNet 훈련
         → RMSNorm 계산 효율 측정

5주차   Chapter 4 전체 — Data Augmentation
         → VRM 정식화, Chapelle 2000 읽기
         → Mixup의 calibration 개선 재현
         → SimCLR 두 view 학습 toy 재현

6주차   Chapter 5 전체 — Label & Calibration
         → ECE 측정 파이프라인 구축
         → Knowledge Distillation 구현 (teacher→student)
         → Temperature Scaling post-hoc 최적화

7주차   Chapter 6 전체 — Implicit Regularization
         → Early Stopping = Ridge 수치적 대응 관찰
         → Soudry 2018 max-margin 수렴 재현 (separable 2D)
         → Hastie 2019 ridgeless asymptotic 재구성

8주차   Chapter 7 (1~2) — SWA · SAM
         → SWA로 CIFAR-100 flat minimum 재현
         → SAM vs SGD landscape 시각화

9주차   Chapter 7 (3~4) — AdamW · 통합
         → Adam + L2 vs AdamW coordinate별 동작 차이 검증
         → Transformer/CNN/GNN recipe 비교표 완성

10주차  종합 — "Regularization의 지도" 다시 그리기
         → 4축(Prior/Ensemble/Landscape/Invariance) 재분류표
         → 각 챕터 한 장 요약
         → 열린 질문 목록 작성
```

</details>

---

## 💡 4축 통합 관점 요약

이 레포의 모든 기법은 다음 네 축으로 분류·통합됩니다. Chapter 7-04에서 완성되지만, 각 챕터에서 해당 축을 반복적으로 호출합니다.

```
┌──────── 1. Bayesian Prior ────────┐
│                                    │
│  MAP = argmax  p(y|θ) p(θ)         │
│      = argmin -log p(y|θ)          │
│                -log p(θ)           │
│                                    │
│  Prior            Neg log-prior     │
│  ─────            ───────────────  │
│  Gaussian    →    λ‖w‖²  (L2)       │
│  Laplace     →    λ‖w‖₁  (L1)       │
│  Uniform     →    0     (no reg)   │
│  Spike-slab  →    sparse+Gaussian   │
└────────────────────────────────────┘

┌──────── 2. Ensemble ────────┐
│                              │
│  Dropout (Srivastava 2014): │
│    2^N 개 subnetwork         │
│    test: weight × (1-p)      │
│    → geometric mean 근사     │
│                              │
│  Dropout ≈ VI (Gal 2016):   │
│    Bernoulli variational    │
│    MC Dropout → uncertainty │
│                              │
│  Dropout ≈ Adaptive L2      │
│    (Wager 2013, linear):    │
│    λ_i ∝ p(1-p) Var(x_i)    │
│                              │
│  Variants:                  │
│    ├── DropConnect (weight) │
│    ├── SpatialDropout (CNN) │
│    ├── VariationalRNNDrop  │
│    └── StochasticDepth     │
└─────────────────────────────┘

┌──────── 3. Landscape Smoothing ────────┐
│                                         │
│  BatchNorm (Ioffe 2015):                │
│    원래 주장: internal covariate shift │
│    Santurkar 2018 반박!                │
│    실제: Loss landscape smoothness     │
│    Gradient Lipschitz 더 작게          │
│                                         │
│  Norm 계보:                             │
│    ├── BN: batch 축                     │
│    ├── LN: feature 축 (Transformer)    │
│    ├── GN: channel group               │
│    ├── IN: sample별 (style)            │
│    ├── WN: weight 재매개변수화         │
│    └── RMSNorm: LN w/o centering       │
│                                         │
│  SAM (Foret 2021):                      │
│    min_θ max_ε L(θ+ε)                  │
│    flat minimum 명시적 탐색             │
│                                         │
│  SWA (Izmailov 2018):                   │
│    θ̄ = average of iterates              │
│    flat minimum implicit                │
└─────────────────────────────────────────┘

┌──────── 4. Invariance Injection ────────┐
│                                          │
│  Data Augmentation:                      │
│    Vicinal Risk Min (Chapelle 2000)     │
│    ERM: δ_(xi,yi)                        │
│    VRM: 𝒟_(xi,yi) vicinity               │
│                                          │
│  기법들:                                 │
│    ├── Rotation, flip, crop              │
│    ├── Mixup: λx_i + (1-λ)x_j            │
│    ├── CutMix: patch 교환                │
│    ├── CutOut: random erasing            │
│    └── AutoAugment/RandAugment           │
│                                          │
│  Label Smoothing (Szegedy 2016):        │
│    y → (1-α)y + α/K                      │
│    Calibration ↑, confidence ↓           │
│                                          │
│  Temperature Scaling (Guo 2017):        │
│    post-hoc calibration                  │
│    p = softmax(z/T)                      │
└──────────────────────────────────────────┘

───── Implicit Regularization (Ch6) ─────

Early Stopping ≈ L2 :
  t ≈ 1/(ηλ) 대응 (Yao 2007)

SGD :
  ├── Flat minimum 선호 (SDE)
  ├── Max-margin 수렴 (Soudry)
  └── Ridgeless interpolation (Hastie)

Initialization :
  Min-norm bias, NTK regime

───── 현대 NN Recipe (Ch7-04) ─────

Transformer :
  ├── Pre-RMSNorm (or LayerNorm)
  ├── AdamW (decoupled decay)
  ├── Warmup + Cosine LR
  ├── Dropout (attention, FFN)
  └── Label smoothing (분류 task)

CNN (ImageNet) :
  ├── BatchNorm
  ├── SGD + momentum
  ├── Weight decay (L2)
  ├── Random crop + flip
  ├── CutMix / Mixup
  └── Label smoothing

GNN :
  ├── LayerNorm
  ├── DropEdge / DropNode
  └── Spectral normalization
```

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [bayesian-ml-deep-dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive) | MAP · VI · prior · ELBO | **Ch1 전체** (L1/L2 = MAP), **Ch2-02** (Dropout = VI), Ch7-01 (SWAG) |
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | Backprop · 초기화 · 아키텍처 | Ch2 (dropout 구현), Ch3 (normalization), Ch3-05 (Fixup 초기화) |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | SGD 수렴 · landscape · implicit bias | **Ch6 전체**, Ch7-01 (SWA), Ch7-02 (SAM) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | Gaussian · Laplace · Beta 분포 | Ch1 (prior), Ch4-03 (Mixup의 Beta), Ch2 (Bernoulli dropout) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | L1/L2 projection · KKT · proximal | Ch1-03 (KKT), Ch1-05 (ISTA/FISTA) |
| [statistical-learning-theory-deep-dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive) | SRM · 복잡도 제어 · Rademacher | Ch1 (regularization = 복잡도 제어), Ch4-02 (augmentation의 Rademacher) |
| [generalization-theory-deep-dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive) | Implicit bias · flat minima · NTK | **Ch6 전체** (교차), Ch7-01,02 (flat minima) |

> 💡 이 레포는 **explicit regularization(L2, Dropout, BN, Mixup, Label Smoothing)** 의 통일 이론에 집중합니다. Bayesian ML에서 MAP과 VI를 이해하고, Optimization Theory에서 SGD와 landscape를 이해한 후 오면 Chapter 2·6·7의 추론이 자연스러워집니다. Generalization Theory Deep Dive와 Chapter 6이 **교집합**입니다.

---

## 📖 Reference

### 🏛️ 교과서 · 표준 참고
- **Deep Learning** (Goodfellow, Bengio, Courville, 2016) — Chapter 7 "Regularization for Deep Learning" 표준
- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman, 2009) — Ridge·Lasso의 고전
- **Pattern Recognition and Machine Learning** (Bishop, 2006) — Bayesian linear regression, MAP

### 🔢 L1 · L2 · Lasso · Ridge
- **Regression Shrinkage and Selection via the Lasso** (Tibshirani, 1996) — **L1 원전**
- **Ridge Regression: Biased Estimation for Nonorthogonal Problems** (Hoerl & Kennard, 1970)
- **Regularization and Variable Selection via the Elastic Net** (Zou & Hastie, 2005)
- **Model Selection and Estimation in Regression with Grouped Variables** (Yuan & Lin, 2006) — Group Lasso

### 🎲 Dropout · Ensemble · Variational Inference
- **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** (Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov, 2014) — **Dropout 원전**
- **Dropout Training as Adaptive Regularization** (Wager, Wang, Liang, 2013) — **Dropout = Adaptive L2**
- **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning** (Gal & Ghahramani, 2016) — **Dropout = VI**
- **Regularization of Neural Networks using DropConnect** (Wan et al., 2013)
- **Deep Networks with Stochastic Depth** (Huang et al., 2016)
- **Concrete Dropout** (Gal, Hron, Kendall, 2017)

### 🧪 Normalization
- **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** (Ioffe & Szegedy, 2015) — **BN 원전**
- **How Does Batch Normalization Help Optimization?** (Santurkar, Tsipras, Ilyas, Madry, 2018) — **ICS 반박**
- **Layer Normalization** (Ba, Kiros, Hinton, 2016)
- **Group Normalization** (Wu & He, 2018)
- **Instance Normalization** (Ulyanov, Vedaldi, Lempitsky, 2016)
- **Weight Normalization** (Salimans & Kingma, 2016)
- **Fixup Initialization: Residual Learning Without Normalization** (Zhang, Dauphin, Ma, 2019)
- **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019) — **RMSNorm**

### 🖼️ Data Augmentation · Vicinal Risk
- **Vicinal Risk Minimization** (Chapelle, Weston, Bottou, Vapnik, 2000) — **VRM 원전**
- **mixup: Beyond Empirical Risk Minimization** (Zhang, Cisse, Dauphin, Lopez-Paz, 2018) — **Mixup 원전**
- **CutMix: Regularization Strategy to Train Strong Classifiers** (Yun et al., 2019)
- **Improved Regularization of Convolutional Neural Networks with Cutout** (DeVries & Taylor, 2017)
- **AutoAugment** (Cubuk et al., 2018) · **RandAugment** (Cubuk et al., 2020)
- **A Simple Framework for Contrastive Learning of Visual Representations** (Chen et al., 2020) — **SimCLR**

### 🏷️ Label Regularization · Calibration
- **Rethinking the Inception Architecture for Computer Vision** (Szegedy et al., 2016) — **Label Smoothing** 도입
- **Distilling the Knowledge in a Neural Network** (Hinton, Vinyals, Dean, 2015) — **Knowledge Distillation**
- **Regularizing Neural Networks by Penalizing Confident Output Distributions** (Pereyra et al., 2017)
- **On Calibration of Modern Neural Networks** (Guo, Pleiss, Sun, Weinberger, 2017) — **Temperature Scaling**
- **When Does Label Smoothing Help?** (Müller, Kornblith, Hinton, 2019)

### ⏳ Implicit Regularization · Early Stopping · SGD
- **On Early Stopping in Gradient Descent Learning** (Yao, Rosasco, Caponnetto, 2007) — **Early Stopping = Ridge**
- **The Implicit Bias of Gradient Descent on Separable Data** (Soudry et al., 2018)
- **Surprises in High-Dimensional Ridgeless Least Squares Interpolation** (Hastie, Montanari, Rosset, Tibshirani, 2019)
- **Gradient Descent Maximizes the Margin of Homogeneous Neural Networks** (Lyu & Li, 2019)
- **On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima** (Keskar et al., 2017)

### 🚀 현대 Regularization · 옵티마이저 · Landscape
- **Averaging Weights Leads to Wider Optima and Better Generalization** (Izmailov et al., 2018) — **SWA**
- **A Simple Baseline for Bayesian Uncertainty in Deep Learning** (Maddox et al., 2019) — **SWAG**
- **Sharpness-Aware Minimization for Efficiently Improving Generalization** (Foret, Kleiner, Mobahi, Neyshabur, 2021) — **SAM**
- **ASAM: Adaptive Sharpness-Aware Minimization** (Kwon et al., 2021)
- **Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2019) — **AdamW**
- **High-Performance Large-Scale Image Recognition Without Normalization** (Brock et al., 2021) — **NFNet**

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"L2 $\lambda\|w\|^2$을 쓰는 것과 — 이것이 Gaussian prior MAP으로 $\lambda = \sigma^2/\sigma_w^2$ 대응임을 유도 · Dropout의 3가지 해석(앙상블·VI·Adaptive L2)을 각각 증명 · BN의 ICS 신화를 Santurkar 2018로 논파 · Mixup을 VRM의 특수 경우로 정식화 · AdamW가 Adam+L2와 왜 다른지 계산 — 이 모든 '왜'를 직접 유도할 수 있는 것은 다르다"*

</div>
