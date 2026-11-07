# Mathematical Methodology: Neuro-Evolutionary Reservoir Computing (NERC)

## 1. Overview
This document details the mathematical modeling, dendritic reservoir dynamics, evolutionary optimization, and classification readout pipeline of the NERC framework for electrocardiogram (ECG) beat-level analysis.

---

## 2. Signal Preprocessing & Temporal Segmentation

### 2.1 Bandpass Filtering
Raw Lead-II ECG records $s(t)$ sampled at 360 Hz are filtered through a 4th-order zero-phase Butterworth bandpass filter ($0.5 \text{ Hz} \le f \le 40 \text{ Hz}$):
$$H(s) = \prod_{k=1}^2 \frac{s^2 + \sqrt{2}\omega_c s + \omega_c^2}{s^2 + 2\cos(\theta_k)\omega_c s + \omega_c^2}$$
suppressing baseline wander, respiration drift, and high-frequency EMG noise.

### 2.2 Beat Morphological Extraction & Spectral Concatenation
For each annotated R-peak at index $t_R$, a window of $[-150\text{ ms}, +150\text{ ms}]$ is sliced (108 samples at 360 Hz) and resampled to $L = 50$ temporal points $u_{\text{time}} \in \mathbb{R}^{50}$.
The Discrete Fourier Transform magnitude spectrum is computed:
$$U_{\text{DFT}}[k] = \left| \sum_{n=0}^{L-1} u_{\text{time}}[n] e^{-j 2\pi k n / L} \right|, \quad k = 0, \dots, 49$$
The dual-domain representation is formed by concatenation:
$$u = [u_{\text{time}} ; U_{\text{DFT}}] \in \mathbb{R}^{D}, \quad D = 100$$
Augmented with RR-interval features (pre-RR, post-RR, local-average RR, and ratio metrics).

---

## 3. Leaky Dendritic Reservoir Dynamics

The reservoir consists of $N$ recurrently connected units with $B$ dendritic branches per neuron. At discrete beat step $t$:

### 3.1 Dendritic Non-linear Integration
$$d_{i, b}(t) = \sigma \left( \sum_{j=1}^D V_{i, b, j} u_j(t) + \sum_{k=1}^N Q_{i, b, k} x_k(t-1) + c_{i, b} \right)$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$ represents local non-linear dendritic branch gating.

### 3.2 State Update with Leakage
$$\tilde{x}_i(t) = \sum_{b=1}^B G_m[i, b] \cdot d_{i, b}(t) + \sum_{b=1}^B G_a[i, b] \cdot d_{i, b}(t) + b_g[i]$$
$$x(t) = (1 - \alpha) x(t-1) + \alpha \tanh \left( \frac{\rho}{\rho_{\max}} W_{\text{res}} x(t-1) + W_{\text{in}} u(t) + \beta \tilde{x}(t) \right)$$
where:
- $\alpha \in (0, 1]$ is the leaking rate (temporal memory scale).
- $\rho$ is the spectral radius scaling factor of reservoir weight matrix $W_{\text{res}}$.
- $\beta$ modulates the relative gain of dendritic input integration.

### 3.3 Reservoir State Aggregation
$$h(t) = [x(t) ; d(t) ; u(t)] \in \mathbb{R}^{H}, \quad H = 2N + D$$

---

## 4. Dual Ridge Readouts

The linear readouts are computed via $L_2$-regularized Ridge regression with penalty parameter $\lambda$:

1. **Next-Beat Waveform Forecasting:**
   $$W_{\text{pred}} = \arg\min_W \sum_{t} \|u(t+1) - W h(t)\|_2^2 + \lambda \|W\|_F^2 = (H^T H + \lambda I)^{-1} H^T U_{+1}$$

2. **Arrhythmia Classification:**
   $$W_{\text{cls}} = (H^T H + \lambda I)^{-1} H^T Y$$
   Calibrated via temperature scaling with factor $\kappa > 0$:
   $$P(y = c \mid h) = \frac{\exp(\kappa^{-1} W_{\text{cls}, c} h)}{\sum_{j=1}^C \exp(\kappa^{-1} W_{\text{cls}, j} h)}$$

---

## 5. Genetic Algorithm Optimization Formulation

The hyperparameter vector $\theta = [N, \rho, s, \gamma, \alpha, \beta]$ is optimized under a multi-objective fitness function combining classification accuracy and next-beat forecasting fidelity:
$$\mathcal{F}(\theta) = \lambda_1 \cdot \text{Macro-F1}(\theta) + \lambda_2 \cdot S_{\text{pred}}(\theta)$$
where:
$$S_{\text{pred}} = 1 - \frac{\text{RMSE}(\hat{u}, u)}{\text{RMSE}_{\text{persistence}}}$$
Optimized using tournament selection, BLX-$\alpha$ crossover, and adaptive Gaussian mutation.
