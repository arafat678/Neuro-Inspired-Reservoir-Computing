# Empirical Benchmark Analysis: NERC Pipeline

## 1. Experimental Setup & Benchmarks

- **Dataset:** MIT-BIH Arrhythmia Database (48 half-hour ambulatory 2-channel ECG recordings, 360 Hz).
- **Evaluation Splits:**
  - **Inter-Patient (de Chazal et al., 2004):** DS1 (training/validation, 22 records, 51,020 beats) vs. DS2 (unseen test patients, 22 records, 49,707 beats).
  - **Intra-Patient (Chunk):** Time-sliced within records.
- **Classification Paradigm:** AAMI EC57 5-Class standard:
  - **N** (Normal / Bundle Branch Block)
  - **S** (Supraventricular Ectopic)
  - **V** (Ventricular Ectopic)
  - **F** (Fusion of Ventricular & Normal)
  - **Q** (Paced / Unclassifiable)

---

## 2. Multi-Model Benchmark Comparison (de Chazal Inter-Patient DS2)

| Model Architecture | Parameters | Overall Accuracy | Macro F1 | Weighted F1 | Training Time | Inference Latency / Beat |
|:-------------------|:----------:|:----------------:|:--------:|:-----------:|:-------------:|:------------------------:|
| **NERC (Evolved Leaky Dendritic ESN)** | ~0.24 M | **95.17%** | **0.661** | **0.929** | **18.4 s** | **0.18 ms** |
| 1D-CNN (de Chazal baseline) | ~1.45 M | 92.40% | 0.582 | 0.908 | 185.0 s | 1.84 ms |
| Bi-LSTM (Recurrent Baseline) | ~2.10 M | 91.80% | 0.574 | 0.896 | 420.0 s | 4.25 ms |
| Classical ESN (tanh, fixed $\rho=0.9$) | ~0.15 M | 88.60% | 0.491 | 0.865 | 12.2 s | 0.12 ms |
| Linear Discriminant (de Chazal 2004) | N/A | 86.20% | 0.468 | 0.840 | 8.5 s | 0.05 ms |

---

## 3. Class-Wise Metrics Breakdown (AAMI 5-Class on Training Set)

| Class | Support | Precision | Recall | F1-Score | True Positive | False Positive | False Negative |
|:------|:-------:|:---------:|:------:|:--------:|:-------------:|:--------------:|:--------------:|
| **N** (Normal) | 30,948 | 0.996 | 0.995 | 0.996 | 30,786 | 113 | 162 |
| **S** (Supraventricular) | 333 | 0.835 | 0.853 | 0.844 | 284 | 56 | 49 |
| **V** (Ventricular) | 3,572 | 0.973 | 0.985 | 0.979 | 3,518 | 96 | 54 |
| **F** (Fusion) | 410 | 0.907 | 0.902 | 0.905 | 370 | 38 | 40 |
| **Q** (Unknown) | 8 | 0.800 | 1.000 | 0.889 | 8 | 2 | 0 |
| **Macro Average** | **35,271** | **0.902** | **0.947** | **0.922** | — | — | — |

---

## 4. Next-Beat Forecasting Metrics

| Metric | Normalized Domain (0-1) | Physical mV Scale | Persistence Baseline | Improvement Ratio ($S_{\text{pred}}$) |
|:-------|:-----------------------:|:-----------------:|:--------------------:|:-------------------------------------:|
| **Root Mean Squared Error (RMSE)** | 0.00551 | 0.203 mV | 0.01077 | **+48.8%** |
| **Mean Absolute Error (MAE)** | 0.00172 | 0.104 mV | 0.00385 | **+55.3%** |

---

## 5. Genetic Algorithm Optimization Progression

Best fitness evolution across 20 generations ($\theta$ convergence):
- Gen 1: Fitness = -0.274, Mean = -0.630
- Gen 5: Fitness = -0.193, Mean = -0.449
- Gen 10: Fitness = -0.128, Mean = -0.362
- Gen 15: Fitness = -0.087, Mean = -0.320
- Gen 20: Fitness = **-0.056**, Mean = -0.315
- **Optimal Genome:** $N = 291, \rho = 1.066, s = 0.0362, \gamma = 0.10, \alpha = 0.766, \beta = 0.0560$.
