#!/usr/bin/env python3
"""
NERC - Neuro-Evolutionary Reservoir Computing for ECG
=====================================================
Implementation of the pipeline in "Adaptive Brain-Inspired Reservoir Computing
for ECG Prediction and Beat-Level Arrhythmia Detection" (Mondal & Kumar).

Pipeline (paper equation numbers in brackets)
---------------------------------------------
 1. Load MIT-BIH (wfdb) ............................ Sec. 4
 2. Butterworth band-pass 0.5-40 Hz, order 4 ....... (1)
 3. Beat segmentation: +/-150 ms around annotation,
    resample to 50 points ........................... (2)
 4. DFT magnitude, concat -> D = 100 ................ (3)-(5)
 5. Train-only global min-max normalisation ......... (6)-(7)
 6. Dendritic reservoir (B branches / unit,
    sigmoid gate, additive term, leaky update) ...... (8)-(13)
 7. Spectral-radius rescaling ....................... (14)
 8. Feature aggregation h = [x ; d ; u], H = 2N + D . (16)-(17)
 9. Dual ridge readout: next-beat prediction (19) and
    class-balanced 17-class classifier (20)-(21) .... (18)-(22)
10. Temperature-scaled softmax (fit on validation) ... (23)-(25)
11. Residual anomaly score + calibration threshold .. (27)-(31)
12. Classifier-based anomaly score .................. (32)
13. Genetic algorithm over theta = (N, rho, s, gamma,
    alpha, beta); fitness = l1*macroF1 + l2*S_pred .. (33)-(42)
14. Record-level train/val/cal/test split ........... (44)
15. Metrics, ablation (A-D), contraction test (15),
    memory capacity, latency ........................ (45)-(54)

v3 changes (driven by the first real DS1/DS2 result)
-----------------------------------------------------
 * --weight-power p : class weights = (balanced weight)**p. p=1 is the paper's
   Eq. 20 (it made 3,784 normal beats be called 'F'); try 0.5 or 0.
 * --zscore         : reservoir input uses per-feature z-scores (train stats)
   instead of global min-max (which squashes the waveform). Targets unchanged.
 * --wide-bounds    : GA ranges widened (gamma<=4, sparsity<=0.4, beta<=1e-1);
   the v2 optimum sat on 4 of 6 bounds.
 * --pin-train 208  : keep these DS1 records in TRAIN (record 208 holds almost
   all DS1 fusion beats; v2 put it in validation, leaving 38 F beats to train).
 * summary prints the trivial always-N accuracy for honest comparison.

v2 changes (for the paper-claim experiments)
-------------------------------------------
 * AAMI 5-class scheme (N,S,V,F,Q) is the default      (--classes aami|17)
 * RR-interval features (pre/post/local-avg RR + ratios) feed the reservoir
   (--no-rr disables).  Post-RR needs the NEXT beat -> one-beat delay.
 * Inter-patient de Chazal DS1/DS2 split is the default (--split dechazal);
   --split chunk is a WITHIN-PATIENT (leaky) split, for comparison only.
 * Ridge penalty search range raised to 1e-8..1e-3 (was 1e-11..1e-6)
 * GA: configurable mutation rate, optional weighted-F1 fitness

Usage
-----
  # real data (downloads MIT-BIH through wfdb if missing)
  python nerc_pipeline.py --data-dir ./mitdb --pop 50 --gens 100 --n-jobs 4

  # quick smoke test with synthetic ECG (no download needed)
  python nerc_pipeline.py --synthetic --pop 6 --gens 3 --n-max 150

  # skip the GA and use the paper's reported configuration (Eq. 43)
  python nerc_pipeline.py --data-dir ./mitdb --skip-ga

  # ablation A/B/C/D
  python nerc_pipeline.py --data-dir ./mitdb --ablation --ablation-seeds 3

Assumptions the paper does NOT specify (flagged in the code with  # ASSUMPTION)
- number of dendritic branches B, initialisation scales of V, Q, c, Gm, Ga, bg
- GA operators (tournament, BLX crossover, Gaussian mutation, elitism)
- identity of class 13 (the paper's class table skips it)
- which classes form the "normal" set N, and the quantile q
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal
from scipy.linalg import solve, LinAlgError
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FS = 360                       # MIT-BIH sampling rate (Hz)
HALF_WIN_MS = 150              # window on either side of the annotation
N_POINTS = 50                  # beat is resampled to 50 points
D = 2 * N_POINTS               # 50 waveform + 50 spectral magnitude = 100

# ---- class schemes ---------------------------------------------------------
# 17-class scheme from the paper's Table 5 (class 13 missing there -> 'j').  # ASSUMPTION
SYMBOLS_17 = ['N', 'L', 'R', 'A', 'a', 'J', 'V', 'E', 'F', 'f', '/', 'Q',
              '?', 'j', '!', 'x', 'e']
# Standard AAMI EC57 grouping (paced beats '/' and 'f' go to Q; 'B','!','x','?'
# are not used, as is common in the literature).
AAMI_GROUPS = {'N': ['N', 'L', 'R', 'e', 'j'], 'S': ['A', 'a', 'J', 'S'],
               'V': ['V', 'E'], 'F': ['F'], 'Q': ['/', 'f', 'Q']}
CLASS_SYMBOLS, SYM2CLASS, C, NORMAL_CLASSES = [], {}, 0, {0}


def set_class_scheme(name='aami'):
    """Select the label set. Must be called before loading data."""
    global CLASS_SYMBOLS, SYM2CLASS, C, NORMAL_CLASSES
    if name == 'aami':
        CLASS_SYMBOLS = list(AAMI_GROUPS)
        SYM2CLASS = {s: i for i, syms in enumerate(AAMI_GROUPS.values())
                     for s in syms}
    else:
        CLASS_SYMBOLS = list(SYMBOLS_17)
        SYM2CLASS = {s: i for i, s in enumerate(CLASS_SYMBOLS)}
    C = len(CLASS_SYMBOLS)
    NORMAL_CLASSES = {0}                    # class 'N'


set_class_scheme('aami')

# de Chazal et al. (2004) inter-patient split; paced records are excluded.
PACED = ['102', '104', '107', '217']      # paced-rhythm records
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
       '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
       '233', '234']

MITDB_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234']


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------
@dataclass
class Record:
    name: str
    U_raw: np.ndarray          # (K, D) un-normalised [waveform(mV) | |DFT|]
    y: np.ndarray              # (K,) class index
    duration_s: float          # recording length (for false alarms / hour)
    R: np.ndarray = None       # (K, 5) RR-interval features (fixed scaling)
    U: np.ndarray = None       # (K, D) min-max normalised morphology
    X: np.ndarray = None       # (K, D[+5]) reservoir input = [U | R]


@dataclass
class Params:
    """theta = (N, rho, s, gamma, alpha, beta)   Eq. (33).  s is a fraction."""
    N: int
    rho: float
    sparsity: float
    gamma: float
    alpha: float
    beta: float


# Paper's reported configuration, Eq. (43) / Table 2
PAPER_THETA = Params(N=433, rho=1.059, sparsity=0.0691, gamma=0.960,
                     alpha=0.892, beta=5.19e-11)
# Hand-picked baseline for ablation A/B                         # ASSUMPTION
FIXED_THETA = Params(N=200, rho=0.9, sparsity=0.10, gamma=1.0,
                     alpha=0.3, beta=1e-6)


def make_bounds(n_min=100, n_max=1000, wide=False):
    """Search ranges. The paper gives two conflicting tables; Table 2's ranges
    are used because they actually contain the reported optimum (rho=1.059,
    beta=5.19e-11).  beta is searched in log10 space."""
    return {
        'N': (n_min, n_max),
        'rho': (0.5, 1.5),
        'sparsity': (0.01, 0.40 if wide else 0.20),
        'gamma': (0.1, 4.0 if wide else 2.0),
        'alpha': (0.1, 1.0),
        'log10_beta': (-8.0, -1.0 if wide else -3.0),
    }


def decode(g) -> Params:
    return Params(N=int(round(g[0])), rho=float(g[1]), sparsity=float(g[2]),
                  gamma=float(g[3]), alpha=float(g[4]), beta=float(10 ** g[5]))


# ----------------------------------------------------------------------------
# 1-5. Preprocessing
# ----------------------------------------------------------------------------
def bandpass(x, fs=FS, lo=0.5, hi=40.0, order=4, causal=False):
    """Eq. (1). Zero-phase by default; use causal=True for a real-time system
    (the paper's real-time claim is only valid with a causal filter)."""
    sos = signal.butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return signal.sosfilt(sos, x) if causal else signal.sosfiltfilt(sos, x)


def segment_beats(x, sample_idx, symbols, fs=FS):
    """Eq. (2): +/-150 ms window around each *annotation* (not an R-peak
    detector), resampled to 50 points. Returns Z (K,50), labels (K,) and the
    sample positions of the kept beats (for RR features)."""
    half = int(round(HALF_WIN_MS * 1e-3 * fs))           # 54 samples
    segs, labels, kept = [], [], []
    for s, sym in zip(sample_idx, symbols):
        c = SYM2CLASS.get(sym)
        if c is None or s - half < 0 or s + half > len(x):
            continue
        segs.append(x[s - half:s + half])
        labels.append(c)
        kept.append(s)
    if not segs:
        return (np.zeros((0, N_POINTS)), np.zeros(0, dtype=int),
                np.zeros(0, dtype=int))
    Z = signal.resample(np.asarray(segs), N_POINTS, axis=1)
    return Z, np.asarray(labels, dtype=int), np.asarray(kept)


def rr_features(samples, fs=FS, local=10):
    """5 RR features per beat (de Chazal style): pre-RR, post-RR, local mean
    RR (last 10 beats), pre/local, post/local. Fixed, data-independent scaling
    to [0,1] so no information leaks from other splits.  NOTE: post-RR needs
    the following beat (one-beat delay in a real-time system)."""
    s = np.asarray(samples, float)
    K = len(s)
    rr = np.diff(s) / fs
    if K < 2:
        return np.full((K, 5), 0.5)
    pre = np.r_[rr[0], rr]                 # beat i: s[i]   - s[i-1]
    post = np.r_[rr, rr[-1]]               # beat i: s[i+1] - s[i]
    avg = np.array([pre[max(0, i - local + 1):i + 1].mean() for i in range(K)])
    clip = lambda v, hi: np.clip(v, 0.0, hi) / hi
    return np.stack([clip(pre, 2.0), clip(post, 2.0), clip(avg, 2.0),
                     clip(pre / avg, 2.0), clip(post / avg, 2.0)], axis=1)


def encode_beats(Z):
    """Eqs. (3)-(5): u_raw = [z ; |DFT(z)|]  in R^100."""
    M = np.abs(np.fft.fft(Z, axis=1))
    return np.hstack([Z, M])


def fit_minmax(records):
    """Eq. (6): global min / max over training beats and all D features."""
    allu = np.vstack([r.U_raw for r in records])
    return float(allu.min()), float(allu.max())


def apply_minmax(records, a, b, use_rr=True, zs=None):
    """Eq. (7). Handles the zero-range case explicitly. Builds the reservoir
    input X = [normalised morphology | RR features]."""
    scale = (b - a) if b > a else 1.0
    for r in records:
        r.U = (r.U_raw - a) / scale
        base = r.U if zs is None else np.clip((r.U_raw - zs[0]) / zs[1], -5, 5)
        r.X = np.hstack([base, r.R]) if (use_rr and r.R is not None) else base


def load_mitdb(data_dir, record_names=None, channel=0, causal=False,
               download=True):
    import wfdb
    os.makedirs(data_dir, exist_ok=True)
    names = record_names or MITDB_RECORDS
    if download and not os.path.exists(os.path.join(data_dir, '100.dat')):
        wfdb.dl_database('mitdb', data_dir)
    recs = []
    for n in names:
        path = os.path.join(data_dir, n)
        sig = wfdb.rdrecord(path, channels=[channel])
        ann = wfdb.rdann(path, 'atr')
        x = bandpass(sig.p_signal[:, 0], fs=sig.fs, causal=causal)
        Z, y, kept = segment_beats(x, ann.sample, ann.symbol, fs=sig.fs)
        if len(y) < 3:
            continue
        recs.append(Record(n, encode_beats(Z), y, len(x) / sig.fs,
                           R=rr_features(kept, sig.fs)))
    return recs


# ----------------------------------------------------------------------------
# Synthetic ECG (for smoke tests only)
# ----------------------------------------------------------------------------
def make_synthetic_records(n_records=8, dur_s=240, seed=0, causal=False):
    rng = np.random.default_rng(seed)
    recs = []
    for r in range(n_records):
        n = int(dur_s * FS)
        t = np.arange(n) / FS
        x = (0.15 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))
             + 0.02 * rng.standard_normal(n))

        def g(tc, amp, w):
            return amp * np.exp(-0.5 * ((t - tc) / w) ** 2)

        samples, syms = [], []
        tc = 0.6
        while tc < dur_s - 0.6:
            kind = rng.choice(['N', 'V', 'A'], p=[0.82, 0.10, 0.08])
            if kind == 'V':
                x += g(tc, 1.3, 0.035) - g(tc + 0.22, 0.4, 0.06)
            else:
                x += (g(tc - 0.16, 0.12, 0.025) - g(tc - 0.03, 0.1, 0.01)
                      + g(tc, 1.0, 0.012) - g(tc + 0.03, 0.2, 0.01)
                      + g(tc + 0.25, 0.3, 0.05))
            samples.append(int(tc * FS))
            syms.append(kind)
            rr = rng.normal(0.8, 0.04)
            tc += rr * (0.65 if kind in 'VA' else 1.0)
        xf = bandpass(x, causal=causal)
        Z, y, kept = segment_beats(xf, samples, syms)
        recs.append(Record(f'syn{r}', encode_beats(Z), y, dur_s,
                           R=rr_features(kept)))
    return recs


# ----------------------------------------------------------------------------
# 6-8. Dendritic reservoir
# ----------------------------------------------------------------------------
def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


class Reservoir:
    """Dendritic reservoir, Eqs. (8)-(14).  dendritic=False gives the standard
    leaky ESN of Eq. (51) used for ablation configs A and C."""

    def __init__(self, p: Params, d_in=D, branches=4, dendritic=True, seed=0):
        self.p, self.D, self.B, self.dendritic = p, d_in, branches, dendritic
        rng = np.random.default_rng(seed)
        N, B = p.N, branches

        # Sparse recurrent matrix rescaled to target spectral radius, Eq. (14)
        mask = rng.random((N, N)) < p.sparsity
        W0 = rng.uniform(-1, 1, (N, N)) * mask
        eig = np.max(np.abs(np.linalg.eigvals(W0))) if mask.any() else 0.0
        self.W = W0 * (p.rho / eig) if eig > 0 else W0
        self.Win = rng.uniform(-1, 1, (N, d_in))

        if dendritic:                                   # ASSUMPTION (scales)
            self.V = rng.uniform(-1, 1, (N * B, d_in)) / np.sqrt(d_in)
            self.Q = rng.uniform(-1, 1, (N * B, N)) / np.sqrt(N)
            self.c = rng.uniform(-0.1, 0.1, N * B)
            self.Gm = rng.normal(0, 1 / np.sqrt(N), (N, N))
            self.bg = np.zeros(N)
            self.Ga = rng.normal(0, 0.1 / np.sqrt(N), (N, N))

    @property
    def feat_dim(self):
        return (2 * self.p.N if self.dendritic else self.p.N) + self.D

    # one beat-level update --------------------------------------------------
    def _step(self, x, win_u, v_u):
        a = self.W @ x + win_u                                    # Eq. (10)
        if self.dendritic:
            N, B = self.p.N, self.B
            br = np.tanh(v_u + self.Q @ x)                        # Eq. (8)
            d = br.reshape(N, B).mean(axis=1)                     # Eq. (9)
            g = _sigmoid(self.Gm @ d + self.bg)                   # Eq. (11)
            m = self.Ga @ d                                       # Eq. (12)
            pre = g * a + m
        else:
            d, pre = None, a                                      # Eq. (51)
        al = self.p.alpha
        return (1 - al) * x + al * np.tanh(pre), d                # Eq. (13)

    def step(self, x, u):
        """Streaming single-beat update (used for latency measurement)."""
        win_u = self.p.gamma * (self.Win @ u)
        v_u = (self.V @ u + self.c) if self.dendritic else None
        return self._step(x, win_u, v_u)

    def run(self, U):
        """Run over one recording, x_0 = 0 (state is reset at every record).
        Returns the aggregated features H (K, feat_dim), Eq. (17)."""
        K, N = len(U), self.p.N
        win_u = self.p.gamma * (U @ self.Win.T)
        v_u = (U @ self.V.T + self.c) if self.dendritic else [None] * K
        X = np.empty((K, N))
        Dm = np.empty((K, N)) if self.dendritic else None
        x = np.zeros(N)
        for i in range(K):
            x, d = self._step(x, win_u[i], v_u[i])
            X[i] = x
            if self.dendritic:
                Dm[i] = d
        parts = [X, Dm, U] if self.dendritic else [X, U]          # h_i
        return np.hstack(parts)


# ----------------------------------------------------------------------------
# 9-10. Dual ridge readout
# ----------------------------------------------------------------------------
def ridge_solve(G, rhs, beta):
    """Solve (G + beta I) W = rhs without forming an inverse (paper's advice)."""
    A = G + beta * np.eye(G.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return solve(A, rhs, assume_a='pos')
        except (LinAlgError, ValueError):
            return np.linalg.lstsq(A, rhs, rcond=None)[0]


class DualReadout:
    def fit(self, Hs, Us, ys, beta, wpow=1.0):
        # Prediction pairs never cross recording boundaries.
        Hp = np.vstack([H[:-1] for H in Hs if len(H) > 1])
        Yp = np.vstack([U[1:] for U in Us if len(U) > 1])
        self.Wp = ridge_solve(Hp.T @ Hp, Hp.T @ Yp, beta).T          # Eq. (19)

        Hc = np.vstack(Hs)
        yc = np.concatenate(ys)
        counts = np.bincount(yc, minlength=C)
        present = (counts > 0).sum()
        w = (len(yc) / (present * np.maximum(counts[yc], 1))) ** wpow  # Eq. (20), wpow=1
        Y1 = np.zeros((len(yc), C))
        Y1[np.arange(len(yc)), yc] = 1.0
        Hw = Hc * w[:, None]
        self.Wc = ridge_solve(Hw.T @ Hc, Hw.T @ Y1, beta).T          # Eq. (21)
        return self

    def predict_next(self, H):
        return H @ self.Wp.T                                         # Eq. (18)

    def scores(self, H):
        return H @ self.Wc.T                                         # Eq. (22)


def softmax_T(S, kappa):
    return np.exp(kappa * S - logsumexp(kappa * S, axis=1, keepdims=True))


def fit_kappa(S_val, y_val):
    """Eq. (24): choose kappa by minimising validation NLL."""
    def nll(logk):
        P = softmax_T(S_val, np.exp(logk))
        return -np.mean(np.log(P[np.arange(len(y_val)), y_val] + 1e-12))
    r = minimize_scalar(nll, bounds=(-5, 8), method='bounded')
    return float(np.exp(r.x))


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def collect(ro: DualReadout, Hs, recs):
    """Gather predictions and aligned targets for a split (no boundary
    crossing). Pair i: forecast made from h_{i-1}, compared with beat i."""
    out = dict(y=[], S=[], T=[], P=[], base=[], y_pair=[], S_pair=[], hours=0.0)
    for H, r in zip(Hs, recs):
        S = ro.scores(H)
        out['y'].append(r.y)
        out['S'].append(S)
        out['hours'] += r.duration_s / 3600.0
        if len(H) > 1:
            out['P'].append(ro.predict_next(H[:-1]))
            out['T'].append(r.U[1:])
            out['base'].append(r.U[:-1])            # persistence, Eq. (35)
            out['y_pair'].append(r.y[1:])
            out['S_pair'].append(S[1:])
    for k in ['y', 'y_pair']:
        out[k] = np.concatenate(out[k])
    for k in ['S', 'T', 'P', 'base', 'S_pair']:
        out[k] = np.vstack(out[k])
    return out


def forecast_metrics(T, P, base, a, b):
    """Eqs. (34)(35)(45)(46) + waveform-only RMSE/MAE in mV (Eq. 26)."""
    rmse = float(np.sqrt(np.mean((T - P) ** 2)))
    mae = float(np.mean(np.abs(T - P)))
    rmse_p = float(np.sqrt(np.mean((T - base) ** 2)))
    scale = (b - a) if b > a else 1.0
    wf_t = a + scale * T[:, :N_POINTS]
    wf_p = a + scale * P[:, :N_POINTS]
    return dict(RMSE=rmse, MAE=mae, RMSE_persistence=rmse_p,
                S_pred=1 - rmse / rmse_p,
                RMSE_waveform_mV=float(np.sqrt(np.mean((wf_t - wf_p) ** 2))),
                MAE_waveform_mV=float(np.mean(np.abs(wf_t - wf_p))))


def classification_metrics(y, S):
    pred = S.argmax(1)
    labels = np.unique(y)                       # classes with test support
    p, r, f, sup = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        macro_F1=float(f1_score(y, pred, labels=labels, average='macro',
                                zero_division=0)),
        weighted_F1=float(f1_score(y, pred, labels=labels, average='weighted',
                                   zero_division=0)),
        per_class={CLASS_SYMBOLS[c]: dict(precision=float(pp), recall=float(rr),
                                          f1=float(ff), support=int(ss))
                   for c, pp, rr, ff, ss in zip(labels, p, r, f, sup)},
        confusion_matrix=confusion_matrix(y, pred, labels=range(C)).tolist())


def anomaly_metrics(score, y_bin, tau, hours):
    """Eqs. (49)(50) + AUROC / AUPRC / false alarms per hour."""
    pred = score > tau
    TP = int(np.sum(pred & (y_bin == 1)))
    TN = int(np.sum(~pred & (y_bin == 0)))
    FP = int(np.sum(pred & (y_bin == 0)))
    FN = int(np.sum(~pred & (y_bin == 1)))
    m = dict(threshold=float(tau), TP=TP, TN=TN, FP=FP, FN=FN,
             sensitivity=TP / max(TP + FN, 1), specificity=TN / max(TN + FP, 1),
             precision=TP / max(TP + FP, 1), F1=2 * TP / max(2 * TP + FP + FN, 1),
             false_alarms_per_hour=FP / max(hours, 1e-9))
    if len(np.unique(y_bin)) == 2:
        m['AUROC'] = float(roc_auc_score(y_bin, score))
        m['AUPRC'] = float(average_precision_score(y_bin, score))
    return m


def anomaly_scores(P, T, S_pair, kappa):
    """Eq. (28): A_i = ||u_i - u_hat_i||_2 / sqrt(D);   Eq. (32): A_cls."""
    a_res = np.linalg.norm(T - P, axis=1) / np.sqrt(T.shape[1])
    p = softmax_T(S_pair, kappa)
    a_cls = 1.0 - p[:, sorted(NORMAL_CLASSES)].sum(1)
    return a_res, a_cls


def is_anomaly(y):
    return (~np.isin(y, list(NORMAL_CLASSES))).astype(int)       # Eq. (31)


# ----------------------------------------------------------------------------
# 13. Genetic algorithm
# ----------------------------------------------------------------------------
def run_ga(fitness_fn, bounds, pop_size=50, gens=100, seed=0, n_jobs=1,
           elite=2, tournament=3, p_cx=0.8, p_mut=0.2, verbose=True):
    """Real-coded GA, Eqs. (37)-(42).  Operators are ASSUMPTIONS: tournament
    selection, BLX-0.3 crossover, per-gene Gaussian mutation (sigma = 10 % of
    range), clipping to bounds (projection Pi_Omega), elitism."""
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    lo = np.array([v[0] for v in bounds.values()], float)
    hi = np.array([v[1] for v in bounds.values()], float)
    pop = rng.uniform(lo, hi, (pop_size, len(lo)))
    cache, history = {}, []

    def key(g):
        return tuple(asdict(decode(g)).values())

    for g in range(gens):
        todo = {}
        for ind in pop:
            if key(ind) not in cache:
                todo[key(ind)] = ind
        res = Parallel(n_jobs=n_jobs)(
            delayed(fitness_fn)(decode(ind)) for ind in todo.values())
        for k, f in zip(todo.keys(), res):
            cache[k] = f
        fit = np.array([cache[key(i)] for i in pop])
        history.append(dict(gen=g + 1, best=float(fit.max()),
                            mean=float(fit.mean())))
        if verbose:
            print(f'  [GA] gen {g + 1:3d}/{gens}  best={fit.max():.4f}  '
                  f'mean={fit.mean():.4f}', flush=True)
        if g == gens - 1:
            break
        order = np.argsort(-fit)
        new = [pop[i].copy() for i in order[:elite]]
        while len(new) < pop_size:
            parents = []
            for _ in range(2):
                idx = rng.integers(0, pop_size, tournament)
                parents.append(pop[idx[np.argmax(fit[idx])]])
            a, b = parents
            child = a.copy()
            if rng.random() < p_cx:
                lam = rng.uniform(-0.3, 1.3, len(lo))
                child = a + lam * (b - a)
            mut = rng.random(len(lo)) < p_mut
            child = child + mut * rng.normal(0, 0.1 * (hi - lo))
            new.append(np.clip(child, lo, hi))
        pop = np.array(new)

    best_key = max(cache, key=cache.get)                      # Eq. (42)
    best = Params(*best_key)
    return best, float(cache[best_key]), history


# ----------------------------------------------------------------------------
# Diagnostics: contraction test (15), memory capacity, latency (53)
# ----------------------------------------------------------------------------
def contraction_test(res: Reservoir, U, n=500, eps=1e-8, seed=0):
    """Eq. (15): finite-time divergence of two trajectories started eps apart
    and driven by the same input. Negative => local contraction (not a proof)."""
    rng = np.random.default_rng(seed)
    n = min(n, len(U))
    x1 = np.zeros(res.p.N)
    dx = rng.standard_normal(res.p.N)
    x2 = x1 + eps * dx / np.linalg.norm(dx)
    d0 = np.linalg.norm(x2 - x1)
    for i in range(n):
        x1, _ = res.step(x1, U[i])
        x2, _ = res.step(x2, U[i])
    return float(np.log(np.linalg.norm(x2 - x1) / d0) / n)


def memory_capacity(res: Reservoir, k_max=60, T=3000, washout=100, seed=0):
    """Short-term MC = sum_k R^2(s_{t-k}, readout(state_t)). The scalar i.i.d.
    input drives input channel 0 only (other channels are 0). Broadcasting it
    over all D channels saturates tanh and underestimates MC.  # ASSUMPTION"""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, T)
    U = np.zeros((T, res.D))
    U[:, 0] = s
    H = res.run(U)[:, :-res.D]                  # reservoir (+dendritic) states
    cut = washout + (T - washout) * 2 // 3
    mc = 0.0
    for k in range(1, k_max + 1):
        Htr, ytr = H[washout + k:cut], s[washout:cut - k]
        Hte, yte = H[cut:T], s[cut - k:T - k]
        Wk = ridge_solve(Htr.T @ Htr, Htr.T @ ytr, 1e-6)
        pred = Hte @ Wk
        if np.std(pred) > 0 and np.std(yte) > 0:
            mc += np.corrcoef(pred, yte)[0, 1] ** 2
    return float(mc)


def latency_benchmark(res: Reservoir, ro: DualReadout, U, n=500):
    """Per-beat inference time: reservoir update + both readouts. Excludes
    filtering / segmentation (report those separately)."""
    n = min(n, len(U))
    x = np.zeros(res.p.N)
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        x, d = res.step(x, U[i])
        h = np.concatenate([x, d, U[i]] if d is not None else [x, U[i]])
        _ = ro.Wp @ h
        _ = ro.Wc @ h
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)))


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def split_records(recs, seed=0, frac=(0.55, 0.15, 0.10, 0.20)):
    """Eq. (44): record-level (patient-level) train / val / cal / test split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))
    n = len(recs)
    cuts = np.cumsum([int(round(f * n)) for f in frac[:-1]])
    parts = np.split(idx, cuts)
    return [[recs[i] for i in p] for p in parts]


def split_dechazal(recs, seed=0, n_val=4, n_cal=3, pin=()):
    """Inter-patient split: DS2 = test (fixed); val / cal / train are carved
    from DS1 by record. Falls back to a random record split if the standard
    record names are not present (e.g. synthetic data)."""
    by = {r.name: r for r in recs}
    ds1 = [by[n] for n in DS1 if n in by]
    ds2 = [by[n] for n in DS2 if n in by]
    pinned = [r for r in ds1 if r.name in pin]
    ds1 = [r for r in ds1 if r.name not in pin]
    if len(ds1) < n_val + n_cal + 3 or len(ds2) < 2:
        print('WARNING: DS1/DS2 records not found -> random record split')
        return split_records(recs, seed=seed)
    idx = np.random.default_rng(seed).permutation(len(ds1))
    pick = lambda ii: [ds1[i] for i in ii]
    return [pick(idx[n_val + n_cal:]) + pinned, pick(idx[:n_val]),
            pick(idx[n_val:n_val + n_cal]), ds2]


def split_chunks(recs, seed=0, chunk=100, frac=(0.55, 0.15, 0.10, 0.20)):
    """WITHIN-PATIENT split (LEAKY): 100-beat chunks of every record are
    assigned randomly to the four partitions, so the same patients appear in
    train and test. For comparison with leaky literature numbers only."""
    print('WARNING: --split chunk is within-patient and LEAKS patient '
          'information; do not report it as generalisation performance.')
    rng = np.random.default_rng(seed)
    parts = [[], [], [], []]
    for r in recs:
        for st in range(0, len(r.y), chunk):
            sl = slice(st, st + chunk)
            if len(r.y[sl]) < 3:
                continue
            k = int(rng.choice(4, p=frac))
            parts[k].append(Record(f'{r.name}_{st}', r.U_raw[sl], r.y[sl],
                                   r.duration_s * len(r.y[sl]) / len(r.y),
                                   R=None if r.R is None else r.R[sl]))
    return parts


def make_splits(recs, mode, seed, pin=()):
    cp = [Record(r.name, r.U_raw, r.y, r.duration_s, r.R) for r in recs]
    if mode == 'dechazal':
        return split_dechazal(cp, seed, pin=tuple(pin))
    if mode == 'chunk':
        return split_chunks(cp, seed)
    return split_records(cp, seed=seed)


def truncate(recs, m):
    """First m beats of each record (keeps temporal order; GA speed-up)."""
    if m is None:
        return recs
    return [Record(r.name, r.U_raw[:m], r.y[:m], r.duration_s,
                   None if r.R is None else r.R[:m], r.U[:m], r.X[:m])
            for r in recs]


def featurize(res, recs):
    return [res.run(r.X) for r in recs]


def make_fitness(train, val, branches, dendritic, lam1, lam2, seed,
                 f1_avg='macro', wpow=1.0):
    """Eqs. (34)-(36): J = l1 * macroF1 + l2 * S_pred on the validation split."""
    def fit(p: Params):
        try:
            res = Reservoir(p, d_in=train[0].X.shape[1], branches=branches,
                            dendritic=dendritic, seed=seed)
            Htr, Hva = featurize(res, train), featurize(res, val)
            ro = DualReadout().fit(Htr, [r.U for r in train],
                                   [r.y for r in train], p.beta, wpow)
            c = collect(ro, Hva, val)
            f1 = f1_score(c['y'], c['S'].argmax(1), labels=np.unique(c['y']),
                          average=f1_avg, zero_division=0)
            rm = np.sqrt(np.mean((c['T'] - c['P']) ** 2))
            rp = np.sqrt(np.mean((c['T'] - c['base']) ** 2))
            J = lam1 * f1 + lam2 * (1 - rm / rp)
            return float(J) if np.isfinite(J) else -1e9
        except Exception:
            return -1e9
    return fit


def run_pipeline(splits, args, dendritic=True, use_ga=True, theta=None,
                 seed=0, verbose=True):
    train, val, cal, test = splits
    a, b = fit_minmax(train)                                     # train only
    zs = None
    if args.zscore:
        allu = np.vstack([r.U_raw for r in train])
        zs = (allu.mean(0), allu.std(0) + 1e-8)
    for part in splits:
        apply_minmax(part, a, b, args.use_rr, zs)

    history = None
    if use_ga:
        bounds = make_bounds(args.n_min, args.n_max, args.wide_bounds)
        fit_fn = make_fitness(truncate(train, args.ga_beats),
                              truncate(val, args.ga_beats), args.branches,
                              dendritic, args.lam1, 1 - args.lam1, seed,
                              args.fitness_f1, args.weight_power)
        t0 = time.time()
        theta, best_J, history = run_ga(fit_fn, bounds, args.pop, args.gens,
                                        seed=seed, n_jobs=args.n_jobs,
                                        p_mut=args.p_mut, verbose=verbose)
        if verbose:
            print(f'  GA done in {time.time() - t0:.0f}s, best J={best_J:.4f}')
    theta = theta or (PAPER_THETA if dendritic else FIXED_THETA)
    if verbose:
        print('  theta* =', asdict(theta))

    # Final model on the full training split ------------------------------
    res = Reservoir(theta, d_in=train[0].X.shape[1], branches=args.branches,
                    dendritic=dendritic, seed=seed)
    H = {k: featurize(res, p) for k, p in
         dict(train=train, val=val, cal=cal, test=test).items()}
    ro = DualReadout().fit(H['train'], [r.U for r in train],
                           [r.y for r in train], theta.beta,
                           args.weight_power)

    cv = collect(ro, H['val'], val)
    kappa = fit_kappa(cv['S'], cv['y'])                          # Eq. (24)

    # Calibration threshold: q-quantile of normal-beat scores, Eq. (29)
    cc = collect(ro, H['cal'], cal)
    a_res_c, a_cls_c = anomaly_scores(cc['P'], cc['T'], cc['S_pair'], kappa)
    normal_c = is_anomaly(cc['y_pair']) == 0
    tau_res = float(np.quantile(a_res_c[normal_c], args.q))
    tau_cls = float(np.quantile(a_cls_c[normal_c], args.q))

    out = dict(theta=asdict(theta), kappa=kappa, dendritic=dendritic,
               ga_history=history, feature_dim=res.feat_dim,
               split_sizes={k: int(sum(len(r.y) for r in p)) for k, p in
                            dict(train=train, val=val, cal=cal, test=test).items()})
    for name, Hs, recs in [('train', H['train'], train), ('val', H['val'], val),
                           ('test', H['test'], test)]:
        c = collect(ro, Hs, recs)
        out[name] = dict(forecast=forecast_metrics(c['T'], c['P'], c['base'], a, b),
                         classification=classification_metrics(c['y'], c['S']))
        if name == 'test':
            a_res, a_cls = anomaly_scores(c['P'], c['T'], c['S_pair'], kappa)
            yb = is_anomaly(c['y_pair'])
            out['test']['anomaly_residual'] = anomaly_metrics(a_res, yb, tau_res,
                                                              c['hours'])
            out['test']['anomaly_classifier'] = anomaly_metrics(a_cls, yb, tau_cls,
                                                                c['hours'])
    # Diagnostics ---------------------------------------------------------
    if verbose or args.diagnostics:
        out['diagnostics'] = dict(
            contraction_Lambda=contraction_test(res, test[0].X),
            memory_capacity=memory_capacity(res),
            latency=latency_benchmark(res, ro, test[0].X),
            n_readout_params=int(ro.Wp.size + ro.Wc.size))
    out['_objects'] = (res, ro, (a, b), kappa, test, H['test'])
    return out


def summarise(out, title):
    t = out['test']
    f, c = t['forecast'], t['classification']
    print(f'\n=== {title} ===')
    print(f"  split sizes (beats): {out['split_sizes']}   H={out['feature_dim']}")
    print(f"  FORECAST  test RMSE={f['RMSE']:.4f}  MAE={f['MAE']:.4f}  "
          f"persistence RMSE={f['RMSE_persistence']:.4f}  S_pred={f['S_pred']:+.3f}"
          f"  waveform RMSE={f['RMSE_waveform_mV']:.4f} mV")
    print(f"  CLASSIFY  test acc={c['accuracy']:.4f}  macroF1={c['macro_F1']:.4f}"
          f"  weightedF1={c['weighted_F1']:.4f}")
    tot = sum(v['support'] for v in c['per_class'].values())
    nN = c['per_class'].get('N', {}).get('support', 0)
    print(f"  (trivial 'always N' accuracy = {nN / max(tot, 1):.4f}  <- the "
          f"number accuracy must be judged against)")
    print(f"  (overfit check) train acc={out['train']['classification']['accuracy']:.4f}"
          f"  val acc={out['val']['classification']['accuracy']:.4f}")
    print('  per-class (test):  class   Se(recall)  +P(precision)   F1   n')
    for sym, v in c['per_class'].items():
        print(f"                     {sym:>5s}   {v['recall']:.3f}       "
              f"{v['precision']:.3f}        {v['f1']:.3f}  {v['support']}")
    for k in ['anomaly_residual', 'anomaly_classifier']:
        m = t[k]
        print(f"  {k:19s} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f}"
              f" F1={m['F1']:.3f} AUROC={m.get('AUROC', float('nan')):.3f}"
              f" AUPRC={m.get('AUPRC', float('nan')):.3f}"
              f" FA/h={m['false_alarms_per_hour']:.1f}")
    if 'diagnostics' in out:
        d = out['diagnostics']
        print(f"  DIAG  Lambda={d['contraction_Lambda']:+.4f}  MC={d['memory_capacity']:.1f}"
              f"  latency mean={d['latency']['mean_ms']:.3f} ms "
              f"p95={d['latency']['p95_ms']:.3f} ms  readout params={d['n_readout_params']}")


def run_ablation(splits_fn, args):
    """Configs A-D (Sec. 'Ablation Study'), mean +/- std over seeds."""
    configs = {'A  ESN, fixed': (False, False), 'B  Dendritic, fixed': (True, False),
               'C  ESN + GA': (False, True), 'D  Full NERC': (True, True)}
    rows = {}
    for name, (dend, ga) in configs.items():
        vals = []
        for s in range(args.ablation_seeds):
            theta = FIXED_THETA if not ga else None
            out = run_pipeline(splits_fn(s), args, dendritic=dend, use_ga=ga,
                               theta=theta, seed=s, verbose=False)
            vals.append((out['test']['forecast']['RMSE'],
                         out['test']['classification']['macro_F1'],
                         out['test']['anomaly_residual'].get('AUROC', np.nan)))
        v = np.array(vals)
        rows[name] = dict(mean=v.mean(0).tolist(), std=v.std(0).tolist())
        print(f'  {name:22s} RMSE {v[:, 0].mean():.4f}+-{v[:, 0].std():.4f} | '
              f'macroF1 {v[:, 1].mean():.4f}+-{v[:, 1].std():.4f} | '
              f'AUROC(res) {np.nanmean(v[:, 2]):.4f}')
    return rows


# ----------------------------------------------------------------------------
# Optional plots
# ----------------------------------------------------------------------------
def make_plots(out, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if out['ga_history']:
        h = out['ga_history']
        plt.figure(figsize=(6, 3.5))
        plt.plot([x['gen'] for x in h], [x['best'] for x in h], label='best')
        plt.plot([x['gen'] for x in h], [x['mean'] for x in h], label='mean')
        plt.xlabel('generation'); plt.ylabel('fitness J'); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(outdir, 'ga_convergence.png'), dpi=150)
        plt.close()
    res, ro, (a, b), kappa, test, Hs = out['_objects']
    r, H = test[0], Hs[0]
    P = ro.predict_next(H[:-1])
    i = min(50, len(P) - 1)
    z_t = a + (b - a) * r.U[i + 1, :N_POINTS]
    z_p = a + (b - a) * P[i, :N_POINTS]
    plt.figure(figsize=(6, 3.5))
    plt.plot(z_t, label='actual next beat'); plt.plot(z_p, '--', label='predicted')
    plt.xlabel('resampled point'); plt.ylabel('mV'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'next_beat_prediction.png'), dpi=150)
    plt.close()


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if k != '_objects'}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='./mitdb')
    ap.add_argument('--synthetic', action='store_true', help='use synthetic ECG')
    ap.add_argument('--records', nargs='*', default=None)
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--causal', action='store_true', help='causal band-pass')
    ap.add_argument('--branches', type=int, default=4)           # ASSUMPTION
    ap.add_argument('--pop', type=int, default=50)
    ap.add_argument('--gens', type=int, default=100)
    ap.add_argument('--ga-beats', type=int, default=3000,
                    help='beats per record used inside GA fitness (speed); '
                         '0 = use all')
    ap.add_argument('--n-min', type=int, default=100)
    ap.add_argument('--n-max', type=int, default=1000)
    ap.add_argument('--lam1', type=float, default=0.5, help='lambda_1 in Eq. 36')
    ap.add_argument('--q', type=float, default=0.95, help='threshold quantile')
    ap.add_argument('--skip-ga', action='store_true',
                    help="use the paper's reported theta (Eq. 43)")
    ap.add_argument('--n-jobs', type=int, default=1)
    ap.add_argument('--classes', choices=['aami', '17'], default='aami')
    ap.add_argument('--split', choices=['dechazal', 'record', 'chunk'],
                    default='dechazal',
                    help='dechazal = inter-patient DS1/DS2 (default); '
                         'record = random record split; chunk = LEAKY')
    ap.add_argument('--no-rr', dest='use_rr', action='store_false',
                    help='disable RR-interval features')
    ap.add_argument('--fitness-f1', choices=['macro', 'weighted'],
                    default='macro')
    ap.add_argument('--p-mut', type=float, default=0.3)
    ap.add_argument('--weight-power', type=float, default=1.0,
                    help='class-weight exponent; 1 = paper Eq. 20')
    ap.add_argument('--zscore', action='store_true',
                    help='per-feature z-score reservoir inputs')
    ap.add_argument('--wide-bounds', action='store_true')
    ap.add_argument('--exclude-paced', action='store_true',
                    help='drop paced records 102/104/107/217 (DS1/DS2 always '
                         'does); needed for a MATCHED leaky-vs-inter-patient '
                         'comparison')
    ap.add_argument('--pin-train', nargs='*', default=['208'],
                    help='DS1 records forced into training')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ablation', action='store_true')
    ap.add_argument('--ablation-seeds', type=int, default=3)
    ap.add_argument('--diagnostics', action='store_true')
    ap.add_argument('--out', default='nerc_results')
    args = ap.parse_args()
    args.ga_beats = args.ga_beats or None
    os.makedirs(args.out, exist_ok=True)

    set_class_scheme(args.classes)
    print(f'Classes: {args.classes} {CLASS_SYMBOLS} | split: {args.split} '
          f'| RR features: {args.use_rr}')
    print('Loading data ...')
    recs = (make_synthetic_records(causal=args.causal, seed=args.seed)
            if args.synthetic else
            load_mitdb(args.data_dir, args.records, args.channel, args.causal))
    if args.exclude_paced:
        recs = [r for r in recs if r.name not in PACED]
    print(f'  {len(recs)} records, {sum(len(r.y) for r in recs)} beats')

    def splits_fn(seed):
        # fresh copies: normalisation is stored on the Record objects
        return make_splits(recs, args.split, seed, args.pin_train)

    if args.ablation:
        print('\nAblation (A: ESN, B: dendritic fixed, C: ESN+GA, D: NERC)')
        rows = run_ablation(splits_fn, args)
        with open(os.path.join(args.out, 'ablation.json'), 'w') as f:
            json.dump(rows, f, indent=2)
        return

    out = run_pipeline(splits_fn(args.seed), args, dendritic=True,
                       use_ga=not args.skip_ga, seed=args.seed)
    summarise(out, 'NERC (full model)')
    make_plots(out, args.out)
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(jsonable(out), f, indent=2)
    print(f'\nSaved results to {args.out}/')


if __name__ == '__main__':
    main()
