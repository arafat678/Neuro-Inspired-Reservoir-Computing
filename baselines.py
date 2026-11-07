#!/usr/bin/env python3
"""
Baselines for the NERC study, on EXACTLY the same data, splits, features,
metrics, thresholds and class weights as nerc_pipeline_v3.py.

Models
------
  ridge : the NERC dual ridge readout applied directly to the input features
          [beat morphology | RR] with NO reservoir.  If this matches NERC, the
          reservoir (and the dendrites) add nothing.
  cnn   : small 1-D CNN on the current beat (2 channels: waveform, |DFT|) + RR
  lstm  : 1-layer LSTM over the last W beats (default 10)
  The CNN and LSTM have two heads, mirroring NERC: class of beat i, and the
  next beat u_{i+1}.  Needs `pip install torch` (ridge does not).

Same protocol switches as NERC:  --split dechazal | chunk   --exclude-paced

Examples
--------
  # inter-patient, 5 seeds, all three baselines
  python baselines.py run --data-dir .\\mitdb --split dechazal --seeds 0 1 2 3 4 --out base_inter

  # matched within-patient (leaky) runs
  python baselines.py run --data-dir .\\mitdb --split chunk --exclude-paced --seeds 0 1 2 3 4 --out base_leaky

  # compare everything (any number of label=glob groups, glob or comma list)
  python baselines.py aggregate "NERC inter=results_inter_s*/results.json" \\
      "ridge inter=base_inter/dechazal_ridge_s*.json" "LSTM inter=base_inter/dechazal_lstm_s*.json"

Fairness notes
--------------
* NERC runs use the paper's fixed theta (no tuning). Baselines use a-priori
  hyper-parameters (hidden=64, lr=1e-3) and choose the stopping epoch /
  ridge penalty on VALIDATION records only. Nothing is tuned on the test set.
* Parameter counts: NERC's "n_readout_params" counts only the trained readout.
  Its fixed random reservoir stores roughly 1.4 million weights (W, Win, V, Q,
  Gm, Ga for N=433, B=4), so do NOT call it a 100k-parameter model.
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

import nerc_pipeline_v3 as n


# ----------------------------------------------------------------------------
# Packing records into flat arrays with sliding windows
# ----------------------------------------------------------------------------
def pack(recs, W, mu, sd):
    """Flat arrays for a split. `win[i]` = global indices of the last W beats
    ending at beat i of the same record (left-padded with the first beat)."""
    X, U, y, win, hn = [], [], [], [], []
    off = 0
    for r in recs:
        K = len(r.y)
        X.append(np.clip((r.X - mu) / sd, -5, 5))
        U.append(r.U)
        y.append(r.y)
        idx = np.arange(K) + off
        win.append(np.maximum(idx[:, None] + np.arange(-W + 1, 1)[None, :], off))
        h = np.ones(K, bool)
        h[-1] = False                      # last beat of a record has no next
        hn.append(h)
        off += K
    return dict(X=np.vstack(X).astype(np.float32),
                U=np.vstack(U).astype(np.float32),
                y=np.concatenate(y).astype(np.int64),
                win=np.vstack(win), has_next=np.concatenate(hn),
                hours=sum(r.duration_s for r in recs) / 3600.0)


def assemble(pk, logits, P):
    """Same structure as nerc_pipeline_v3.collect(): the forecast made at beat i
    is compared with beat i+1; class scores of the TARGET beat are used for the
    classifier-based anomaly score."""
    idx = np.where(pk['has_next'])[0]
    return dict(y=pk['y'], S=logits, T=pk['U'][idx + 1], P=P[idx],
                base=pk['U'][idx], y_pair=pk['y'][idx + 1],
                S_pair=logits[idx + 1], hours=pk['hours'])


def finish(cv, cc, ct, a, b, q=0.95):
    """Identical post-processing to NERC: temperature on validation, anomaly
    threshold = q-quantile of NORMAL calibration beats, metrics on test."""
    kappa = n.fit_kappa(cv['S'], cv['y'])
    ar_c, ac_c = n.anomaly_scores(cc['P'], cc['T'], cc['S_pair'], kappa)
    normal = n.is_anomaly(cc['y_pair']) == 0
    tau_r = float(np.quantile(ar_c[normal], q))
    tau_c = float(np.quantile(ac_c[normal], q))
    ar, ac = n.anomaly_scores(ct['P'], ct['T'], ct['S_pair'], kappa)
    yb = n.is_anomaly(ct['y_pair'])
    return dict(
        kappa=kappa,
        forecast=n.forecast_metrics(ct['T'], ct['P'], ct['base'], a, b),
        classification=n.classification_metrics(ct['y'], ct['S']),
        anomaly_residual=n.anomaly_metrics(ar, yb, tau_r, ct['hours']),
        anomaly_classifier=n.anomaly_metrics(ac, yb, tau_c, ct['hours']))


# ----------------------------------------------------------------------------
# Model 1: ridge readout on the raw features (no reservoir)
# ----------------------------------------------------------------------------
def run_ridge(train, val, cal, test, args):
    Hs = lambda recs: [r.X for r in recs]
    best = None
    for beta in 10.0 ** np.arange(-6, 1):
        ro = n.DualReadout().fit(Hs(train), [r.U for r in train],
                                 [r.y for r in train], beta, args.weight_power)
        cv = n.collect(ro, Hs(val), val)
        f1 = f1_score(cv['y'], cv['S'].argmax(1), labels=np.unique(cv['y']),
                      average='macro', zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, beta, ro)
    _, beta, ro = best
    cs = {k: n.collect(ro, Hs(p), p)
          for k, p in (('val', val), ('cal', cal), ('test', test))}
    x = test[0].X[:300]
    t0 = time.perf_counter()
    for i in range(len(x)):
        _ = ro.Wp @ x[i]
        _ = ro.Wc @ x[i]
    lat = (time.perf_counter() - t0) / len(x) * 1e3
    return cs, int(ro.Wp.size + ro.Wc.size), lat, dict(beta=float(beta))


# ----------------------------------------------------------------------------
# Models 2/3: CNN and LSTM (PyTorch)
# ----------------------------------------------------------------------------
def build_net(torch, name, C, hidden):
    nn = torch.nn

    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Sequential(
                nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
            s.fc = nn.Sequential(nn.Linear(32 * 12 + 5, hidden), nn.ReLU())
            s.cls = nn.Linear(hidden, C)
            s.fore = nn.Linear(hidden, n.D)

        def forward(s, x):                    # x: (B, W, 105) -> last beat only
            x = x[:, -1, :]
            z = s.conv(torch.stack([x[:, :50], x[:, 50:100]], 1)).flatten(1)
            h = s.fc(torch.cat([z, x[:, 100:105]], 1))
            return s.cls(h), s.fore(h)

    class LSTMNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(n.D + 5, hidden, batch_first=True)
            s.cls = nn.Linear(hidden, C)
            s.fore = nn.Linear(hidden, n.D)

        def forward(s, x):
            out, _ = s.lstm(x)
            h = out[:, -1]
            return s.cls(h), s.fore(h)

    return CNN() if name == 'cnn' else LSTMNet()


def run_torch(name, packs, Umu, Usd, args, seed):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    C = n.C
    net = build_net(torch, name, C, args.hidden).to(dev)
    nparams = int(sum(p.numel() for p in net.parameters()))

    tr, va = packs['train'], packs['val']
    counts = np.bincount(tr['y'], minlength=C)
    present = (counts > 0).sum()
    w = np.where(counts > 0, (len(tr['y']) / (present * np.maximum(counts, 1)))
                 ** args.weight_power, 0.0)
    ce = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def batch(pk, ib):
        x = torch.from_numpy(pk['X'][pk['win'][ib]]).to(dev)
        y = torch.from_numpy(pk['y'][ib]).to(dev)
        nxt = np.minimum(ib + 1, len(pk['y']) - 1)
        z = torch.from_numpy(((pk['U'][nxt] - Umu) / Usd).astype(np.float32)).to(dev)
        m = torch.from_numpy(pk['has_next'][ib].astype(np.float32)).to(dev)
        return x, y, z, m

    def loss_fn(lo, fo, y, z, m):
        l_f = (((fo - z) ** 2).mean(1) * m).sum() / m.sum().clamp(min=1.0)
        return ce(lo, y) + args.lam_forecast * l_f

    best_loss, best_state, bad = np.inf, None, 0
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(len(tr['y']))
        for s in range(0, len(perm), args.batch):
            x, y, z, m = batch(tr, perm[s:s + args.batch])
            lo, fo = net(x)
            loss = loss_fn(lo, fo, y, z, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va['y']), 4096):
                ib = np.arange(s, min(s + 4096, len(va['y'])))
                x, y, z, m = batch(va, ib)
                lo, fo = net(x)
                tot += float(loss_fn(lo, fo, y, z, m)) * len(ib)
                cnt += len(ib)
        vl = tot / cnt
        print(f'    [{name}] epoch {ep + 1:2d}  val loss {vl:.4f}', flush=True)
        if vl < best_loss - 1e-4:
            best_loss, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(pk):
        L, F = [], []
        with torch.no_grad():
            for s in range(0, len(pk['y']), 4096):
                ib = np.arange(s, min(s + 4096, len(pk['y'])))
                lo, fo = net(torch.from_numpy(pk['X'][pk['win'][ib]]).to(dev))
                L.append(lo.cpu().numpy())
                F.append(fo.cpu().numpy())
        return np.vstack(L), np.vstack(F) * Usd + Umu

    cs = {k: assemble(packs[k], *predict(packs[k])) for k in ('val', 'cal', 'test')}
    # per-beat latency (batch of 1, window W) -- upper bound for streaming use
    pk = packs['test']
    x1 = torch.from_numpy(pk['X'][pk['win'][:1]]).to(dev)
    with torch.no_grad():
        for _ in range(20):
            net(x1)
        t0 = time.perf_counter()
        for _ in range(300):
            net(x1)
    lat = (time.perf_counter() - t0) / 300 * 1e3
    return cs, nparams, lat, dict(epochs_run=ep + 1, device=dev)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(args):
    n.set_class_scheme('aami')
    os.makedirs(args.out, exist_ok=True)
    if args.synthetic:
        recs = n.make_synthetic_records(n_records=12, dur_s=120)
    else:
        recs = n.load_mitdb(args.data_dir)
    if args.exclude_paced:
        recs = [r for r in recs if r.name not in n.PACED]
    print(f'{len(recs)} records, {sum(len(r.y) for r in recs)} beats | '
          f'split={args.split} | models={args.models}')

    for seed in args.seeds:
        splits = n.make_splits(recs, args.split, seed, tuple(args.pin_train))
        train, val, cal, test = splits
        a, b = n.fit_minmax(train)
        for p in splits:
            n.apply_minmax(p, a, b, True, None)
        Xtr = np.vstack([r.X for r in train])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Utr = np.vstack([r.U for r in train])
        Umu, Usd = Utr.mean(0), Utr.std(0) + 1e-8
        packs = {k: pack(p, args.window, mu, sd) for k, p in
                 zip(['train', 'val', 'cal', 'test'], splits)}
        sizes = {k: int(len(v['y'])) for k, v in packs.items()}
        for model in args.models:
            print(f'== seed {seed} | {model} | sizes {sizes}', flush=True)
            t0 = time.time()
            if model == 'ridge':
                cs, nparams, lat, extra = run_ridge(train, val, cal, test, args)
            else:
                cs, nparams, lat, extra = run_torch(model, packs, Umu, Usd, args, seed)
            test_res = finish(cs['val'], cs['cal'], cs['test'], a, b, args.q)
            res = dict(model=model, split=args.split, seed=seed, extra=extra,
                       split_sizes=sizes, train_seconds=time.time() - t0,
                       test=test_res,
                       diagnostics=dict(latency=dict(mean_ms=lat),
                                        n_readout_params=nparams))
            path = os.path.join(args.out, f'{args.split}_{model}_s{seed}.json')
            with open(path, 'w') as f:
                json.dump(n.jsonable(res), f, indent=2)
            c = test_res['classification']
            pc = c['per_class']
            tot = sum(v['support'] for v in pc.values())
            print(f"   acc={c['accuracy']:.4f} (always-N {pc['N']['support'] / tot:.4f}) "
                  f"macroF1={c['macro_F1']:.3f} S_pred={test_res['forecast']['S_pred']:+.3f} "
                  f"AUROC cls={test_res['anomaly_classifier'].get('AUROC', float('nan')):.3f} "
                  f"res={test_res['anomaly_residual'].get('AUROC', float('nan')):.3f} "
                  f"| params={nparams} lat={lat:.3f} ms -> {path}", flush=True)


# ----------------------------------------------------------------------------
# Aggregation: label=glob (or comma-separated files) -> mean +- std table
# ----------------------------------------------------------------------------
def _metrics():
    def pcv(c, k):
        return lambda d: d['test']['classification']['per_class'].get(c, {}).get(k, np.nan)
    def alwaysN(d):
        pc = d['test']['classification']['per_class']
        return pc['N']['support'] / sum(v['support'] for v in pc.values())
    def macro4(d):
        pc = d['test']['classification']['per_class']
        return np.mean([pc[c]['f1'] for c in 'NSVF' if c in pc])
    T = lambda d: d['test']
    return [
        ('accuracy', lambda d: T(d)['classification']['accuracy']),
        ('accuracy - always-N', lambda d: T(d)['classification']['accuracy'] - alwaysN(d)),
        ('macro-F1 (N,S,V,F)', macro4),
        ('S sensitivity', pcv('S', 'recall')), ('S precision', pcv('S', 'precision')),
        ('V sensitivity', pcv('V', 'recall')), ('V precision', pcv('V', 'precision')),
        ('F sensitivity', pcv('F', 'recall')),
        ('forecast S_pred', lambda d: T(d)['forecast']['S_pred']),
        ('forecast RMSE (mV)', lambda d: T(d)['forecast']['RMSE_waveform_mV']),
        ('classifier AUROC', lambda d: T(d)['anomaly_classifier'].get('AUROC', np.nan)),
        ('classifier AUPRC', lambda d: T(d)['anomaly_classifier'].get('AUPRC', np.nan)),
        ('residual AUROC', lambda d: T(d)['anomaly_residual'].get('AUROC', np.nan)),
        ('classifier FA/hour', lambda d: T(d)['anomaly_classifier']['false_alarms_per_hour']),
        ('latency (ms/beat)', lambda d: d.get('diagnostics', {}).get('latency', {}).get('mean_ms', np.nan)),
        ('trained params', lambda d: d.get('diagnostics', {}).get('n_readout_params', np.nan)),
    ]


def aggregate(specs):
    groups = []
    for sp in specs:
        label, _, pat = sp.partition('=')
        files = sorted(sum([glob.glob(p) for p in pat.split(',')], []))
        if not files:
            print(f'WARNING: no files for "{label}" ({pat})')
            continue
        groups.append((label, [json.load(open(f)) for f in files]))
    if not groups:
        return
    w = 22
    print('%-24s' % 'metric' + ''.join('%-*s' % (w, f'{l} (n={len(D)})') for l, D in groups))
    for name, fn in _metrics():
        row = '%-24s' % name
        for _, D in groups:
            v = np.array([fn(d) for d in D], float)
            row += '%-*s' % (w, '%.3f ± %.3f' % (np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else 0.0))
        print(row)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'aggregate':
        aggregate(sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run'])
    ap.add_argument('--data-dir', default='./mitdb')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--split', choices=['dechazal', 'chunk', 'record'], default='dechazal')
    ap.add_argument('--exclude-paced', action='store_true')
    ap.add_argument('--pin-train', nargs='*', default=['208'])
    ap.add_argument('--models', nargs='+', default=['ridge', 'cnn', 'lstm'],
                    choices=['ridge', 'cnn', 'lstm'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--window', type=int, default=10, help='LSTM history (beats)')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam-forecast', type=float, default=1.0)
    ap.add_argument('--weight-power', type=float, default=1.0,
                    help='same class-weight exponent as the NERC runs (default 1)')
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--out', default='baselines_out')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
