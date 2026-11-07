#!/usr/bin/env python3
"""Baselines and ablation for NERC, WITHOUT modifying nerc_ecg.py.

Models (all use the same record split, 100-D features, train-only min-max
scaling, ridge readout, calibration-quantile threshold and test evaluation):

  P0_persistence          predict next beat = current beat (no training)
  P1_linear_ridge         ridge regression u_t -> u_{t+1} (no reservoir)
  A_standard_esn_fixed    standard ESN, hand-set parameters, no GA
  B_dendritic_esn_fixed   dendritic ESN, same hand-set parameters, no GA
  C_standard_esn_ga       standard ESN, GA-tuned (same GA operators/budget)
  D_nerc_dendritic_ga     full NERC (dendritic + GA); genome is loaded from
                          your existing results/repeated_seeds/seed_X run
                          so it is the SAME model you already reported.

Fairness rules enforced here:
  * Hyper-parameters are chosen on VALIDATION records only.
  * The anomaly threshold comes from CALIBRATION residuals (label-free).
  * TEST records are used once, for final metrics.
  * Standard-ESN GA reuses nerc_ecg.genetic_search (same population,
    generations, mutation, crossover, elitism); only the state-update
    function is swapped.
"""

import argparse
import csv
import json
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import nerc_ecg as N

LOG = logging.getLogger("baselines")

# Conventional hand-set ESN values. Fixed in advance; NOT tuned on test data.
FIXED_GENOME = N.Genome(
    n_reservoir=100,
    spectral_radius=0.9,
    sparsity=0.10,
    input_scale=0.5,
    leak_rate=0.3,
    ridge_alpha=1e-4,
)

METRICS = [
    "normalized_feature_rmse",
    "normalized_feature_mae",
    "roc_auc",
    "average_precision",
    "threshold_precision",
    "threshold_recall",
    "threshold_f1",
]

MODEL_ORDER = [
    "P0_persistence",
    "P1_linear_ridge",
    "A_standard_esn_fixed",
    "B_dendritic_esn_fixed",
    "C_standard_esn_ga",
    "D_nerc_dendritic_ga",
]
REFERENCE_MODEL = "A_standard_esn_fixed"


# --------------------------------------------------------------------------
# Standard ESN state update (no dendrites, no gate)
# --------------------------------------------------------------------------
def standard_esn_states(features, genome, seed):
    """x_t = (1-a) x_{t-1} + a tanh(W x_{t-1} + Win u_t).

    Uses nerc_ecg.make_reservoir, so for the same genome and seed the
    recurrent matrix W and input matrix Win are IDENTICAL to the dendritic
    model; only the dendritic term and gate are removed.
    """
    W, Win, _Wd, _bias = N.make_reservoir(genome, features.shape[1], seed)
    n = genome.n_reservoir
    a = genome.leak_rate
    states = np.zeros((len(features), n), dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)
    for t, u in enumerate(features):
        x = ((1.0 - a) * x + a * np.tanh(W @ x + Win @ u)).astype(np.float32)
        states[t] = x
    return states


@contextmanager
def use_state_function(fn):
    """Temporarily swap nerc_ecg.reservoir_states (restored afterwards).

    genetic_search, state_target_pairs and predict_record all look up
    `reservoir_states` in the nerc_ecg module namespace, so this changes the
    reservoir for them without editing nerc_ecg.py.
    """
    original = N.reservoir_states
    N.reservoir_states = fn
    try:
        yield
    finally:
        N.reservoir_states = original


# --------------------------------------------------------------------------
# Generic evaluation
# --------------------------------------------------------------------------
def empty_output(record):
    return {
        "record_id": record["record_id"],
        "target": np.empty((0, 100)),
        "pred": np.empty((0, 100)),
        "scores": np.empty(0),
        "target_peaks": np.empty(0, int),
    }


def predict_with_function(record, lo, hi, predict_fn):
    f = N.apply_minmax(record["features"], lo, hi)
    if len(f) < 2:
        return empty_output(record)
    target = f[1:]
    pred = predict_fn(f[:-1])
    scores = np.linalg.norm(target - pred, axis=1) / math.sqrt(target.shape[1])
    return {
        "record_id": record["record_id"],
        "target": target,
        "pred": pred,
        "scores": scores,
        "target_peaks": record["peaks"][1:],
    }


def summarise(cal_out, test_out, test_records, cfg):
    cal_scores = np.concatenate(
        [o["scores"] for o in cal_out if len(o["scores"])]
    )
    threshold = float(np.quantile(cal_scores, cfg.calibration_quantile))

    targets = np.vstack([o["target"] for o in test_out if len(o["target"])])
    preds = np.vstack([o["pred"] for o in test_out if len(o["pred"])])

    row = {
        "test_pairs": int(len(targets)),
        "normalized_feature_rmse": float(
            np.sqrt(mean_squared_error(targets, preds))),
        "normalized_feature_mae": float(mean_absolute_error(targets, preds)),
    }
    anomaly = N.evaluate_anomaly(test_out, test_records, threshold)
    row["threshold"] = anomaly.get("threshold")
    row["evaluated_beats"] = anomaly.get("evaluated_beats")
    for key in ("roc_auc", "average_precision", "threshold_precision",
                "threshold_recall", "threshold_f1"):
        row[key] = anomaly.get(key)
    row["confusion_tn_fp_fn_tp"] = anomaly.get("confusion_matrix_tn_fp_fn_tp")
    return row


def run_reservoir_model(genome, state_fn, train, cal, test, lo, hi, cfg, seed):
    with use_state_function(state_fn):
        Xtr, Ytr = N.state_target_pairs(train, lo, hi, genome, seed)
        readout = N.fit_ridge(Xtr, Ytr, genome.ridge_alpha)
        cal_out = [N.predict_record(r, lo, hi, genome, readout, seed)
                   for r in cal]
        test_out = [N.predict_record(r, lo, hi, genome, readout, seed)
                    for r in test]
    return summarise(cal_out, test_out, test, cfg)


def run_persistence(cal, test, lo, hi, cfg):
    def fn(x):
        return x
    cal_out = [predict_with_function(r, lo, hi, fn) for r in cal]
    test_out = [predict_with_function(r, lo, hi, fn) for r in test]
    return summarise(cal_out, test_out, test, cfg)


def run_linear_ridge(train, val, cal, test, lo, hi, cfg):
    """No-reservoir baseline. Ridge alpha chosen on VALIDATION only."""
    X, Y = N.collect_pairs(train, lo, hi)
    grid = [10.0 ** e for e in range(-8, 0)]
    best_alpha, best_mse = None, float("inf")
    for alpha in grid:
        W = N.fit_ridge(X, Y, alpha)
        errs = []
        for r in val:
            f = N.apply_minmax(r["features"], lo, hi)
            if len(f) < 3:
                continue
            pred = np.hstack([f[:-1], np.ones((len(f) - 1, 1))]) @ W
            errs.append(np.mean((f[1:] - pred) ** 2))
        mse = float(np.mean(errs)) if errs else float("inf")
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse
    W = N.fit_ridge(X, Y, best_alpha)

    def fn(x):
        return np.hstack([x, np.ones((len(x), 1))]) @ W

    cal_out = [predict_with_function(r, lo, hi, fn) for r in cal]
    test_out = [predict_with_function(r, lo, hi, fn) for r in test]
    row = summarise(cal_out, test_out, test, cfg)
    row["selected_ridge_alpha"] = best_alpha
    return row


# --------------------------------------------------------------------------
# One seed
# --------------------------------------------------------------------------
def nerc_genome_for_seed(seed, args, train, val, lo, hi, cfg):
    path = Path(args.nerc_dir) / f"seed_{seed}" / "best_genome.json"
    if path.exists() and not args.rerun_nerc:
        LOG.info("Seed %d: loading existing NERC genome %s", seed, path)
        with path.open(encoding="utf-8") as f:
            return N.Genome(**json.load(f)), "loaded_from_existing_run"
    LOG.info("Seed %d: running NERC GA (no saved genome found)", seed)
    best, _ = N.genetic_search(train, val, lo, hi, cfg)
    return best, "re-run_in_this_script"


def run_seed(seed, args, data, lo, hi):
    train, val, cal, test = data["train"], data["val"], data["cal"], data["test"]
    cfg = N.Config(
        data_dir=args.data_dir,
        output_dir=str(args.output_dir),
        random_seed=seed,
        population_size=args.population,
        generations=args.generations,
        max_beats_per_record=args.max_beats,
    )
    N.set_seed(seed)
    rows, genomes = {}, {}

    t = time.time()
    rows["P0_persistence"] = run_persistence(cal, test, lo, hi, cfg)
    rows["P1_linear_ridge"] = run_linear_ridge(
        train, val, cal, test, lo, hi, cfg)
    LOG.info("Seed %d: simple baselines done (%.1fs)", seed, time.time() - t)

    genomes["A_standard_esn_fixed"] = FIXED_GENOME
    rows["A_standard_esn_fixed"] = run_reservoir_model(
        FIXED_GENOME, standard_esn_states, train, cal, test, lo, hi, cfg, seed)

    genomes["B_dendritic_esn_fixed"] = FIXED_GENOME
    rows["B_dendritic_esn_fixed"] = run_reservoir_model(
        FIXED_GENOME, N.reservoir_states, train, cal, test, lo, hi, cfg, seed)
    LOG.info("Seed %d: fixed-parameter ESNs done", seed)

    genome_source = {}
    if not args.skip_ga:
        # C: GA-optimised STANDARD ESN, same GA and budget as NERC.
        t = time.time()
        with use_state_function(standard_esn_states):
            c_genome, _ = N.genetic_search(train, val, lo, hi, cfg)
        LOG.info("Seed %d: standard-ESN GA done (%.1f min)",
                 seed, (time.time() - t) / 60)
        genomes["C_standard_esn_ga"] = c_genome
        rows["C_standard_esn_ga"] = run_reservoir_model(
            c_genome, standard_esn_states, train, cal, test, lo, hi, cfg, seed)

        # D: full NERC (dendritic + GA).
        d_genome, source = nerc_genome_for_seed(
            seed, args, train, val, lo, hi, cfg)
        genome_source["D_nerc_dendritic_ga"] = source
        genomes["D_nerc_dendritic_ga"] = d_genome
        rows["D_nerc_dendritic_ga"] = run_reservoir_model(
            d_genome, N.reservoir_states, train, cal, test, lo, hi, cfg, seed)

    for name, row in rows.items():
        row["model"] = name
        row["seed"] = seed
        if name in genomes:
            row["genome"] = asdict(genomes[name])
        if name in genome_source:
            row["genome_source"] = genome_source[name]
    return rows


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate(all_rows):
    summary = {}
    for model in MODEL_ORDER:
        model_rows = [r for r in all_rows if r["model"] == model]
        if not model_rows:
            continue
        entry = {"n_seeds": len(model_rows)}
        for metric in METRICS:
            vals = [float(r[metric]) for r in model_rows
                    if r.get(metric) is not None]
            entry[metric] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                # sample SD (ddof=1); undefined for a single run
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            }
        summary[model] = entry

    # Relative change vs the reference model (Eq. 52 in the paper).
    ref = summary.get(REFERENCE_MODEL)
    if ref:
        for model, entry in summary.items():
            entry["relative_change_vs_" + REFERENCE_MODEL + "_percent"] = {}
            for metric in METRICS:
                base = ref[metric]["mean"]
                cur = entry[metric]["mean"]
                if base and cur is not None:
                    entry["relative_change_vs_" + REFERENCE_MODEL
                          + "_percent"][metric] = (cur - base) / base * 100.0
    return summary


def fmt(entry, metric):
    m = entry[metric]
    if m["mean"] is None:
        return "NA"
    if m["std"] is None:
        return f'{m["mean"]:.4f}'
    return f'{m["mean"]:.4f} +/- {m["std"]:.4f}'


def print_table(summary):
    short = ["RMSE", "MAE", "ROC-AUC", "AP", "Prec", "Recall", "F1"]
    print("\n" + " | ".join(["Model".ljust(24)] + short))
    for model in MODEL_ORDER:
        if model in summary:
            cells = [model.ljust(24)] + [fmt(summary[model], m)
                                         for m in METRICS]
            print(" | ".join(cells))


# --------------------------------------------------------------------------
def load_split(cfg):
    data = {}
    for key, ids in (("train", N.TRAIN_RECORDS), ("val", N.VAL_RECORDS),
                     ("cal", N.CAL_RECORDS), ("test", N.TEST_RECORDS)):
        data[key] = []
        for rid in ids:
            data[key].append(N.load_record(rid, cfg))
            LOG.info("%s: loaded %s (%d beats)", key, rid,
                     len(data[key][-1]["features"]))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Baselines and ablation for NERC (nerc_ecg.py unchanged).")
    parser.add_argument("--data-dir", default="data/mitdb")
    parser.add_argument("--output-dir", default="results/baselines_ablation")
    parser.add_argument("--nerc-dir", default="results/repeated_seeds",
                        help="Folder with seed_X/best_genome.json from "
                             "run_repeated_seeds.py.")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 2025, 3407, 2026])
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-beats", type=int, default=0)
    parser.add_argument("--skip-ga", action="store_true",
                        help="Only run P0, P1, A, B (fast, minutes).")
    parser.add_argument("--rerun-nerc", action="store_true",
                        help="Re-run the NERC GA instead of loading the "
                             "saved genome.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute seeds that already have results.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parent
    for attr in ("data_dir", "output_dir", "nerc_dir"):
        p = Path(getattr(args, attr))
        setattr(args, attr, p if p.is_absolute() else (root / p).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_cfg = N.Config(data_dir=str(args.data_dir),
                        max_beats_per_record=args.max_beats)
    LOG.info("Loading records (several minutes)...")
    data = load_split(load_cfg)
    lo, hi = N.fit_global_minmax(data["train"])  # training records only

    all_rows = []
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result_path = seed_dir / "ablation_results.json"

        if result_path.exists() and not args.force:
            LOG.info("Seed %d: reusing %s", seed, result_path)
            with result_path.open(encoding="utf-8") as f:
                rows = json.load(f)
        else:
            rows_by_model = run_seed(seed, args, data, lo, hi)
            rows = list(rows_by_model.values())
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
        all_rows.extend(rows)

    summary = aggregate(all_rows)

    flat_cols = ["model", "seed", "test_pairs", "evaluated_beats"] + METRICS + [
        "threshold", "genome_source"]
    csv_path = args.output_dir / "baselines_ablation_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    out = {
        "seeds": args.seeds,
        "population": args.population,
        "generations": args.generations,
        "fixed_genome_for_A_and_B": asdict(FIXED_GENOME),
        "reference_model_for_relative_change": REFERENCE_MODEL,
        "summary_mean_and_sample_std": summary,
        "note": ("Hyper-parameters chosen on validation records, threshold "
                 "from calibration residuals, test records used only for "
                 "final metrics. P0/P1 are deterministic, so their std "
                 "reflects only the shared pipeline, not model randomness."),
    }
    json_path = args.output_dir / "baselines_ablation_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print_table(summary)
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")


if __name__ == "__main__":
    main()
