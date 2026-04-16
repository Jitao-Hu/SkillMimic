#!/usr/bin/env python3
"""
Scientific comparison for manual ablation CSV (seeded runs).

No third-party dependencies (pure Python stdlib).

Outputs:
  - Prints a summary table to stdout
  - Writes a CSV summary (default: logs/manual_ablation_summary.csv)

Typical use:
  python scripts/analyze_manual_ablation_comparison.py --input logs/manual_ablation_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from typing import Dict, Iterable, List, Tuple

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None


DEFAULT_METRICS = [
    "avg_reward",
    "avg_steps",
    "pass_success_rate",
    "catch_success_rate",
    "pass_successes",
    "pass_attempts",
    "catch_successes",
    "catch_attempts",
    "catch_fails",
]


def _infer_group_label(row: Dict[str, str]) -> str:
    """
    Infer group label from checkpoint path / command line.
    - Baseline: default
    - NoTrajPred: if checkpoint contains "NoTrajPred" or command uses ablation cfg
    """
    ckpt = (row.get("checkpoint") or "").strip()
    cmd = (row.get("command") or "").strip()
    if ("NoTrajPred" in ckpt) or ("ablate_no_traj_pred" in cmd):
        return "no_traj_pred"
    return "baseline"


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _mean(x: List[float]) -> float:
    xs = [v for v in x if _is_finite(v)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _std(x: List[float]) -> float:
    xs = [v for v in x if _is_finite(v)]
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = sum(xs) / n
    var = sum((v - mu) ** 2 for v in xs) / (n - 1)
    return math.sqrt(var)


def _format_mean_std(x: List[float]) -> str:
    xs = [v for v in x if _is_finite(v)]
    if not xs:
        return "NA"
    mu = _mean(xs)
    sd = _std(xs)
    if not _is_finite(sd):
        return f"{mu:.6g}"
    return f"{mu:.6g} ± {sd:.3g}"


def _cohens_d(a: List[float], b: List[float]) -> float:
    """
    Cohen's d for independent samples: (mean(b)-mean(a)) / pooled_std.
    Here we define d = (mean(group1) - mean(group0)) / pooled.
    """
    aa = [v for v in a if _is_finite(v)]
    bb = [v for v in b if _is_finite(v)]
    if len(aa) < 2 or len(bb) < 2:
        return float("nan")
    va = _std(aa) ** 2
    vb = _std(bb) ** 2
    pooled = math.sqrt(((len(aa) - 1) * va + (len(bb) - 1) * vb) / (len(aa) + len(bb) - 2))
    if pooled == 0.0 or not _is_finite(pooled):
        return float("nan")
    return float((_mean(bb) - _mean(aa)) / pooled)


def _bootstrap_mean_diff_ci(
    a: List[float],
    b: List[float],
    n_boot: int,
    seed: int,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for mean difference: mean(b) - mean(a).
    Returns: (diff, lo, hi)
    """
    aa = [v for v in a if _is_finite(v)]
    bb = [v for v in b if _is_finite(v)]
    if not aa or not bb:
        return float("nan"), float("nan"), float("nan")

    # Fast path with numpy (vectorized bootstrap)
    if _np is not None:
        rng = _np.random.default_rng(seed)
        aa_np = _np.asarray(aa, dtype=float)
        bb_np = _np.asarray(bb, dtype=float)
        idx_a = rng.integers(0, aa_np.size, size=(n_boot, aa_np.size))
        idx_b = rng.integers(0, bb_np.size, size=(n_boot, bb_np.size))
        boot = bb_np[idx_b].mean(axis=1) - aa_np[idx_a].mean(axis=1)
        diff = float(bb_np.mean() - aa_np.mean())
        alpha = (1.0 - ci) / 2.0
        lo = float(_np.quantile(boot, alpha))
        hi = float(_np.quantile(boot, 1.0 - alpha))
        return diff, lo, hi

    rng = random.Random(seed)
    boot: List[float] = []
    na, nb = len(aa), len(bb)
    for _ in range(n_boot):
        sa = [aa[rng.randrange(na)] for _ in range(na)]
        sb = [bb[rng.randrange(nb)] for _ in range(nb)]
        boot.append(_mean(sb) - _mean(sa))

    boot.sort()
    diff = _mean(bb) - _mean(aa)
    alpha = (1.0 - ci) / 2.0
    lo_i = int(math.floor(alpha * (len(boot) - 1)))
    hi_i = int(math.floor((1.0 - alpha) * (len(boot) - 1)))
    lo = float(boot[max(0, min(lo_i, len(boot) - 1))])
    hi = float(boot[max(0, min(hi_i, len(boot) - 1))])
    return diff, lo, hi


def _permutation_p_value_mean_diff(
    a: List[float], b: List[float], n_perm: int, seed: int
) -> float:
    """
    Two-sided permutation test p-value for mean(b) - mean(a).
    """
    aa = [v for v in a if _is_finite(v)]
    bb = [v for v in b if _is_finite(v)]
    if not aa or not bb:
        return float("nan")

    # Fast path with numpy (vectorized-ish permutations for small n)
    if _np is not None:
        rng = _np.random.default_rng(seed)
        aa_np = _np.asarray(aa, dtype=float)
        bb_np = _np.asarray(bb, dtype=float)
        obs = float(bb_np.mean() - aa_np.mean())
        pooled = _np.concatenate([aa_np, bb_np], axis=0)
        n_a = aa_np.size
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(pooled.size)
            da = pooled[perm[:n_a]]
            db = pooled[perm[n_a:]]
            stat = float(db.mean() - da.mean())
            if abs(stat) >= abs(obs):
                count += 1
        return float((count + 1) / (n_perm + 1))

    rng = random.Random(seed)
    obs = _mean(bb) - _mean(aa)
    pooled = aa + bb
    n_a = len(aa)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        da = pooled[:n_a]
        db = pooled[n_a:]
        stat = _mean(db) - _mean(da)
        if abs(stat) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))  # add-one smoothing


def _select_metrics(rows: List[Dict[str, str]], metrics: List[str]) -> List[str]:
    if not rows:
        return []
    cols = set(rows[0].keys())
    return [m for m in metrics if m in cols]


def _print_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("No rows to print.")
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="logs/manual_ablation_comparison.csv")
    ap.add_argument(
        "--output",
        default="logs/manual_ablation_summary.csv",
        help="Output summary CSV path.",
    )
    ap.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to compare.",
    )
    ap.add_argument(
        "--group_col",
        default="group",
        help="Name of inferred group column.",
    )
    ap.add_argument(
        "--baseline_label",
        default="baseline",
        help="Label for baseline group.",
    )
    ap.add_argument(
        "--treatment_label",
        default="no_traj_pred",
        help="Label for ablation/treatment group.",
    )
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--filter_completed_only",
        action="store_true",
        default=True,
        help="Keep only completed/normal runs (default: true).",
    )
    args = ap.parse_args()

    with open(args.input, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)
    if not rows:
        raise SystemExit(f"Empty input: {args.input}")

    # Optional filter to ensure comparable runs
    if args.filter_completed_only:
        def _keep(r: Dict[str, str]) -> bool:
            es = (r.get("exit_status") or "").strip()
            ec = (r.get("exit_category") or "").strip()
            return (es == "completed") and (ec == "normal")
        rows = [r for r in rows if _keep(r)]

    # Infer group label
    for r in rows:
        r[args.group_col] = _infer_group_label(r)

    metrics = _select_metrics(rows, [m.strip() for m in args.metrics.split(",") if m.strip()])

    base_rows = [r for r in rows if r.get(args.group_col) == args.baseline_label]
    treat_rows = [r for r in rows if r.get(args.group_col) == args.treatment_label]

    if not base_rows or not treat_rows:
        labels = sorted({r.get(args.group_col, "") for r in rows})
        raise SystemExit(
            "Could not find both groups after filtering.\n"
            f"Found labels={labels}, expected baseline='{args.baseline_label}', "
            f"treatment='{args.treatment_label}'."
        )

    summary_rows: List[Dict[str, str]] = []
    out_rows: List[Dict[str, str]] = []

    for m in metrics:
        def _to_float(v: str) -> float:
            try:
                return float(v)
            except Exception:
                return float("nan")

        a = [_to_float(r.get(m, "")) for r in base_rows]
        b = [_to_float(r.get(m, "")) for r in treat_rows]

        diff, lo, hi = _bootstrap_mean_diff_ci(a, b, n_boot=args.n_boot, seed=args.seed)
        p = _permutation_p_value_mean_diff(a, b, n_perm=args.n_perm, seed=args.seed + 1)
        d = _cohens_d(a, b)

        summary_rows.append(
            {
                "metric": m,
                args.baseline_label: _format_mean_std(a),
                args.treatment_label: _format_mean_std(b),
                "delta(treat-base)": "NA"
                if not _is_finite(diff)
                else f"{diff:.6g}  [{lo:.6g}, {hi:.6g}]",
                "cohens_d": "NA" if not _is_finite(d) else f"{d:.3g}",
                "perm_p": "NA" if not _is_finite(p) else f"{p:.3g}",
                "n_base": str(sum(1 for v in a if _is_finite(v))),
                "n_treat": str(sum(1 for v in b if _is_finite(v))),
            }
        )

        out_rows.append(
            {
                "metric": m,
                "baseline_mean": str(_mean(a)),
                "baseline_std": str(_std(a)),
                "treatment_mean": str(_mean(b)),
                "treatment_std": str(_std(b)),
                "delta_mean": str(diff),
                "delta_ci_lo": str(lo),
                "delta_ci_hi": str(hi),
                "cohens_d": str(d),
                "perm_p_value": str(p),
                "n_baseline": str(sum(1 for v in a if _is_finite(v))),
                "n_treatment": str(sum(1 for v in b if _is_finite(v))),
            }
        )

    # Print and write CSV
    print(f"Input: {args.input}")
    print(
        f"Groups: {args.baseline_label} (n={len(base_rows)}) vs "
        f"{args.treatment_label} (n={len(treat_rows)})"
    )
    print(f"Bootstrap: n_boot={args.n_boot}, Permutation: n_perm={args.n_perm}, seed={args.seed}")
    print()
    _print_table(summary_rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print()
    print(f"Wrote summary CSV: {args.output}")


if __name__ == "__main__":
    main()

