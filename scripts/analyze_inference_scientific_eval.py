#!/usr/bin/env python3
"""
Scientific eval Steps 6–7: sanity-check inference CSV rows and write an augmented subset CSV.

Reads inference logs only (no checkpoint loading). Never modifies the input CSV or weights.

Requires numpy (and optionally scipy for pairwise p-values). Run with the project conda env, e.g.:
  conda run -n skillmimic python scripts/analyze_inference_scientific_eval.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None

DEFAULT_MILESTONE_BASENAMES = ("CTDE_2000.pth", "CTDE_5000.pth", "CTDE_8000.pth")
DEFAULT_TASK = "HRLCTDEHumanoid"
DEFAULT_MOTION = "skillmimic/data/motions/BallPlay-M/pass"
DEFAULT_CFG_ENV = "skillmimic/data/cfg/hrl_ctde_humanoid.yaml"
DEFAULT_CFG_TRAIN = "skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml"
EXPECTED_SEEDS = frozenset(range(5))
Z_95 = 1.96


def _parse_bool(val: Any) -> Optional[bool]:
    if val is None or val == "":
        return None
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _parse_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _parse_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _row_sort_key(row: Dict[str, str]) -> Tuple[str, str]:
    """Prefer end_time, then timestamp (ISO strings sort lexicographically)."""
    end = (row.get("end_time") or "").strip()
    ts = (row.get("timestamp") or "").strip()
    return (end or ts, ts)


def default_checkpoint_filter(checkpoint: str) -> bool:
    c = checkpoint.strip()
    if not c.startswith("output/"):
        return False
    base = os.path.basename(c)
    return base in DEFAULT_MILESTONE_BASENAMES


def milestone_order_key(checkpoint: str) -> Tuple[int, str]:
    base = os.path.basename(checkpoint)
    for prefix, ep in (("CTDE_2000", 2000), ("CTDE_5000", 5000), ("CTDE_8000", 8000)):
        if base.startswith(prefix):
            return (ep, checkpoint)
    return (10**9, checkpoint)


def analysis_condition_label(checkpoint: str) -> str:
    base = os.path.basename(checkpoint).replace(".pth", "")
    return base or checkpoint


def row_passes_protocol(
    row: Dict[str, str],
    *,
    motion_file: str,
    task: str,
    cfg_env: str,
    cfg_train: str,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if (row.get("exit_status") or "").strip() != "completed":
        reasons.append("exit_status!=completed")
    if _parse_int(row.get("test_episodes")) != 500:
        reasons.append("test_episodes!=500")
    if _parse_int(row.get("num_envs")) != 1:
        reasons.append("num_envs!=1")
    hb = _parse_bool(row.get("headless"))
    if hb is not True:
        reasons.append("headless!=True")
    if (row.get("task") or "").strip() != task:
        reasons.append(f"task!={task}")
    if (row.get("motion_file") or "").strip() != motion_file:
        reasons.append("motion_file_mismatch")
    cmd = row.get("command") or ""
    if cfg_env not in cmd:
        reasons.append("command_missing_cfg_env")
    if cfg_train not in cmd:
        reasons.append("command_missing_cfg_train")
    return (len(reasons) == 0, reasons)


def dedupe_latest(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep latest row per (checkpoint, seed) by end_time/timestamp."""
    sorted_rows = sorted(rows, key=_row_sort_key, reverse=True)
    seen: set[Tuple[str, int]] = set()
    out: List[Dict[str, str]] = []
    for r in sorted_rows:
        ckpt = (r.get("checkpoint") or "").strip()
        seed = _parse_int(r.get("seed"))
        if seed is None:
            continue
        key = (ckpt, seed)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def group_stats(values: Sequence[float]) -> Tuple[int, float, float, float, float, float, float]:
    """n, mean, std, se, halfwidth, ci_low, ci_high for 95% CI (normal approx)."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return (0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n)
    half = Z_95 * se
    return (n, mean, std, se, half, mean - half, mean + half)


def _series_for_seeds(seed_map: Dict[int, float]) -> Optional[List[float]]:
    if not all(s in seed_map for s in EXPECTED_SEEDS):
        return None
    return [seed_map[s] for s in sorted(EXPECTED_SEEDS)]


def paired_mean_ci_p(
    a: Sequence[float], b: Sequence[float]
) -> Tuple[float, float, float, float, float, Optional[float]]:
    """
    Paired differences a - b (same length). Returns mean_d, std_d, se_d, ci_low, ci_high, p_value or None.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = int(aa.size)
    if n == 0 or bb.size != n:
        return (float("nan"),) * 5 + (None,)
    d = aa - bb
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se_d = std_d / math.sqrt(n)
    half = Z_95 * se_d
    p_val: Optional[float] = None
    if ttest_rel is not None and n > 1:
        _, p_val = ttest_rel(aa, bb)
    return (mean_d, std_d, se_d, mean_d - half, mean_d + half, p_val)


@dataclass
class ConditionBlock:
    checkpoint: str
    label: str
    rows: List[Dict[str, str]]
    seed_to_reward: Dict[int, float]
    seed_to_steps: Dict[int, float]
    seed_to_catch_rate: Dict[int, float]
    seed_to_pass_rate: Dict[int, float]


def build_condition_blocks(
    deduped: List[Dict[str, str]],
    *,
    motion_file: str,
    task: str,
    cfg_env: str,
    cfg_train: str,
) -> List[ConditionBlock]:
    by_ckpt: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in deduped:
        by_ckpt[(r.get("checkpoint") or "").strip()].append(r)

    checkpoints = sorted(by_ckpt.keys(), key=milestone_order_key)
    blocks: List[ConditionBlock] = []
    for ckpt in checkpoints:
        rows = by_ckpt[ckpt]
        seed_to_reward: Dict[int, float] = {}
        seed_to_steps: Dict[int, float] = {}
        seed_to_catch_rate: Dict[int, float] = {}
        seed_to_pass_rate: Dict[int, float] = {}
        for r in rows:
            s = _parse_int(r.get("seed"))
            if s is None:
                continue
            ar = _parse_float(r.get("avg_reward"))
            ast = _parse_float(r.get("avg_steps"))
            if ar is not None:
                seed_to_reward[s] = ar
            if ast is not None:
                seed_to_steps[s] = ast
            csr = _parse_float(r.get("catch_success_rate"))
            psr = _parse_float(r.get("pass_success_rate"))
            if csr is not None:
                seed_to_catch_rate[s] = csr
            if psr is not None:
                seed_to_pass_rate[s] = psr
        blocks.append(
            ConditionBlock(
                checkpoint=ckpt,
                label=analysis_condition_label(ckpt),
                rows=sorted(rows, key=lambda x: _parse_int(x.get("seed")) or 0),
                seed_to_reward=seed_to_reward,
                seed_to_steps=seed_to_steps,
                seed_to_catch_rate=seed_to_catch_rate,
                seed_to_pass_rate=seed_to_pass_rate,
            )
        )
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Scientific eval Step 6–7 on inference_runs.csv")
    parser.add_argument(
        "--input",
        default="logs/inference_runs.csv",
        help="Input CSV path (read-only).",
    )
    parser.add_argument(
        "--output",
        default="logs/inference_scientific_eval_augmented.csv",
        help="Output CSV path (must differ from --input).",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "Exact checkpoint path string to include (repeatable). "
            "If set, only these checkpoints are used (replaces default milestone basename filter)."
        ),
    )
    parser.add_argument("--motion-file", default=DEFAULT_MOTION)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--cfg-env", default=DEFAULT_CFG_ENV)
    parser.add_argument("--cfg-train", default=DEFAULT_CFG_TRAIN)
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)
    if in_path == out_path:
        print("ERROR: --output must not equal --input (refusing to overwrite the log).", file=sys.stderr)
        return 2

    if not os.path.isfile(in_path):
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2

    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames
        if not fieldnames_in:
            print("ERROR: empty or invalid CSV header", file=sys.stderr)
            return 2
        all_rows = list(reader)

    allowlist: Optional[set[str]] = None
    if args.checkpoint:
        allowlist = {c.strip() for c in args.checkpoint if c.strip()}

    def include_row(r: Dict[str, str]) -> bool:
        ckpt = (r.get("checkpoint") or "").strip()
        if allowlist is not None:
            return ckpt in allowlist
        return default_checkpoint_filter(ckpt)

    filtered = [r for r in all_rows if include_row(r)]
    deduped = dedupe_latest(filtered)

    blocks = build_condition_blocks(
        deduped,
        motion_file=args.motion_file,
        task=args.task,
        cfg_env=args.cfg_env,
        cfg_train=args.cfg_train,
    )

    # --- Step 6 report ---
    print("=== Step 6: Scientific eval sanity checks (CSV-only) ===")
    print(
        "Note: 'Correct checkpoint loaded' vs runtime log line is NOT verified from this CSV alone.\n"
    )
    if ttest_rel is None:
        print("Note: scipy not installed; pairwise_p_value columns will be empty.\n")

    all_ok = True
    ckpt_to_block = {b.checkpoint: b for b in blocks}

    for block in blocks:
        seeds_found = {_parse_int(r.get("seed")) for r in block.rows}
        seeds_found.discard(None)
        missing = sorted(EXPECTED_SEEDS - seeds_found)
        extra = sorted(seeds_found - EXPECTED_SEEDS)

        row_issues: List[str] = []
        all_rows_ok = True
        for r in block.rows:
            ok, reasons = row_passes_protocol(
                r,
                motion_file=args.motion_file,
                task=args.task,
                cfg_env=args.cfg_env,
                cfg_train=args.cfg_train,
            )
            if not ok:
                all_rows_ok = False
                s = _parse_int(r.get("seed"))
                row_issues.append(f"seed {s}: {', '.join(reasons)}")

        seed_set_ok = not missing and not extra
        condition_ok = seed_set_ok and all_rows_ok
        if not condition_ok:
            all_ok = False

        print(f"Condition: {block.label} ({block.checkpoint})")
        print(f"  Rows (deduped): {len(block.rows)}")
        print(f"  Seeds present: {sorted(seeds_found)}")
        if missing:
            print(f"  MISSING seeds vs {{0..4}}: {missing}")
        if extra:
            print(f"  UNEXPECTED seeds: {extra}")
        if row_issues:
            print("  Row protocol failures:")
            for line in row_issues:
                print(f"    - {line}")
        else:
            print("  All deduped rows pass row-level protocol checks.")
        catch_series = _series_for_seeds(block.seed_to_catch_rate)
        pass_series = _series_for_seeds(block.seed_to_pass_rate)
        print(f"  Secondary (catch/pass) present for all 5 seeds: {catch_series is not None and pass_series is not None}")
        print(f"  condition_step6_ok: {condition_ok}")
        print()

    # Pairwise: build maps for prev and baseline
    ordered_ckpts = [b.checkpoint for b in blocks]
    prev_ckpt: Dict[str, Optional[str]] = {}
    for i, ck in enumerate(ordered_ckpts):
        prev_ckpt[ck] = ordered_ckpts[i - 1] if i > 0 else None

    baseline_ckpt = ordered_ckpts[0] if ordered_ckpts else None

    # Per-condition group stats and pairwise summaries
    group_reward_stats: Dict[str, Tuple[int, float, float, float, float, float, float]] = {}
    group_steps_stats: Dict[str, Tuple[int, float, float, float, float, float, float]] = {}
    group_catch_stats: Dict[str, Tuple[int, float, float, float, float, float, float]] = {}
    group_pass_stats: Dict[str, Tuple[int, float, float, float, float, float, float]] = {}
    for block in blocks:
        rewards = [block.seed_to_reward[s] for s in sorted(block.seed_to_reward) if s in EXPECTED_SEEDS]
        steps = [block.seed_to_steps[s] for s in sorted(block.seed_to_steps) if s in EXPECTED_SEEDS]
        group_reward_stats[block.checkpoint] = group_stats(rewards)
        group_steps_stats[block.checkpoint] = group_stats(steps)
        cs = _series_for_seeds(block.seed_to_catch_rate)
        ps = _series_for_seeds(block.seed_to_pass_rate)
        group_catch_stats[block.checkpoint] = group_stats(cs) if cs is not None else group_stats([])
        group_pass_stats[block.checkpoint] = group_stats(ps) if ps is not None else group_stats([])

    # pairwise vs baseline (CTDE_2000) and vs immediate predecessor
    pairwise_vs_base: Dict[str, Tuple[float, float, float, float, float, Optional[float]]] = {}
    pairwise_vs_prev: Dict[str, Tuple[float, float, float, float, float, Optional[float]]] = {}
    pairwise_catch_vs_base: Dict[str, Tuple[float, float, float, float, float, Optional[float]]] = {}
    pairwise_catch_vs_prev: Dict[str, Tuple[float, float, float, float, float, Optional[float]]] = {}
    pairwise_pass_vs_base: Dict[str, Tuple[float, float, float, float, float, Optional[float]]] = {}
    pairwise_pass_vs_prev: Dict[str, Tuple[float, float, float, float, float, Optional[float]]] = {}

    for block in blocks:
        ck = block.checkpoint
        if baseline_ckpt and ck != baseline_ckpt:
            base_block = ckpt_to_block.get(baseline_ckpt)
            if base_block and all(s in block.seed_to_reward for s in EXPECTED_SEEDS) and all(
                s in base_block.seed_to_reward for s in EXPECTED_SEEDS
            ):
                sa = [block.seed_to_reward[s] for s in sorted(EXPECTED_SEEDS)]
                sb = [base_block.seed_to_reward[s] for s in sorted(EXPECTED_SEEDS)]
                pairwise_vs_base[ck] = paired_mean_ci_p(sa, sb)

        pck = prev_ckpt.get(ck)
        if pck:
            prev_b = ckpt_to_block.get(pck)
            if prev_b and all(s in block.seed_to_reward for s in EXPECTED_SEEDS) and all(
                s in prev_b.seed_to_reward for s in EXPECTED_SEEDS
            ):
                sa = [block.seed_to_reward[s] for s in sorted(EXPECTED_SEEDS)]
                sb = [prev_b.seed_to_reward[s] for s in sorted(EXPECTED_SEEDS)]
                pairwise_vs_prev[ck] = paired_mean_ci_p(sa, sb)

        c_block = _series_for_seeds(block.seed_to_catch_rate)
        p_block = _series_for_seeds(block.seed_to_pass_rate)
        if baseline_ckpt and ck != baseline_ckpt and c_block is not None:
            base_block = ckpt_to_block.get(baseline_ckpt)
            c_base = _series_for_seeds(base_block.seed_to_catch_rate) if base_block else None
            if c_base is not None:
                pairwise_catch_vs_base[ck] = paired_mean_ci_p(c_block, c_base)
        if baseline_ckpt and ck != baseline_ckpt and p_block is not None:
            base_block = ckpt_to_block.get(baseline_ckpt)
            p_base = _series_for_seeds(base_block.seed_to_pass_rate) if base_block else None
            if p_base is not None:
                pairwise_pass_vs_base[ck] = paired_mean_ci_p(p_block, p_base)

        if pck:
            prev_b = ckpt_to_block.get(pck)
            if prev_b is not None and c_block is not None:
                c_prev = _series_for_seeds(prev_b.seed_to_catch_rate)
                if c_prev is not None:
                    pairwise_catch_vs_prev[ck] = paired_mean_ci_p(c_block, c_prev)
            if prev_b is not None and p_block is not None:
                p_prev = _series_for_seeds(prev_b.seed_to_pass_rate)
                if p_prev is not None:
                    pairwise_pass_vs_prev[ck] = paired_mean_ci_p(p_block, p_prev)

    new_columns = [
        "analysis_condition",
        "protocol_row_ok",
        "protocol_row_fail_reasons",
        "condition_step6_ok",
        "group_n_seeds",
        "group_mean_avg_reward",
        "group_std_avg_reward",
        "group_se_avg_reward",
        "group_ci95_halfwidth_avg_reward",
        "group_ci95_low_avg_reward",
        "group_ci95_high_avg_reward",
        "group_mean_avg_steps",
        "group_std_avg_steps",
        "group_se_avg_steps",
        "group_ci95_halfwidth_avg_steps",
        "group_ci95_low_avg_steps",
        "group_ci95_high_avg_steps",
        "paired_diff_avg_reward_vs_prev_milestone",
        "pairwise_mean_delta_vs_first_milestone_reward",
        "pairwise_std_delta_vs_first_milestone_reward",
        "pairwise_se_delta_vs_first_milestone_reward",
        "pairwise_ci95_low_delta_vs_first_milestone_reward",
        "pairwise_ci95_high_delta_vs_first_milestone_reward",
        "pairwise_p_value_vs_first_milestone_reward",
        "pairwise_mean_delta_vs_prev_milestone_reward",
        "pairwise_std_delta_vs_prev_milestone_reward",
        "pairwise_se_delta_vs_prev_milestone_reward",
        "pairwise_ci95_low_delta_vs_prev_milestone_reward",
        "pairwise_ci95_high_delta_vs_prev_milestone_reward",
        "pairwise_p_value_vs_prev_milestone_reward",
        "group_n_seeds_catch_pass",
        "group_mean_catch_success_rate",
        "group_std_catch_success_rate",
        "group_se_catch_success_rate",
        "group_ci95_halfwidth_catch_success_rate",
        "group_ci95_low_catch_success_rate",
        "group_ci95_high_catch_success_rate",
        "group_mean_pass_success_rate",
        "group_std_pass_success_rate",
        "group_se_pass_success_rate",
        "group_ci95_halfwidth_pass_success_rate",
        "group_ci95_low_pass_success_rate",
        "group_ci95_high_pass_success_rate",
        "pairwise_mean_delta_vs_first_milestone_catch_rate",
        "pairwise_std_delta_vs_first_milestone_catch_rate",
        "pairwise_se_delta_vs_first_milestone_catch_rate",
        "pairwise_ci95_low_delta_vs_first_milestone_catch_rate",
        "pairwise_ci95_high_delta_vs_first_milestone_catch_rate",
        "pairwise_p_value_vs_first_milestone_catch_rate",
        "pairwise_mean_delta_vs_prev_milestone_catch_rate",
        "pairwise_std_delta_vs_prev_milestone_catch_rate",
        "pairwise_se_delta_vs_prev_milestone_catch_rate",
        "pairwise_ci95_low_delta_vs_prev_milestone_catch_rate",
        "pairwise_ci95_high_delta_vs_prev_milestone_catch_rate",
        "pairwise_p_value_vs_prev_milestone_catch_rate",
        "pairwise_mean_delta_vs_first_milestone_pass_rate",
        "pairwise_std_delta_vs_first_milestone_pass_rate",
        "pairwise_se_delta_vs_first_milestone_pass_rate",
        "pairwise_ci95_low_delta_vs_first_milestone_pass_rate",
        "pairwise_ci95_high_delta_vs_first_milestone_pass_rate",
        "pairwise_p_value_vs_first_milestone_pass_rate",
        "pairwise_mean_delta_vs_prev_milestone_pass_rate",
        "pairwise_std_delta_vs_prev_milestone_pass_rate",
        "pairwise_se_delta_vs_prev_milestone_pass_rate",
        "pairwise_ci95_low_delta_vs_prev_milestone_pass_rate",
        "pairwise_ci95_high_delta_vs_prev_milestone_pass_rate",
        "pairwise_p_value_vs_prev_milestone_pass_rate",
    ]

    out_fieldnames = list(fieldnames_in) + new_columns
    out_rows: List[Dict[str, str]] = []

    for block in blocks:
        ck = block.checkpoint
        seeds_found = {_parse_int(r.get("seed")) for r in block.rows}
        seeds_found.discard(None)
        cond_ok = seeds_found == EXPECTED_SEEDS and all(
            row_passes_protocol(
                r,
                motion_file=args.motion_file,
                task=args.task,
                cfg_env=args.cfg_env,
                cfg_train=args.cfg_train,
            )[0]
            for r in block.rows
        )

        gr = group_reward_stats[ck]
        gs = group_steps_stats[ck]
        gc = group_catch_stats[ck]
        gp = group_pass_stats[ck]
        pv_base = pairwise_vs_base.get(ck)
        pv_prev = pairwise_vs_prev.get(ck)
        pv_c_base = pairwise_catch_vs_base.get(ck)
        pv_c_prev = pairwise_catch_vs_prev.get(ck)
        pv_p_base = pairwise_pass_vs_base.get(ck)
        pv_p_prev = pairwise_pass_vs_prev.get(ck)

        prev_c = prev_ckpt.get(ck)
        prev_block = ckpt_to_block[prev_c] if prev_c else None

        for r in block.rows:
            row_ok, reasons = row_passes_protocol(
                r,
                motion_file=args.motion_file,
                task=args.task,
                cfg_env=args.cfg_env,
                cfg_train=args.cfg_train,
            )
            seed = _parse_int(r.get("seed"))
            paired_vs_prev = ""
            if prev_block is not None and seed is not None:
                br = prev_block.seed_to_reward.get(seed)
                ar = _parse_float(r.get("avg_reward"))
                if br is not None and ar is not None:
                    paired_vs_prev = str(ar - br)

            def fmt(x: Any) -> str:
                if x is None:
                    return ""
                if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                    return ""
                if isinstance(x, float):
                    return repr(x)
                return str(x)

            def fmt_p(x: Optional[float]) -> str:
                if x is None:
                    return ""
                return repr(x)

            out_row = dict(r)
            out_row["analysis_condition"] = block.label
            out_row["protocol_row_ok"] = str(row_ok)
            out_row["protocol_row_fail_reasons"] = ";".join(reasons)
            out_row["condition_step6_ok"] = str(cond_ok)
            out_row["group_n_seeds"] = str(gr[0])
            out_row["group_mean_avg_reward"] = fmt(gr[1])
            out_row["group_std_avg_reward"] = fmt(gr[2])
            out_row["group_se_avg_reward"] = fmt(gr[3])
            out_row["group_ci95_halfwidth_avg_reward"] = fmt(gr[4])
            out_row["group_ci95_low_avg_reward"] = fmt(gr[5])
            out_row["group_ci95_high_avg_reward"] = fmt(gr[6])
            out_row["group_mean_avg_steps"] = fmt(gs[1])
            out_row["group_std_avg_steps"] = fmt(gs[2])
            out_row["group_se_avg_steps"] = fmt(gs[3])
            out_row["group_ci95_halfwidth_avg_steps"] = fmt(gs[4])
            out_row["group_ci95_low_avg_steps"] = fmt(gs[5])
            out_row["group_ci95_high_avg_steps"] = fmt(gs[6])
            out_row["paired_diff_avg_reward_vs_prev_milestone"] = paired_vs_prev
            if pv_base:
                out_row["pairwise_mean_delta_vs_first_milestone_reward"] = fmt(pv_base[0])
                out_row["pairwise_std_delta_vs_first_milestone_reward"] = fmt(pv_base[1])
                out_row["pairwise_se_delta_vs_first_milestone_reward"] = fmt(pv_base[2])
                out_row["pairwise_ci95_low_delta_vs_first_milestone_reward"] = fmt(pv_base[3])
                out_row["pairwise_ci95_high_delta_vs_first_milestone_reward"] = fmt(pv_base[4])
                out_row["pairwise_p_value_vs_first_milestone_reward"] = fmt_p(pv_base[5])
            else:
                out_row["pairwise_mean_delta_vs_first_milestone_reward"] = ""
                out_row["pairwise_std_delta_vs_first_milestone_reward"] = ""
                out_row["pairwise_se_delta_vs_first_milestone_reward"] = ""
                out_row["pairwise_ci95_low_delta_vs_first_milestone_reward"] = ""
                out_row["pairwise_ci95_high_delta_vs_first_milestone_reward"] = ""
                out_row["pairwise_p_value_vs_first_milestone_reward"] = ""
            if pv_prev:
                out_row["pairwise_mean_delta_vs_prev_milestone_reward"] = fmt(pv_prev[0])
                out_row["pairwise_std_delta_vs_prev_milestone_reward"] = fmt(pv_prev[1])
                out_row["pairwise_se_delta_vs_prev_milestone_reward"] = fmt(pv_prev[2])
                out_row["pairwise_ci95_low_delta_vs_prev_milestone_reward"] = fmt(pv_prev[3])
                out_row["pairwise_ci95_high_delta_vs_prev_milestone_reward"] = fmt(pv_prev[4])
                out_row["pairwise_p_value_vs_prev_milestone_reward"] = fmt_p(pv_prev[5])
            else:
                out_row["pairwise_mean_delta_vs_prev_milestone_reward"] = ""
                out_row["pairwise_std_delta_vs_prev_milestone_reward"] = ""
                out_row["pairwise_se_delta_vs_prev_milestone_reward"] = ""
                out_row["pairwise_ci95_low_delta_vs_prev_milestone_reward"] = ""
                out_row["pairwise_ci95_high_delta_vs_prev_milestone_reward"] = ""
                out_row["pairwise_p_value_vs_prev_milestone_reward"] = ""

            out_row["group_n_seeds_catch_pass"] = str(gc[0]) if gc[0] == gp[0] else ""
            out_row["group_mean_catch_success_rate"] = fmt(gc[1])
            out_row["group_std_catch_success_rate"] = fmt(gc[2])
            out_row["group_se_catch_success_rate"] = fmt(gc[3])
            out_row["group_ci95_halfwidth_catch_success_rate"] = fmt(gc[4])
            out_row["group_ci95_low_catch_success_rate"] = fmt(gc[5])
            out_row["group_ci95_high_catch_success_rate"] = fmt(gc[6])
            out_row["group_mean_pass_success_rate"] = fmt(gp[1])
            out_row["group_std_pass_success_rate"] = fmt(gp[2])
            out_row["group_se_pass_success_rate"] = fmt(gp[3])
            out_row["group_ci95_halfwidth_pass_success_rate"] = fmt(gp[4])
            out_row["group_ci95_low_pass_success_rate"] = fmt(gp[5])
            out_row["group_ci95_high_pass_success_rate"] = fmt(gp[6])

            def _fill_pw(
                prefix: str,
                pv: Optional[Tuple[float, float, float, float, float, Optional[float]]],
            ) -> None:
                if pv:
                    out_row[f"pairwise_mean_delta_{prefix}"] = fmt(pv[0])
                    out_row[f"pairwise_std_delta_{prefix}"] = fmt(pv[1])
                    out_row[f"pairwise_se_delta_{prefix}"] = fmt(pv[2])
                    out_row[f"pairwise_ci95_low_delta_{prefix}"] = fmt(pv[3])
                    out_row[f"pairwise_ci95_high_delta_{prefix}"] = fmt(pv[4])
                    out_row[f"pairwise_p_value_{prefix}"] = fmt_p(pv[5])
                else:
                    out_row[f"pairwise_mean_delta_{prefix}"] = ""
                    out_row[f"pairwise_std_delta_{prefix}"] = ""
                    out_row[f"pairwise_se_delta_{prefix}"] = ""
                    out_row[f"pairwise_ci95_low_delta_{prefix}"] = ""
                    out_row[f"pairwise_ci95_high_delta_{prefix}"] = ""
                    out_row[f"pairwise_p_value_{prefix}"] = ""

            _fill_pw("vs_first_milestone_catch_rate", pv_c_base)
            _fill_pw("vs_prev_milestone_catch_rate", pv_c_prev)
            _fill_pw("vs_first_milestone_pass_rate", pv_p_base)
            _fill_pw("vs_prev_milestone_pass_rate", pv_p_prev)

            out_rows.append(out_row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in out_fieldnames})

    print(f"Wrote {len(out_rows)} rows to {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
