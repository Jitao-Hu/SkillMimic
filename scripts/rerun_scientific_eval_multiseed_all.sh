#!/usr/bin/env bash
# Re-run full scientific-eval inference matrix (500 episodes × seeds 0–4) so
# logs/inference_runs.csv picks up catch/pass secondary metrics from the player.
#
# Run from repository root. Requires GPU + Isaac Gym (same as training).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CTDE_2000="output/CTDE_2000_20260331-13-48-07/nn/CTDE_2000.pth"
CTDE_5000="output/CTDE_5000_20260401-13-07-32/nn/CTDE_5000.pth"
CTDE_8000="output/CTDE_8000_20260402-09-45-21/nn/CTDE_8000.pth"
DUAL_CKPT="${DUAL_CKPT:-output/SkillMimicDualHRL_20260319-15-53-47/nn/SkillMimicDualHRL.pth}"

for ck in "$CTDE_2000" "$CTDE_5000" "$CTDE_8000"; do
  [[ -f "$ck" ]] || { echo "Skip missing: $ck" >&2; continue; }
  echo "========== CTDE $(basename "$(dirname "$(dirname "$ck")")") =========="
  CUDA_VISIBLE_DEVICES="$GPU" ./run_multiseed.sh --ckpt "$ck" --algo ctde --gpu "$GPU"
done

if [[ "${RUN_DUAL_BASELINE:-0}" == "1" ]]; then
  [[ -f "$DUAL_CKPT" ]] || { echo "DUAL_CKPT missing: $DUAL_CKPT" >&2; exit 1; }
  echo "========== HRL-DUAL baseline =========="
  CUDA_VISIBLE_DEVICES="$GPU" ./run_multiseed.sh --ckpt "$DUAL_CKPT" --algo dual --gpu "$GPU"
fi

echo "Done. Regenerate augmented CSV:"
echo "  conda run -n skillmimic python scripts/analyze_inference_scientific_eval.py"
