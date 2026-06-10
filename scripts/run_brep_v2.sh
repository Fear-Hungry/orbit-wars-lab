#!/usr/bin/env bash
# Experiment #2: richer edit set (n_edit=6, finer ship-scale {KEEP,CANCEL,x0.25,
# x0.5,x1.5,x2.0}). FRESH train (KEEP-init parity floor) — avoids the continued-from-
# c05 drift seen in the seat experiment. Eval-gated: final eval on the BEST chunk.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/ppo/brep_v2
mkdir -p "$OUT"
OPP="producer_h30,producer_h50,producer_h70,producer,greedy"
PREV=""
BEST=""; BEST_M="-99"
for c in 0 1 2 3 4 5; do
  CKPT="$OUT/c0${c}.pt"
  INIT="--policy-arch producer_residual"
  [ -n "$PREV" ] && INIT="$INIT --checkpoint-in $PREV"
  echo "=== TRAIN chunk $c (fresh richer-edit) ==="
  uv run --no-sync --extra dev --extra train python -m python.train.train_ppo \
    $INIT --reward-mode relative_margin --opponents "$OPP" \
    --rollout-num-envs 16 --rollout-steps 256 --device cuda --ent-coef 0.01 \
    --total-timesteps 75000 --seed "$c" --checkpoint-out "$CKPT" \
    > "$OUT/train_c0${c}.log" 2>&1
  uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct \
    --checkpoint "$CKPT" --opponent producer --seeds 16 --episode-steps 500 \
    --device cpu --out "$OUT/eval_c0${c}.json" > "$OUT/eval_c0${c}.log" 2>&1
  M=$(python3 -c "import json;print(json.load(open('$OUT/eval_c0${c}.json'))['mean_score_margin'])")
  echo "chunk $c vs Producer(16): $M" | tee -a "$OUT/trajectory.log"
  if python3 -c "import sys;sys.exit(0 if float('$M')>float('$BEST_M') else 1)"; then BEST_M="$M"; BEST="$CKPT"; fi
  PREV="$CKPT"
done
echo "=== BEST chunk: $BEST (16-seed margin $BEST_M) — final eval ===" | tee -a "$OUT/trajectory.log"
cp "$BEST" "$OUT/best.pt"
uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent producer --seeds 96 --episode-steps 500 --device cpu --out "$OUT/final_vs_producer96.json" > "$OUT/final_prod.log" 2>&1
echo "FINAL vs Producer(96): $(grep -E 'mean_score_margin|seat0|seat1' "$OUT/final_vs_producer96.json")" | tee -a "$OUT/trajectory.log"
uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent oep --seeds 32 --episode-steps 500 --device cpu --out "$OUT/final_vs_oep32.json" > "$OUT/final_oep.log" 2>&1
echo "FINAL vs OEP(32): $(grep -E 'mean_score_margin|seat0|seat1' "$OUT/final_vs_oep32.json")" | tee -a "$OUT/trajectory.log"
echo "=== DONE (best=$BEST) ==="
