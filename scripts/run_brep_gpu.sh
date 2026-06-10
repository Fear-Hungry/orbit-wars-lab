#!/usr/bin/env bash
# BReP (producer_residual) GPU run: cumulative chunks + direct eval per chunk.
# Trains the batchable residual-over-Producer arch with relative_margin on a
# handicapped-Producer ladder; evals vs full Producer at 500 steps both seats
# (no submission export). KEEP-init = parity floor, so margin should start ~0
# and climb if the residual edits beat Producer.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/ppo/brep_gpu
mkdir -p "$OUT"
OPP="producer_h30,producer_h50,producer_h70,producer,greedy"
CHUNK_TS=75000
PREV=""
for c in 0 1 2 3 4 5; do
  CKPT="$OUT/c0${c}.pt"
  if [ -z "$PREV" ]; then
    INIT="--policy-arch producer_residual"
  else
    INIT="--policy-arch producer_residual --checkpoint-in $PREV"
  fi
  echo "=== TRAIN chunk $c (cumulative) ==="
  uv run --no-sync --extra dev --extra train python -m python.train.train_ppo \
    $INIT --reward-mode relative_margin --opponents "$OPP" \
    --rollout-num-envs 16 --rollout-steps 256 --device cuda --ent-coef 0.01 \
    --total-timesteps "$CHUNK_TS" --seed "$c" --checkpoint-out "$CKPT" \
    > "$OUT/train_c0${c}.log" 2>&1
  echo "=== EVAL chunk $c (vs Producer, 16 seeds, both seats, 500 steps) ==="
  uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct \
    --checkpoint "$CKPT" --seeds 16 --episode-steps 500 --device cpu \
    --out "$OUT/eval_c0${c}.json" > "$OUT/eval_c0${c}.log" 2>&1
  M=$(grep mean_score_margin "$OUT/eval_c0${c}.json")
  echo "chunk $c done: $M" | tee -a "$OUT/trajectory.log"
  PREV="$CKPT"
done
echo "=== BReP GPU run complete ==="
