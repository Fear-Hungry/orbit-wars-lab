#!/usr/bin/env bash
# Experiment: continue BReP from c05 with SEAT-BALANCED training (collector now
# alternates the agent's seat) — tests if fixing the measured player-1 weakness
# (vs OEP seat0 +0.06 / seat1 -0.13) improves the OEP matchup + seat symmetry.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/ppo/brep_seat
mkdir -p "$OUT"
OPP="producer_h30,producer_h50,producer_h70,producer,greedy"
PREV="artifacts/ppo/brep_gpu/c05.pt"
for c in 0 1 2 3; do
  CKPT="$OUT/c0${c}.pt"
  echo "=== TRAIN chunk $c (seat-balanced, from $PREV) ==="
  uv run --no-sync --extra dev --extra train python -m python.train.train_ppo \
    --policy-arch producer_residual --checkpoint-in "$PREV" \
    --reward-mode relative_margin --opponents "$OPP" \
    --rollout-num-envs 16 --rollout-steps 256 --device cuda --ent-coef 0.01 \
    --total-timesteps 75000 --seed "$c" --checkpoint-out "$CKPT" \
    > "$OUT/train_c0${c}.log" 2>&1
  uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct \
    --checkpoint "$CKPT" --opponent producer --seeds 16 --episode-steps 500 \
    --device cpu --out "$OUT/eval_prod_c0${c}.json" > "$OUT/eval_prod_c0${c}.log" 2>&1
  echo "chunk $c vs Producer(16): $(grep mean_score_margin "$OUT/eval_prod_c0${c}.json")" | tee -a "$OUT/trajectory.log"
  PREV="$CKPT"
done
BEST="$OUT/c03.pt"
echo "=== FINAL eval BEST ($BEST) ==="
uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent producer --seeds 96 --episode-steps 500 --device cpu --out "$OUT/final_vs_producer96.json" > "$OUT/final_prod.log" 2>&1
echo "FINAL vs Producer(96): $(grep -E 'mean_score_margin|seat0|seat1' "$OUT/final_vs_producer96.json")" | tee -a "$OUT/trajectory.log"
uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent oep --seeds 32 --episode-steps 500 --device cpu --out "$OUT/final_vs_oep32.json" > "$OUT/final_oep.log" 2>&1
echo "FINAL vs OEP(32): $(grep -E 'mean_score_margin|seat0|seat1' "$OUT/final_vs_oep32.json")" | tee -a "$OUT/trajectory.log"
echo "=== DONE ==="
