#!/usr/bin/env bash
# Experiment #1: LEAGUE/self-play (AlphaStar/PSRO-style mixture). Lesson from the
# PGS leaderboard failure (local +0.218 vs Producer, LB 1001.7 << 1228, annihilated
# by rushers): training AND gating only vs Producer overfits the gate opponent.
# So: (a) the training pool mixes styles — Producer (economic), the frozen BReP
# champion c05 (self-play), rush (the LB killer), anti_meta, greedy, producer_h50;
# (b) the per-chunk gate is a FIELD PANEL average (producer+rush+anti_meta+greedy),
# not the vs-Producer margin. Fresh train (continued-from-c05 drifts, Exp #3) with
# ent-coef 0.003 (0.01 pushed n_edit=6 off the KEEP floor early in the v2 run).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/ppo/brep_league
mkdir -p "$OUT"
CHAMP=artifacts/ppo/brep_gpu/c05.pt
OPP="producer,brep:$CHAMP,rush,anti_meta,greedy,producer_h50"
GATE_OPPS="producer rush anti_meta greedy"
PREV=""
BEST=""; BEST_M="-99"
for c in 0 1 2 3 4 5; do
  CKPT="$OUT/c0${c}.pt"
  INIT="--policy-arch producer_residual"
  [ -n "$PREV" ] && INIT="$INIT --checkpoint-in $PREV"
  echo "=== TRAIN chunk $c (league) ===" | tee -a "$OUT/trajectory.log"
  uv run --no-sync --extra dev --extra train python -m python.train.train_ppo \
    $INIT --reward-mode relative_margin --opponents "$OPP" \
    --rollout-num-envs 16 --rollout-steps 256 --device cuda --ent-coef 0.003 \
    --total-timesteps 75000 --seed "$c" --checkpoint-out "$CKPT" \
    > "$OUT/train_c0${c}.log" 2>&1
  # Field-panel gate: average margin across styles, 12 seeds each, 500 steps, both seats.
  PANEL_SUM=0; PANEL_N=0; PANEL_MIN=99
  for opp in $GATE_OPPS; do
    uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct \
      --checkpoint "$CKPT" --opponent "$opp" --seeds 12 --episode-steps 500 \
      --device cpu --out "$OUT/eval_c0${c}_${opp}.json" > "$OUT/eval_c0${c}_${opp}.log" 2>&1
    M=$(python3 -c "import json;print(json.load(open('$OUT/eval_c0${c}_${opp}.json'))['mean_score_margin'])")
    echo "chunk $c vs ${opp}(12): $M" | tee -a "$OUT/trajectory.log"
    PANEL_SUM=$(python3 -c "print($PANEL_SUM + $M)")
    PANEL_N=$((PANEL_N + 1))
    PANEL_MIN=$(python3 -c "print(min($PANEL_MIN, $M))")
  done
  AVG=$(python3 -c "print($PANEL_SUM / $PANEL_N)")
  echo "chunk $c PANEL avg=$AVG min=$PANEL_MIN" | tee -a "$OUT/trajectory.log"
  if python3 -c "import sys;sys.exit(0 if float('$AVG')>float('$BEST_M') else 1)"; then BEST_M="$AVG"; BEST="$CKPT"; fi
  PREV="$CKPT"
done
echo "=== BEST chunk: $BEST (panel avg $BEST_M) — final eval ===" | tee -a "$OUT/trajectory.log"
cp "$BEST" "$OUT/best.pt"
uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent producer --seeds 96 --episode-steps 500 --device cpu --out "$OUT/final_vs_producer96.json" > "$OUT/final_prod.log" 2>&1
echo "FINAL vs Producer(96): $(grep -E 'mean_score_margin|seat0|seat1' "$OUT/final_vs_producer96.json")" | tee -a "$OUT/trajectory.log"
for opp in rush anti_meta greedy oep; do
  uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent "$opp" --seeds 32 --episode-steps 500 --device cpu --out "$OUT/final_vs_${opp}32.json" > "$OUT/final_${opp}.log" 2>&1
  echo "FINAL vs ${opp}(32): $(grep -E 'mean_score_margin|seat0|seat1' "$OUT/final_vs_${opp}32.json")" | tee -a "$OUT/trajectory.log"
done
echo "=== DONE (best=$BEST) ===" | tee -a "$OUT/trajectory.log"
