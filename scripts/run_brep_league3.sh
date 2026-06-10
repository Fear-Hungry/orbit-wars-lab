#!/usr/bin/env bash
# League v3 — informative-pool redesign after league/league2 stayed at the KEEP
# floor for 4 chunks. Root cause (measured): 3/6 league opponents (rush/anti_meta/
# greedy) are crushed by the Producer floor at saturated +1.0 margin, so half the
# batch carries no learning signal and the argmax never flips (KEEP logit gap fell
# only 4.58->4.09 over 3 chunks; warm-started chunk at ent 0.01 also stalled —
# value head fitted to floor play gives near-zero advantages). PFSP/AlphaStar
# principle: train mostly vs ~50% win-rate opponents. So the TRAINING pool is the
# PROVEN v1 producer-family curriculum with greedy (saturated) swapped for the
# frozen champion brep:c05 (the only opponent ABOVE the floor = informative
# self-play pressure), and the run is FRESH (the escape-from-floor path proven by
# both v1 and v2). Field robustness stays in the GATE: per-chunk panel
# (producer/rush/anti_meta/greedy) + survival metrics, final eval panel + OEP.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/ppo/brep_league3
mkdir -p "$OUT"
CHAMP=artifacts/ppo/brep_gpu/c05.pt
OPP="producer_h30,producer_h50,producer_h70,producer,brep:$CHAMP"
GATE_OPPS="producer rush anti_meta greedy"
FLOOR_PANEL="0.7499736686889114"
PREV=""
BEST=""; BEST_M="$FLOOR_PANEL"
for c in 0 1 2 3 4 5; do
  CKPT="$OUT/c0${c}.pt"
  INIT="--policy-arch producer_residual"
  [ -n "$PREV" ] && INIT="$INIT --checkpoint-in $PREV"
  echo "=== TRAIN chunk $c (league3 fresh, pool informativa) ===" | tee -a "$OUT/trajectory.log"
  uv run --no-sync --extra dev --extra train python -m python.train.train_ppo \
    $INIT --reward-mode relative_margin --opponents "$OPP" \
    --rollout-num-envs 16 --rollout-steps 256 --device cuda --ent-coef 0.01 \
    --total-timesteps 75000 --seed "$c" --checkpoint-out "$CKPT" \
    > "$OUT/train_c0${c}.log" 2>&1
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
if [ -z "$BEST" ]; then
  echo "=== DONE: NENHUM chunk superou o painel do piso ($FLOOR_PANEL) — sem best.pt ===" | tee -a "$OUT/trajectory.log"
  exit 0
fi
echo "=== BEST chunk: $BEST (panel avg $BEST_M) — final eval ===" | tee -a "$OUT/trajectory.log"
cp "$BEST" "$OUT/best.pt"
uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent producer --seeds 96 --episode-steps 500 --device cpu --out "$OUT/final_vs_producer96.json" > "$OUT/final_prod.log" 2>&1
echo "FINAL vs Producer(96): $(grep -E 'mean_score_margin|seat0|seat1|annihilated|survival' "$OUT/final_vs_producer96.json")" | tee -a "$OUT/trajectory.log"
for opp in rush anti_meta greedy oep; do
  uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct --checkpoint "$BEST" --opponent "$opp" --seeds 32 --episode-steps 500 --device cpu --out "$OUT/final_vs_${opp}32.json" > "$OUT/final_${opp}.log" 2>&1
  echo "FINAL vs ${opp}(32): $(grep -E 'mean_score_margin|seat0|seat1|annihilated|survival' "$OUT/final_vs_${opp}32.json")" | tee -a "$OUT/trajectory.log"
done
echo "=== DONE (best=$BEST) ===" | tee -a "$OUT/trajectory.log"
