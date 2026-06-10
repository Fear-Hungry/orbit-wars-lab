#!/usr/bin/env bash
# League CONTINUATION (chunks 3-6) after the ent-coef course-correction.
# Measured on c00-c02 (ent 0.003): KEEP logit gap shrank only ~0.24/chunk
# (4.58 -> 4.26 -> 4.09 vs init bias 5.0), 0% argmax flips — the greedy policy
# would stay at the parity floor for the whole run. 0.01 flipped the argmax by
# chunk 1 in both prior runs (v1 dip-recover to +0.18 triage). So: resume from
# c02 (value head already trained, EV~0.88) with ent 0.01 and 4 chunks of
# headroom for the dip-recover cycle. Same league pool + field-panel gate.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/ppo/brep_league
mkdir -p "$OUT"
CHAMP=artifacts/ppo/brep_gpu/c05.pt
OPP="producer,brep:$CHAMP,rush,anti_meta,greedy,producer_h50"
GATE_OPPS="producer rush anti_meta greedy"
PREV="$OUT/c02.pt"
# Floor panel (c00-c02 all at KEEP floor) = 0.74997; only a chunk that BEATS the
# floor panel becomes BEST. If none does, best.pt is not written and the run
# reports floor — do not fall back to a floor checkpoint as "best".
BEST=""; BEST_M="0.7499736686889114"
for c in 3 4 5 6; do
  CKPT="$OUT/c0${c}.pt"
  echo "=== TRAIN chunk $c (league cont, ent 0.01) ===" | tee -a "$OUT/trajectory.log"
  uv run --no-sync --extra dev --extra train python -m python.train.train_ppo \
    --policy-arch producer_residual --checkpoint-in "$PREV" \
    --reward-mode relative_margin --opponents "$OPP" \
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
  echo "=== DONE: NENHUM chunk superou o painel do piso ($BEST_M) — sem best.pt ===" | tee -a "$OUT/trajectory.log"
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
