#!/usr/bin/env bash
# League v4 — PGS variants in the pool + league-based champion selection.
#
# (1) Pool: the proven informative core (producer_h50/h70/producer + frozen c05)
#     with pgs_hold / pgs_wave_s100 introduced GRADUALLY (chunks 2+ / 4+), so the
#     early win-rate stays ~50% and the argmax escapes the KEEP floor first
#     (league1 lesson: saturated/too-hard opponents dilute the gradient and the
#     run never leaves the floor). PGS knobs are byte-identical to the main
#     worktree's league factories; measured cost ~23ms/call in our engine.
# (2) Champion selection happens in the MAIN worktree's continuous league
#     (BT rating, promotion = P(>=ref) >= 0.6 and above the previous B champion)
#     — NOT by local margin vs Producer (id=129 lesson). The local field panel
#     below only (a) tracks the trajectory and (b) picks WHICH checkpoint to
#     export; a floor clone (producer margin ~0) is never exported.
# (3) Clean split: B trains and drops a tarball in the main league's inbox;
#     the league scores it. No league artifacts are copied back into B.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/ppo/brep_league4
mkdir -p "$OUT"
CHAMP=artifacts/ppo/brep_gpu/c05.pt
CORE="producer_h50,producer_h70,producer,brep:$CHAMP"
GATE_OPPS="producer pgs_hold rush anti_meta greedy"
LEAGUE_INBOX="$HOME/projects/Kaggle/orbit-wars-lab/artifacts/league/tarballs"
PREV=""
BEST=""; BEST_M="-99"; BEST_PROD="0"
for c in 0 1 2 3 4 5; do
  CKPT="$OUT/c0${c}.pt"
  INIT="--policy-arch producer_residual"
  [ -n "$PREV" ] && INIT="$INIT --checkpoint-in $PREV"
  OPP="$CORE"
  [ "$c" -ge 2 ] && OPP="$OPP,pgs_hold"
  [ "$c" -ge 4 ] && OPP="$OPP,pgs_wave_s100"
  echo "=== TRAIN chunk $c (league4, pool=$OPP) ===" | tee -a "$OUT/trajectory.log"
  uv run --no-sync --extra dev --extra train python -m python.train.train_ppo \
    $INIT --reward-mode relative_margin --opponents "$OPP" \
    --rollout-num-envs 16 --rollout-steps 256 --device cuda --ent-coef 0.01 \
    --total-timesteps 75000 --seed "$c" --checkpoint-out "$CKPT" \
    > "$OUT/train_c0${c}.log" 2>&1
  PANEL_SUM=0; PANEL_N=0; PANEL_MIN=99; PROD_M=0
  for opp in $GATE_OPPS; do
    uv run --no-sync --extra dev --extra train python -m scripts.eval_brep_direct \
      --checkpoint "$CKPT" --opponent "$opp" --seeds 12 --episode-steps 500 \
      --device cpu --out "$OUT/eval_c0${c}_${opp}.json" > "$OUT/eval_c0${c}_${opp}.log" 2>&1
    M=$(python3 -c "import json;print(json.load(open('$OUT/eval_c0${c}_${opp}.json'))['mean_score_margin'])")
    echo "chunk $c vs ${opp}(12): $M" | tee -a "$OUT/trajectory.log"
    [ "$opp" = "producer" ] && PROD_M="$M"
    PANEL_SUM=$(python3 -c "print($PANEL_SUM + $M)")
    PANEL_N=$((PANEL_N + 1))
    PANEL_MIN=$(python3 -c "print(min($PANEL_MIN, $M))")
  done
  AVG=$(python3 -c "print($PANEL_SUM / $PANEL_N)")
  echo "chunk $c PANEL avg=$AVG min=$PANEL_MIN" | tee -a "$OUT/trajectory.log"
  if python3 -c "import sys;sys.exit(0 if float('$AVG')>float('$BEST_M') else 1)"; then
    BEST_M="$AVG"; BEST="$CKPT"; BEST_PROD="$PROD_M"; BEST_NAME="c0${c}"
  fi
  PREV="$CKPT"
done
# Export the panel-best checkpoint to the main league's inbox UNLESS it is a
# floor clone (producer margin ~0 => it IS Producer; the league already has one).
if python3 -c "import sys;sys.exit(0 if float('$BEST_PROD') > 0.02 else 1)"; then
  cp "$BEST" "$OUT/best.pt"
  TAR="$LEAGUE_INBOX/brep_league4_${BEST_NAME}.tar.gz"
  mkdir -p "$LEAGUE_INBOX"
  PYTHONPATH=. .venv/bin/python scripts/package_brep_submission.py \
    --checkpoint "$BEST" --out "$TAR" --main-out "$OUT/brep_main.py" \
    > "$OUT/package.log" 2>&1
  echo "=== EXPORTADO p/ liga: $TAR (panel $BEST_M, vs producer $BEST_PROD) ===" | tee -a "$OUT/trajectory.log"
else
  echo "=== SEM EXPORT: best=$BEST e floor-clone (vs producer $BEST_PROD) ===" | tee -a "$OUT/trajectory.log"
fi
echo "=== DONE (best=$BEST) ===" | tee -a "$OUT/trajectory.log"
