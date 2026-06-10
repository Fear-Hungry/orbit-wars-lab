#!/usr/bin/env bash
# Passo 1 (workflow ppo-explore): isolation ablation shares-Φ vs margin-Φ.
# Same selector init (b4_prodinit_60k.pt, at parity, entropy collapsed to 0.129),
# same seed/opponents/budget — ONLY reward_mode differs. Diagnostic: does the
# opponent-relative potential break the Producer-parity ceiling?
set -euo pipefail
cd "$(dirname "$0")/.."

INIT="artifacts/ppo/frente_b/b4_prodinit_60k.pt"
COMMON=(--init "$INIT"
        --opponents "producer,greedy"
        --eval-opponents "producer"
        --chunks 4 --chunk-timesteps 50000
        --rollout-num-envs 1 --rollout-steps 256
        --ent-coef 0.01
        --eval-seeds 16 --eval-episode-steps 500
        --device cpu --min-free-gb 4.0)

echo "=== TREATMENT: relative_margin (margin-Φ) ==="
uv run --no-sync --extra dev --extra train python -m scripts.run_campaign \
  "${COMMON[@]}" --reward-mode relative_margin \
  --out-dir artifacts/ppo/ablation_relmargin/treatment

echo "=== CONTROL: dense_potential (shares-Φ) ==="
uv run --no-sync --extra dev --extra train python -m scripts.run_campaign \
  "${COMMON[@]}" --reward-mode dense_potential \
  --out-dir artifacts/ppo/ablation_relmargin/control

echo "=== ABLATION DONE ==="
