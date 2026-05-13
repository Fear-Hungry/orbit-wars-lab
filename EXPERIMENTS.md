# Experiments
<!-- autoresearch: environment: docker=lab mem=16g shm=2g cpus=4 gpu_profile=lab-gpu-available -->
<!-- autoresearch: metric_direction: lower -->
<!-- autoresearch: mode: loop -->
<!-- autoresearch: run_tag: orbit-wars-competitive-loop -->
<!-- autoresearch: parallel: serial -->
<!-- autoresearch: web_search: disabled -->
<!-- autoresearch: goal: Develop competitive Orbit Wars agents using vault-grounded PSRO/ranking-diversity pressure, Docker-contained validation, and fail-fast promotion gates until a candidate passes hardened objective validation and beats the live champion baseline. -->
<!-- autoresearch: scope: python/**,configs/**,tests/**,scripts/**,AGENTS.md,Makefile,docker-compose.yml,Dockerfile -->
<!-- autoresearch: repos_json: [{"path":"/home/marcusvinicius/Repositorios/Kaggle/orbit-wars-lab","role":"primary","scope":"python/**,configs/**,tests/**,scripts/**,AGENTS.md,Makefile,docker-compose.yml,Dockerfile"}] -->
<!-- autoresearch: metric: promotion_gate_deficit -->
<!-- autoresearch: verify: TRAIN_MEMORY=16g TRAIN_SHM_SIZE=2g TRAIN_CPUS=4.0 docker compose run --rm lab python -m python.train.objective_validation --manifest configs/final_candidate_pool.yaml --selection-config configs/final_selection.yaml --validation-config configs/objective_validation.yaml --out-dir artifacts/final_candidates -->
<!-- autoresearch: guard: TRAIN_MEMORY=16g TRAIN_SHM_SIZE=2g TRAIN_CPUS=4.0 docker compose run --rm lab pytest -q tests/test_heuristics.py tests/test_submission_pipeline.py tests/test_final_selection.py tests/test_objective_validation.py -->
<!-- autoresearch: stop_condition: objective_ready=true and promotion_gate_deficit=0; exported candidate satisfies shipping_blind and live_baseline_challenge gates -->
<!-- autoresearch: rollback_policy: revert -->

| iteration | commit | metric | delta | guard | status | description |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 09e3c7c | 4.625 | 0 | - | baseline | Docker objective_validation baseline after hardened gates: export=opening_gate_rush_meta_candidate; failed worst_decile_ok, shipping_blind_hall_of_fame_ok, shipping_blind_seed_strata_ok, live_baseline_challenge_ok. |
| 1 | f874062 | 3.75 | -0.875 | docker objective_validation exit 0; docker pytest guard 62 passed | keep | [labels: gate-tail, live-baseline, docker-validated] Locked field_control for live-tail rush-meta opening signatures; improved promotion_gate_deficit 4.625000 -> 3.750000, fixing live win/mean thresholds while worst-decile tail remains. |
| 2 | - | 3.75 | 0 | docker parametrized oracle search completed | search | [labels: search, response-oracle, docker-contained] Searched simple tail-expansion response policies over live/shipping 2p tail seeds; best candidate fixed selected greedy collapses and identified safe archetypes for a locked subpolicy, but broad dynamic routing was rejected. |
| 3 | 2b2347f | 3.25 | -0.5 | docker objective_validation exit 0; docker pytest guard 64 passed | keep | [labels: tail-expansion, response-oracle, docker-validated] Added tail_expansion response gate for low-spin cheap/rich far and clustered opening archetypes; promotion_gate_deficit improved 3.750000 -> 3.250000 and fixed shipping blind low_spin:cheap_near_clustered tail while remaining deficits are high_spin:rich_cheap_far and low_spin:rich_heavy_far/close tails. |
| 4 | 0ce1981 | 2.75 | -0.5 | docker objective_validation exit 0; docker pytest guard 65 passed | keep | [labels: high-spin-tail, shipping-blind, docker-validated] Routed high-spin rich cheap far player-1 openings to tail_expansion; promotion_gate_deficit improved 3.250000 -> 2.750000 and shipping blind high_spin:rich_cheap_far now passes in 2p, while remaining blockers are low_spin:rich_heavy_far/close live and shipping tails plus retained-selection worst decile. |
| 5 | - | 2.75 | 0 | docker style cross-search completed | search | [labels: search, low-spin-tail, docker-contained] Searched existing opening style substitutions for remaining low_spin:rich_heavy_far/close tails. close_p0=rush_then_field_control_one removed the live 1301 p0 loss in probe, but did not address live/shipping worst-decile blockers; low_far substitutions added or worsened 1201 losses. No code change retained. |
