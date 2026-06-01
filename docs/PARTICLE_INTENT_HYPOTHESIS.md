# Particle-Intent Hypothesis

## Hypothesis

Orbit Wars has fully observed planet and fleet positions, so a particle filter is not useful for
physical target localization. It is useful as a lightweight belief model over opponent intent in
four-player games, where multiple opponents can simultaneously expand, pressure us, or attack the
leader.

The tested model uses a small discrete particle set:

- `expand`: observed enemy fleets are likely aimed at neutral planets.
- `pressure`: observed enemy fleets are likely aimed at our planets.
- `leader`: observed enemy fleets are likely aimed at the current economic leader.
- `economy`: observed enemy fleets are likely aimed at high-production neutral planets.

The submission keeps these beliefs per player in module state and updates them from visible fleet
rays each turn. The policy uses the belief only in FFA games. In 1v1, the policy intentionally
falls back to the previous baseline behavior.

## Literature Support

- Arulampalam et al. (2002), "A tutorial on particle filters for online nonlinear/non-Gaussian
  Bayesian tracking", DOI `10.1109/78.978374`.
- Greco and Vasile (2022), "Robust Bayesian Particle Filter for Space Object Tracking Under
  Severe Uncertainty", DOI `10.2514/1.G006157`.
- Ito and Godsill (2020), "A Multi-Target Track-Before-Detect Particle Filter Using
  Superpositional Data in Non-Gaussian Noise", DOI `10.1109/LSP.2020.3002704`.
- Reid (1979), "An algorithm for tracking multiple targets", DOI `10.1109/TAC.1979.1102177`.
- Fortmann, Bar-Shalom, and Scheffe (1983), "Sonar tracking of multiple targets using joint
  probabilistic data association", DOI `10.1109/JOE.1983.1145560`.

The Zotero collection `Orbit Wars` contains these references or related tracking papers. Ito and
Godsill (2020) was added during this experiment because it directly supports track-before-detect
style inference without a hard prior detection threshold.

## Validation

Protocol:

- Baseline artifact: `/tmp/orbit_pf_baseline.py`.
- Variant artifact: `/tmp/orbit_pf_variant3.py`.
- Seeds: `0..7`.
- Episode steps: `500`.
- Comets: enabled.
- Opponent pool: `greedy`, `defensive`, `rush`, `anti_meta`, `weak_random`.

Results:

| Format | Metric | Baseline | Variant |
| --- | ---: | ---: | ---: |
| 2p average | win rate | 0.6625 | 0.6625 |
| 2p average | score margin | 0.3271 | 0.3271 |
| 4p | win rate | 0.5000 | 0.6250 |
| 4p | score margin | 0.0497 | 0.4070 |
| all | invalid action rate | 0.0000 | 0.0000 |

Decision: the variant is eligible for submission because it preserves 2p local performance while
improving the local 4p result under the final comparison protocol.
