# Documentação — Orbit Wars Lab

Esta pasta é a **fonte de verdade** do projeto. Cada documento cobre um tópico e
não se repete nos outros; quando um assunto aparece em mais de um lugar, o dono
canônico está marcado abaixo e os demais apenas referenciam.

## Ordem de leitura sugerida

Para entender o projeto do zero (ex.: leitor de portfólio):

1. [`../README.md`](../README.md) — visão geral, problema e instalação.
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) — o modelo de três camadas (Rust simula, Python decide, submissão leve) e a fronteira que o sustenta.
3. [`DECISIONS.md`](DECISIONS.md) — por que cada escolha de arquitetura foi feita (D1–D11).
4. [`BLUEPRINT.md`](BLUEPRINT.md) — o método competitivo: ciclo de geração, espaço de ação, fórmula de fitness, critérios de promoção, antipadrões.
5. [`PARITY.md`](PARITY.md) — como garantimos que o motor local é fiel ao ambiente oficial do Kaggle.
6. [`COMPETITIVE_INTEL.md`](COMPETITIVE_INTEL.md) — pesquisa externa (fórum/notebooks) que orienta as hipóteses.
7. [`TRAINING.md`](TRAINING.md) — plano de treino por fases e a decisão atual sobre PPO.
8. [`SUBMISSION.md`](SUBMISSION.md) — regra e checklist do agente que vai para o Kaggle.
9. [`PLAYBOOK.md`](PLAYBOOK.md) — comandos operacionais para reproduzir tudo.
10. [`SUBMISSION_AUDIT_2026-06-11.md`](SUBMISSION_AUDIT_2026-06-11.md) —
    auditoria dos bugs silenciosos e pacotes top-score antes de nova submissão.

## Quem é dono de quê (uma fonte de verdade por tópico)

| Tópico | Dono canônico | Os outros docs apenas referenciam |
| --- | --- | --- |
| Fronteira Rust/Python e por que a submissão não usa Rust | `DECISIONS.md` (D10/D11) | `ARCHITECTURE.md`, `SUBMISSION.md` |
| Constantes e regras do ambiente oficial | `PARITY.md` | `../README.md` |
| Ciclo de treino, fitness e promoção | `BLUEPRINT.md` | `TRAINING.md` |
| Estado atual do PPO (deferido) | `TRAINING.md` | `DECISIONS.md` (D4) |
| "Producer é piso, não teto" | `COMPETITIVE_INTEL.md` | `BLUEPRINT.md`, `PLAYBOOK.md` |
| Comandos de benchmark e gates | `PLAYBOOK.md` | `../README.md` |
| Auditoria de submissoes top-score | `SUBMISSION_AUDIT_2026-06-11.md` | `SUBMISSION.md`, `PLAYBOOK.md` |

## Logs (não são fonte de verdade curada)

Ficam na raiz e são **registros append-only**, mantidos para continuidade — não
para um leitor de portfólio:

- `experiments.duckdb` — store de ablações (status feito/rejeitado/todo); relatório em [`EXPERIMENTS_REPORT.md`](EXPERIMENTS_REPORT.md).
- [`../todo.md`](../todo.md) — journal de trabalho em andamento (roadmap, threads, checkboxes).
- [`../AGENTS.md`](../AGENTS.md) — instruções para o executor de IA (harness Codex), não documentação do projeto.
