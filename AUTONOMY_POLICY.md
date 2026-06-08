# AUTONOMY POLICY

> A **constituição de execução** completa é dona de [`AGENTS.md`](AGENTS.md); a política de
> compute (GPU livre p/ treino, inferência/submissão CPU-only) é dona de
> [`CLAUDE.md`](CLAUDE.md). Esta página é o **resumo de fronteira de autonomia** que o loop
> (`.claude/loop.md`) lê antes de agir. Em conflito, `AGENTS.md`/`CLAUDE.md` mandam.

**Modo: autonomia com portões (gated-autonomy), Nível 2.5.**

O loop **pode**:
- ler descrição pública da competição, fóruns públicos (discussão, não código/pesos),
  arquivos locais, configs e logs;
- propor ideias de experimento (em `loop_queue.md ## proposed` e acompanhando `todo.md`);
- implementar **um** experimento por iteração;
- rodar validação local (margem pareada vs Producer), `make gate-check-final` e checks de invariantes;
- gerar candidatos de submissão e atualizar `EXPERIMENTS.md`/`loop_queue.md`.

O loop **não pode**:
- usar dados externos, pesos pré-treinados ou código de fórum a menos que `AGENTS.md`/`CLAUDE.md` permitam explicitamente;
- chamar `kaggle competitions submit` diretamente — o **único** caminho é `scripts/auto_submit_gate.py`;
- criar múltiplas contas, burlar limites ou automatizar contornos de restrição;
- otimizar só para o public LB sem suporte de margem pareada local;
- mexer em lógica de modelo e pipeline de submissão no **mesmo** experimento;
- depender de CUDA em qualquer coisa de `bots/`/`artifacts/` (D10/D11).

**Auto-submit** só ocorre via `scripts/auto_submit_gate.py` com `AUTO_SUBMIT=1` E todos os
portões de [`SUBMISSION_POLICY.md`](SUBMISSION_POLICY.md) aprovando. Default `AUTO_SUBMIT=0`
(dry-run). Teto: `MAX_AUTO_SUBMISSIONS_PER_DAY=1`; o resto das 5/dia é decisão humana.
