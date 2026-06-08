# Loop queue

Machine-owned work queue for the `/loop` agent (see `.claude/loop.md`).
Human backlog lives in `todo.md` — do not duplicate it here.

## ready

<!-- At most ONE item is executed per loop. Each item should already have a
     deterministic verification path and not violate AGENTS.md/CLAUDE.md. -->

## proposed

<!-- New ideas land here. Each MUST include all six fields below. -->
<!--
- [ ] <short title>
  - hypothesis: <what you expect to be true>
  - expected benefit: <metric moved, by how much>
  - verification command: <exact command that proves it>
  - compute cost: <rough GPU/CPU time>
  - leakage risk: <none | describe>
  - promotion rule: <condition under which this moves to ## ready>
-->

## blocked

<!-- Items parked with a concrete reason (unsafe / unclear / too expensive /
     rule-dependent). State the reason and what would unblock it. -->
