"""Process-parallel opponent evaluation for batched rollout (todo P3 throughput).

The Producer/OEP planners are the per-step bottleneck of the batched rollout and
are GIL-bound pure Python, so a thread pool does NOT help (measured ~20x slower).
Real parallelism needs separate processes. This pool spawns persistent workers
with a FIXED env->worker assignment so each opponent instance's per-game memory
lives entirely inside one worker and is reused across rollout steps (it resets on
``step == 0`` like the in-process instances, so the same pool is reused across
segments / games).

Workers are spawned (not forked) so a CUDA-initialised parent is safe; each worker
builds its own isolated opponent instances on the CPU. The pool is cached per
(opponent, num_envs, workers); ``close_all`` is registered at interpreter exit.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import Any

Move = list[float]
Moves = list[Move]


def _worker_loop(name: str, env_indices: list[int], conn) -> None:
    # Built inside the worker process (torch runtimes are not picklable).
    from python.agents.registry import _make_isolated_policy

    policies = {env: _make_isolated_policy(name) for env in env_indices}
    try:
        while True:
            msg = conn.recv()
            if msg == "STOP":
                break
            conn.send([(env, policies[env](state, 1)) for env, state in msg])
    finally:
        conn.close()


class ProcessOpponentPool:
    def __init__(self, name: str, num_envs: int, num_workers: int) -> None:
        if num_workers < 2:
            raise ValueError("ProcessOpponentPool requires num_workers >= 2")
        ctx = mp.get_context("spawn")
        self.name = name
        self.num_envs = num_envs
        self.assignment: dict[int, int] = {}
        buckets: list[list[int]] = [[] for _ in range(num_workers)]
        for env in range(num_envs):
            w = env % num_workers
            buckets[w].append(env)
            self.assignment[env] = w
        self._conns = []
        self._procs = []
        for w in range(num_workers):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_loop, args=(name, buckets[w], child_conn), daemon=True
            )
            proc.start()
            child_conn.close()  # parent keeps only its end
            self._conns.append(parent_conn)
            self._procs.append(proc)
        self._closed = False

    def moves(self, states: list[dict[str, Any]], active_indices: list[int]) -> dict[int, Moves]:
        """Return ``{env_index: opponent_moves}`` for the active envs.

        Dispatches each active env's state to its assigned worker, runs all workers
        concurrently, and collects results. A dead worker surfaces loudly (recv
        raises) rather than silently degrading.
        """
        if self._closed:
            raise RuntimeError("ProcessOpponentPool used after close()")
        per_worker: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in self._conns]
        for env in active_indices:
            per_worker[self.assignment[env]].append((env, states[env]))

        sent = [w for w, batch in enumerate(per_worker) if batch]
        for w in sent:
            self._conns[w].send(per_worker[w])
        result: dict[int, Moves] = {}
        for w in sent:
            for env, moves in self._conns[w].recv():
                result[env] = list(moves) if isinstance(moves, list) else []
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send("STOP")
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass


_POOLS: dict[tuple[str, int, int], ProcessOpponentPool] = {}


def get_process_opponent_pool(name: str, num_envs: int, num_workers: int) -> ProcessOpponentPool:
    """Return a cached persistent pool; built once and reused across segments."""
    key = (name, int(num_envs), int(num_workers))
    pool = _POOLS.get(key)
    if pool is None:
        pool = ProcessOpponentPool(name, int(num_envs), int(num_workers))
        _POOLS[key] = pool
    return pool


def close_all() -> None:
    for pool in list(_POOLS.values()):
        pool.close()
    _POOLS.clear()


atexit.register(close_all)
