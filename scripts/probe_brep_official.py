"""[DEBUG-off] Reproduce the Kaggle ERROR via the OFFICIAL simulator.
Runs the BReP submission agent in make('orbit_wars') for 2p AND 4p (Kaggle
validates both). status != DONE => the cause of the submission ERROR. Throwaway."""
from __future__ import annotations
import importlib.util, sys, tarfile, tempfile, traceback
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

tmp = tempfile.mkdtemp()
with tarfile.open("artifacts/submission_brep.tar.gz") as t:
    t.extractall(tmp)
sys.path.insert(0, tmp)
# Load the agent EXACTLY as Kaggle does — get_last_callable returns the LAST
# callable in the module namespace, NOT necessarily a function named `agent`.
# (Using mod.agent here would hide the wrong-function-picked bug that ERRORed.)
main_path = str(Path(tmp) / "main.py")
agent = get_last_callable(Path(main_path).read_text(), path=main_path)
print(f"[DEBUG-off] get_last_callable resolved: {getattr(agent, '__name__', agent)} "
      f"(MUST be 'agent', not a _brep_* helper)")

for n in (2, 4):
    print(f"\n[DEBUG-off] ===== official orbit_wars, {n} players =====")
    try:
        env = make("orbit_wars", configuration={"agents": n}, debug=True)
        env.run([agent] * n)
        statuses = [s.status for s in env.state]
        print(f"[DEBUG-off]   statuses={statuses}")
        for i, s in enumerate(env.state):
            if s.status != "DONE":
                err = s.get("info", {}).get("error") if hasattr(s, "get") else None
                print(f"[DEBUG-off]   player {i}: status={s.status} error={err}")
        print(f"[DEBUG-off]   VERDICT {n}p: {'ALL DONE' if all(s.status=='DONE' for s in env.state) else 'FAILURE — see above'}")
    except Exception:
        print(f"[DEBUG-off]   {n}p RAISED:")
        traceback.print_exc()
