from __future__ import annotations

import argparse

from rich import print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    print("[yellow]Parity probe scaffold[/yellow]")
    print({"episodes": args.episodes, "steps": args.steps})
    print("TODO: instantiate kaggle_environments.make('orbit_wars'), snapshot official states, step Rust from snapshot, compare planets/fleets/comets.")


if __name__ == "__main__":
    main()
