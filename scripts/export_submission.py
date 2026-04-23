from __future__ import annotations

import argparse
import ast
from pathlib import Path


def validate_submission_template(template: str) -> None:
    tree = ast.parse(template)
    fallback_defined = False
    agent_has_fallback = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fallback_greedy":
            fallback_defined = True
        if not isinstance(node, ast.FunctionDef) or node.name != "agent":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Try):
                continue
            for handler in stmt.handlers:
                for fallback_stmt in handler.body:
                    if not isinstance(fallback_stmt, ast.Return):
                        continue
                    call = fallback_stmt.value
                    if not isinstance(call, ast.Call):
                        continue
                    func = call.func
                    if isinstance(func, ast.Name) and func.id == "fallback_greedy":
                        agent_has_fallback = True
                        break

    if not fallback_defined or not agent_has_fallback:
        raise ValueError("submission template must define fallback_greedy and return it from agent() exception handling")


def render_submission(
    template: str,
    checkpoint: str | None = None,
    *,
    heuristic_policy: str | None = None,
) -> str:
    validate_submission_template(template)
    if heuristic_policy is not None:
        if "__HEURISTIC_POLICY__" not in template:
            raise ValueError("heuristic submission template requires __HEURISTIC_POLICY__ placeholder")
        template = template.replace("__HEURISTIC_POLICY__", heuristic_policy)
    if checkpoint is None:
        return template
    return f"# export_source: {checkpoint}\n{template}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="submission.py")
    args = parser.parse_args()

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    template = Path("python/submission/submission_template.py").read_text(encoding="utf-8")
    rendered = render_submission(template, args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print({"wrote": args.out, "checkpoint": args.checkpoint})


if __name__ == "__main__":
    main()
