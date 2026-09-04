"""Kaggle helper for PS-26094 training, export, and eval jobs.

Examples:
  python kaggle/runner.py train --adapter a
  python kaggle/runner.py train --adapter b
  python kaggle/runner.py export --adapter all
  python kaggle/runner.py eval --kind dialogue --model sahayak --base-url http://localhost:8000/v1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_CONFIG = {
    "a": ROOT / "configs" / "adapter_a.yaml",
    "b": ROOT / "configs" / "adapter_b.yaml",
}


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("::", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=ROOT, env=env, check=True)  # noqa: S603


def install(kind: str) -> None:
    reqs = ["requirements-orch.txt"]
    if kind in {"train", "all"}:
        reqs.append("requirements-train.txt")
    for req in reqs:
        run([sys.executable, "-m", "pip", "install", "-r", ROOT / req])


def train(adapter: str) -> None:
    run([sys.executable, "src/training/train_sft.py", ADAPTER_CONFIG[adapter]])


def export(adapter: str) -> None:
    run([sys.executable, "src/deployment/export_gguf.py", "--adapter", adapter])


def evaluate(kind: str, model: str, base_url: str, api_key: str) -> None:
    run(
        [
            sys.executable,
            "src/evals/run_evals.py",
            "--base-url",
            base_url,
            "--kind",
            kind,
            "--model",
            model,
            "--api-key",
            api_key,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle orchestration helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_install = sub.add_parser("install")
    p_install.add_argument("--kind", choices=["orch", "train", "all"], default="all")

    p_train = sub.add_parser("train")
    p_train.add_argument("--adapter", choices=["a", "b"], required=True)
    p_train.add_argument("--install", action="store_true")

    p_export = sub.add_parser("export")
    p_export.add_argument("--adapter", choices=["a", "b", "all"], required=True)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--kind", choices=["dialogue", "summary"], required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"))
    p_eval.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    args = parser.parse_args()

    if args.cmd == "install":
        install(args.kind)
    elif args.cmd == "train":
        if args.install:
            install("train")
        train(args.adapter)
    elif args.cmd == "export":
        export(args.adapter)
    elif args.cmd == "eval":
        evaluate(args.kind, args.model, args.base_url, args.api_key)


if __name__ == "__main__":
    main()
