"""Unified PS-26094 job runner — pure Python, no IPython magics. VS Code + Kaggle.

  install --kind {auto,kaggle,train,orch}
  corpus  --adapter {a,b,both}
  train   --adapter {a,b} [--install]
  export  --adapter {a,b,all}
  eval    --kind {dialogue,summary} --model M [--base-url U] [--api-key K]

data/ is gitignored, so corpora NEVER arrive via git clone — `train` auto-builds
them (idempotent) before launching. Kaggle auto-uses requirements-train-kaggle.txt
(no torch: Kaggle's preinstalled torch+torchvision pair is version-matched)."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/AVIK-B/hackheritage_4_AI.git")
REPO_DIR = Path(REPO_URL).stem
REQUIRED_FILES = ("configs/adapter_a.yaml", "src/training/train_sft.py")
ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

REQ_FILES = {
    "orch": "requirements-orch.txt",
    "train": "requirements-train.txt",
    "kaggle": "requirements-train-kaggle.txt",
}
DATA_SENTINELS = {
    "b": "data/processed/case_summary_train.jsonl",
    "a": "data/processed/dialogue_train.jsonl",
}
CORPUS_CMDS = {
    "b": ["src/data/build_case_summary_corpus.py", "--n", "12000"],
    "a": ["src/data/build_dialogue_corpus.py", "--n-threads", "12000",
          "--ed-cap", "400", "--crisis-share", "0.12"],
}


def get_secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val and val.strip():
        return val.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name).strip()
    except Exception:
        return None


def _run(cmd: list, cwd: Path | None = None) -> None:
    print("::", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)  # noqa: S603 - fixed arg lists


def find_repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if all((root / f).exists() for f in REQUIRED_FILES):
        return root
    alt = Path.cwd() / REPO_DIR
    if all((alt / f).exists() for f in REQUIRED_FILES):
        return alt
    pat = get_secret("GITHUB_PAT")
    if not pat:
        raise SystemExit(f"repo markers not found near {root}; clone {REPO_URL} or set GITHUB_PAT")
    url = f"https://x-access-token:{pat}@{REPO_URL.split('https://')[-1]}"
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    _run(["git", "clone", "--depth", "1", url])
    return Path.cwd() / REPO_DIR


def ensure_hf_token() -> None:
    tok = get_secret("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
    if not os.environ.get("HF_TOKEN"):
        print(":: WARNING: HF_TOKEN unavailable — HF uploads will fail")


def install(kind: str, root: Path) -> None:
    if kind == "auto":
        kind = "kaggle" if ON_KAGGLE else "orch"
    req = REQ_FILES[kind]
    if not (root / req).exists():
        raise SystemExit(f"{req} not found in {root}")
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", req], cwd=root)
    if kind == "kaggle":  # canary: preinstalled pair must be intact
        _run([sys.executable, "-c",
              "import torch, torchvision, transformers, peft; "
              "print('torch', torch.__version__, '| torchvision', torchvision.__version__, "
              "'| cuda', torch.cuda.is_available())"], cwd=root)


def ensure_corpus(adapter: str, root: Path) -> None:
    """Idempotent: build the corpus for this adapter if its sentinel file is missing."""
    sentinel = root / DATA_SENTINELS[adapter]
    if sentinel.exists():
        print(f":: corpus present: {sentinel}")
        return
    script, *args = CORPUS_CMDS[adapter]
    print(f":: data/ is gitignored — building corpus for adapter {adapter}")
    _run([sys.executable, script, *args], cwd=root)

def train(adapter: str, root: Path) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # one T4 — no DP, no pipeline split
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # OOM-fragmentation guard
    ensure_hf_token()
    ensure_corpus(adapter, root)
    _run([sys.executable, "src/training/train_sft.py",
          root / "configs" / f"adapter_{adapter}.yaml"], cwd=root)


def export(adapter: str, root: Path) -> None:
    ensure_hf_token()
    _run([sys.executable, "src/deployment/export_gguf.py", "--adapter", adapter], cwd=root)


def evaluate(kind: str, model: str, base_url: str, api_key: str, root: Path) -> None:
    _run([sys.executable, "src/evals/run_evals.py", "--base-url", base_url,
          "--kind", kind, "--model", model, "--api-key", api_key], cwd=root)


def main() -> None:
    p = argparse.ArgumentParser(description="PS-26094 job runner (VS Code + Kaggle)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("install")
    pi.add_argument("--kind", choices=[*REQ_FILES, "auto"], default="auto")

    pc = sub.add_parser("corpus")
    pc.add_argument("--adapter", choices=["a", "b", "both"], default="both")

    pt = sub.add_parser("train")
    pt.add_argument("--adapter", choices=["a", "b"], required=True)
    pt.add_argument("--install", action="store_true")

    pe = sub.add_parser("export")
    pe.add_argument("--adapter", choices=["a", "b", "all"], required=True)

    pv = sub.add_parser("eval")
    pv.add_argument("--kind", choices=["dialogue", "summary"], required=True)
    pv.add_argument("--model", required=True)
    pv.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"))
    pv.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    args = p.parse_args()
    root = find_repo_root()
    print(f":: repo root: {root} | on_kaggle={ON_KAGGLE}")

    if args.cmd == "install":
        install(args.kind, root)
    elif args.cmd == "corpus":
        targets = ["a", "b"] if args.adapter == "both" else [args.adapter]
        for t in targets:
            ensure_corpus(t, root)
    elif args.cmd == "train":
        if args.install:
            install("auto", root)
        train(args.adapter, root)
    elif args.cmd == "export":
        export(args.adapter, root)
    elif args.cmd == "eval":
        evaluate(args.kind, args.model, args.base_url, args.api_key, root)


if __name__ == "__main__":
    main()