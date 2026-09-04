"""Kaggle-only: merge LoRA -> GGUF f16 -> Q8_0 -> private HF repo.
python src/deployment/export_gguf.py --adapter all   # or b | a"""

import argparse
import os
import subprocess

BASE = "sarvamai/sarvam-1"
JOBS = {
    "a": ("adapters/adapter_a/final", "sarvam1-dialogue-q8_0.gguf"),
    "b": ("adapters/adapter_b/final", "sarvam1-summary-q8_0.gguf"),
}


def sh(cmd: list) -> None:
    print("::", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export(which: str) -> None:
    adapter, _ = JOBS[which]
    merged, f16 = f"merged_{which}", f"{which}-f16.gguf"
    sh(["python", "src/training/merge_adapter.py", "--base", BASE, "--adapter", adapter, "--out", merged])
    if not os.path.isdir("llama.cpp"):
        sh(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp"])
    sh(["pip", "install", "-q", "-r", "llama.cpp/requirements.txt"])
    sh(["python", "llama.cpp/convert_hf_to_gguf.py", merged, "--outfile", f16, "--outtype", "f16"])
    if not os.path.exists("llama.cpp/llama-quantize"):
        sh(["make", "-C", "llama.cpp", "llama-quantize", "-j2"])
    q8 = f"sarvam1-{'dialogue' if which == 'a' else 'summary'}-q8_0.gguf"
    sh(["llama.cpp/llama-quantize", f16, q8, "Q8_0"])
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=q8,
        path_in_repo=q8,
        repo_id=os.environ.get("GGUF_REPO", "<you>/ps26094-gguf"),
        repo_type="model",
        private=True,
    )
    print(f"[gguf] uploaded {q8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", choices=["a", "b", "all"], required=True)
    args = p.parse_args()
    for w in ["a", "b"] if args.adapter == "all" else [args.adapter]:
        export(w)
