import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", default="sarvamai/sarvam-1")
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
m.save_pretrained(a.out)
AutoTokenizer.from_pretrained(a.base, trust_remote_code=True).save_pretrained(a.out)
print(f"[merge] {a.out}")
