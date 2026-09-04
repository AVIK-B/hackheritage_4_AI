"""QLoRA SFT with prefix/completion masking (-100) — model learns to emit ONLY assistant text."""

import os
from dataclasses import dataclass

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


def build_collator(tok):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @dataclass
    class C:
        def __call__(self, feats):
            m = max(len(f["input_ids"]) for f in feats)

            def pad(seq, fill):
                return seq + [fill] * (m - len(seq))

            return {
                "input_ids": torch.tensor([pad(f["input_ids"], pad_id) for f in feats]),
                "labels": torch.tensor([pad(f["labels"], -100) for f in feats]),
                "attention_mask": torch.tensor([pad([1] * len(f["input_ids"]), 0) for f in feats]),
            }

    return C()


def encode(tok, prefix_msgs, target, max_len):
    p = tok(
        tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False
    ).input_ids
    c = tok(target + tok.eos_token, add_special_tokens=False).input_ids
    ids, labels = (p + c)[:max_len], ([-100] * len(p) + c)[:max_len]
    return None if all(lab == -100 for lab in labels) else {"input_ids": ids, "labels": labels}


def get_ds(path, tok, max_len, fmt):
    ds = load_dataset("json", data_files=path, split="train")

    degenerate = {"n": 0}

    def m(ex):
        if fmt == "prefix_completion":
            e = encode(tok, ex["prefix"], ex["target"], max_len)
        else:
            e = encode(tok, ex["messages"][:-1], ex["messages"][-1]["content"], max_len)
        if e is None:
            degenerate["n"] += 1
            return {"input_ids": [tok.eos_token_id], "labels": [tok.eos_token_id]}
        return e

    out = ds.map(m, remove_columns=ds.column_names, num_proc=4)
    if degenerate["n"]:
        raise SystemExit(
            f"DATA LOSS: {degenerate['n']} examples' prefixes exceed max_seq_len={max_len} "
            f"and were fully masked. Raise max_seq_len or shorten the prompt — DO NOT TRAIN."
        )
    return out


def main(cfg_path):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    t = cfg["training"]
    set_seed(t["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias="none",
            target_modules=cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    kwargs = {}
    try:  # MLflow optional — never blocks a local run
        import mlflow

        mlflow.set_experiment(cfg["run_name"])
        kwargs["report_to"] = "mlflow"
    except Exception as exc:
        print(f"[mlflow] unavailable, continuing without experiment tracking: {exc}")

    args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        fp16=True,
        bf16=False,  # T4: fp16 only
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["eval_steps"],
        save_total_limit=2,
        group_by_length=True,
        seed=t["seed"],
        dataloader_num_workers=2,
        **kwargs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=get_ds(cfg["data"]["train"], tok, t["max_seq_len"], cfg["data"]["format"]),
        eval_dataset=get_ds(cfg["data"]["val"], tok, t["max_seq_len"], cfg["data"]["format"]),
        data_collator=build_collator(tok),
    )
    trainer.train(resume_from_checkpoint=os.environ.get("RESUME") or None)
    final = os.path.join(t["output_dir"], "final")
    trainer.save_model(final)
    tok.save_pretrained(final)

    up = cfg.get("upload")
    if up and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=final, repo_id=up["repo_id"], path_in_repo=up["path_in_repo"], repo_type="model", private=True
        )
        print(f"[upload] {up['repo_id']}:{up['path_in_repo']}")


if __name__ == "__main__":
    main(__import__("sys").argv[1])
