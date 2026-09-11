"""QLoRA SFT using trl SFTTrainer for native multi-turn chat_template masking."""

import os
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer  # <-- Use TRL's SFTTrainer

def main(cfg_path):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    t = cfg["training"]
    set_seed(t["seed"])
    
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

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
    
    # Apply LoRA
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

    # Load datasets directly (no manual encoding needed!)
    train_ds = load_dataset("json", data_files=cfg["data"]["train"], split="train")
    val_ds = load_dataset("json", data_files=cfg["data"]["val"], split="train")

    kwargs = {}
    try:  # MLflow optional
        import mlflow
        mlflow.set_experiment(cfg["run_name"])
        kwargs["report_to"] = "mlflow"
    except Exception:
        pass

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

    # SFTTrainer handles the chat_template formatting and -100 masking automatically
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=None,  # Already applied above
        tokenizer=tok,
        max_seq_length=t["max_seq_len"],
        dataset_text_field="messages", # Tells SFTTrainer to use the messages array
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