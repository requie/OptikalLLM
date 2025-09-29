"""
Example training script for OptikalAI.

This script demonstrates how to perform parameter‑efficient fine‑tuning of a
pretrained causal language model using Low‑Rank Adaptation (LoRA).  It uses
Hugging Face's `transformers`, `datasets` and `peft` libraries to fine‑tune a
base model on a domain‑specific instruction‑response dataset.  The resulting
LoRA adapter weights can then be saved and uploaded to the Hugging Face Hub.

Note: This script is for demonstration purposes and may need to be modified
depending on the size of your dataset and available hardware.  Fine‑tuning
large language models requires GPUs with substantial memory.
"""
import argparse
import json
import os
from typing import Dict, List

import datasets
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Fine‑tune a base LLM using LoRA for cybersecurity tasks.")
    parser.add_argument("--base_model", type=str, required=True, help="Hugging Face ID or path of the base model (e.g., meta-llama/Llama-2-7b-hf).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to a JSONL dataset with 'instruction' and 'response' fields.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA adapter weights.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r).  Higher values increase parameter count.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    return parser.parse_args()


def load_instruction_dataset(path: str) -> datasets.Dataset:
    """
    Load a JSONL dataset where each line contains an `instruction` and a
    corresponding `response`.  Returns a Hugging Face `Dataset` object.

    Args:
        path: Path to the JSON Lines file.
    Returns:
        A `datasets.Dataset` containing `prompt` and `text` fields for training.
    """
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                instruction = obj.get("instruction", "").strip()
                response = obj.get("response", "").strip()
                if instruction and response:
                    # Concatenate instruction and response using a special separator.
                    records.append({"prompt": instruction, "text": response})
            except json.JSONDecodeError:
                continue
    return datasets.Dataset.from_list(records)


def main() -> None:
    args = parse_args()

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # Ensure the tokenizer uses a padding token (required for batch collation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")

    # Prepare LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_instruction_dataset(args.dataset_path)

    # Tokenize the dataset
    def tokenize_function(example: Dict[str, str]) -> Dict[str, List[int]]:
        # Create a prompt with instruction and response separated by a newline
        merged = example["prompt"] + "\n\n" + example["text"] + tokenizer.eos_token
        tokenized = tokenizer(merged, truncation=True, max_length=1024)
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=["prompt", "text"])

    # Data collator for language modelling tasks
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapter weights only (exclude the base model weights)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()