"""Fine-tune a small chat model on the prepared conversations.

Only the assistant's replies count towards the loss. Rerunning the same command
continues from the newest checkpoint in --output.

    python train/train_finetune.py --model HuggingFaceTB/SmolLM2-360M-Instruct \
        --output runs/smollm2-360m
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

IGNORE_INDEX = -100


def assistant_prefix(tokenizer):
    """Text the chat template puts before a reply, such as Qwen3's empty thinking block."""
    plain = [{"role": "user", "content": "x"}]
    prompt = tokenizer.apply_chat_template(plain, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt.split("<|im_start|>assistant\n", 1)[1]


def encode_conversation(messages, tokenizer, max_tokens, prefix=""):
    """ChatML tokens, with labels only on replies and their end-of-turn token."""
    input_ids, labels = [], []

    def add(text, learn):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids.extend(ids)
        labels.extend(ids if learn else [IGNORE_INDEX] * len(ids))

    for index, message in enumerate(messages):
        add(f"<|im_start|>{message['role']}\n", False)
        if message["role"] == "assistant":
            # Chat templates only add the prefix to the newest reply.
            if index == len(messages) - 1:
                add(prefix, False)
            add(f"{message['content']}<|im_end|>", True)
            add("\n", False)
        else:
            add(f"{message['content']}<|im_end|>\n", False)
    return {"input_ids": input_ids[:max_tokens], "labels": labels[:max_tokens]}


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_tokens, limit=None):
        prefix = assistant_prefix(tokenizer)
        self.examples = []
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if limit is not None and len(self.examples) >= limit:
                    break
                example = encode_conversation(json.loads(line)["messages"], tokenizer, max_tokens, prefix)
                if any(label != IGNORE_INDEX for label in example["labels"]):
                    self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        ids = [torch.tensor(example["input_ids"]) for example in examples]
        labels = [torch.tensor(example["labels"]) for example in examples]
        return {
            "input_ids": pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id),
            "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
            "attention_mask": pad_sequence([torch.ones_like(i) for i in ids], batch_first=True, padding_value=0),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--eval-every", type=int, default=500, help="Steps between validation and checkpoints")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop early, for smoke tests")
    parser.add_argument("--limit", type=int, help="Only load this many examples per split, for smoke tests")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    use_gpu = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_gpu:
        parser.error("CUDA was requested but no GPU is available")
    native_bf16 = use_gpu and torch.cuda.is_bf16_supported(including_emulation=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Mixed precision needs full-precision weights, whatever the checkpoint was saved in.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    train_data = ConversationDataset(args.data / "train.jsonl", tokenizer, args.max_tokens, args.limit)
    validation_data = ConversationDataset(args.data / "validation.jsonl", tokenizer, args.max_tokens, args.limit)
    print(f"{len(train_data)} training and {len(validation_data)} validation examples", flush=True)

    steps_per_epoch = max(1, len(train_data) // (args.batch_size * args.accumulation))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    training = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=int(total_steps * 0.03),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_strategy="steps",
        save_steps=args.eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, min(50, args.eval_every)),
        bf16=native_bf16,
        fp16=use_gpu and not native_bf16,
        use_cpu=not use_gpu,
        dataloader_num_workers=2 if use_gpu else 0,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=Collator(pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    checkpoint = get_last_checkpoint(str(args.output)) if args.output.exists() else None
    if checkpoint:
        print(f"Resuming from {checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)

    final = args.output / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"Saved best model to {final}", flush=True)


if __name__ == "__main__":
    main()
