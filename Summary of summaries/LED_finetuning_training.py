import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    LEDTokenizerFast,
    LEDForConditionalGeneration,
    Trainer,
    TrainingArguments
)

# ─── Device Setup ───────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("🖥️  Training on:", device)

# ─── Dataset ─────────────────────────────────────────────────────────────────────
class LEDSoapSummarizationDataset(Dataset):
    def __init__(self, input_folder, target_folder, tokenizer,
                 max_source_length=1024, max_target_length=256):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = []

        csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
        groups = {}
        for path in csv_files:
            pid = os.path.basename(path).split("_")[0]
            groups.setdefault(pid, []).append(path)

        for pid, paths in groups.items():
            paths = sorted(paths)
            sessions = []
            for idx, p in enumerate(paths, 1):
                df = pd.read_csv(p)
                row = df.iloc[0]
                parts = []
                for col in df.columns:
                    v = str(row[col]).strip()
                    if v.lower() != "\"nothing reported\"":
                        parts.append(f"{col}: {v}")
                sessions.append(f"<Session {idx}>\n" + "\n".join(parts))

            input_text = (
                "Summarize the following therapy sessions in one paragraph, "
                "in chronological order:\n\n" + "\n\n".join(sessions)
            )

            summary_path = os.path.join(target_folder, f"{pid}.txt")
            if os.path.exists(summary_path):
                summary_text = open(summary_path, "r", encoding="utf-8").read().strip()
                self.data.append((input_text, summary_text))
            else:
                print(f"⚠️  Warning: no summary for patient {pid}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        source = self.tokenizer(
            src,
            truncation=True,
            padding="max_length",
            max_length=self.max_source_length,
            return_tensors="pt"
        )
        target = self.tokenizer(
            tgt,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt"
        )
        labels = target.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "global_attention_mask": self._build_global_attention(source.input_ids.squeeze()),
            "labels": labels,
        }

    def _build_global_attention(self, input_ids):
        mask = torch.zeros_like(input_ids)
        session_id = self.tokenizer.encode("<Session", add_special_tokens=False)[0]
        for i, tok in enumerate(input_ids):
            if tok == session_id:
                mask[i] = 1
        mask[0] = 1
        return mask

# ─── Fine-Tuning ─────────────────────────────────────────────────────────────────
def main():
    input_folder  = "TRAINING/sessions"
    target_folder = "TRAINING/target_summaries"
    model_name    = "allenai/led-large-16384-arxiv"

    print("🔁 Loading tokenizer and model...")
    tokenizer = LEDTokenizerFast.from_pretrained(model_name)
    model     = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.use_cache = False  # required when using gradient_checkpointing

    print("📚 Building dataset...")
    ds = LEDSoapSummarizationDataset(input_folder, target_folder, tokenizer)
    print(f"   → total training examples: {len(ds)}")

    training_args = TrainingArguments(
        output_dir="./fine_tuned_led",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy="no",      # no built-in eval
        save_strategy="steps",
        save_steps=200,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=5,
        learning_rate=3e-5,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        use_cpu=True
    )

    print("🚀 Starting training on entire dataset...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "global_attention_mask": torch.stack([b["global_attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        },
    )
    trainer.train()

    print("💾 Saving fine-tuned model...")
    model.save_pretrained("./fine_tuned_led")
    tokenizer.save_pretrained("./fine_tuned_led")

if __name__ == "__main__":
    main()
