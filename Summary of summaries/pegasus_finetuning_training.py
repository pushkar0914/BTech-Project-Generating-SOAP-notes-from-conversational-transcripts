import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    Trainer,
    TrainingArguments
)

# ─── Device Setup ───────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("🖥️  Training on:", device)

# ─── Dataset ─────────────────────────────────────────────────────────────────────
class PegasusSoapSummarizationDataset(Dataset):
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
                df = pd.read_csv(p).iloc[0]
                parts = []
                for col, val in df.items():
                    v = str(val).strip()
                    if v.lower() != "nothing reported":
                        parts.append(f"{col}: {v}")
                sessions.append(f"[Session {idx}]\n" + "\n".join(parts))

            src = "\n\n".join(sessions)
            tgt_path = os.path.join(target_folder, f"{pid}.txt")
            if os.path.exists(tgt_path):
                tgt = open(tgt_path, encoding="utf-8").read().strip()
                self.data.append((src, tgt))
            else:
                print(f"⚠️  No summary for patient {pid}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        enc = self.tokenizer(
            src,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        dec = self.tokenizer(
            tgt,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = dec.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "labels": labels
        }

# ─── Fine-Tuning ─────────────────────────────────────────────────────────────────
def main():
    input_folder  = "TRAINING/sessions"
    target_folder = "TRAINING/target_summaries"
    model_name    = "google/pegasus-large"

    print("🔁 Loading tokenizer and model...")
    tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir="D:\\Sharvari_btech_project\\cache")
    model     = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir="D:\\Sharvari_btech_project\\cache").to(device)
    model.config.use_cache = False  # required when using gradient_checkpointing

    print("📚 Building dataset...")
    ds = PegasusSoapSummarizationDataset(
        input_folder, target_folder,
        tokenizer,
        max_source_length=1024,
        max_target_length=256
    )
    print(f"   → total training examples: {len(ds)}")

    training_args = TrainingArguments(
        output_dir="./fine_tuned_pegasus",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # effective batch size = 4
        save_strategy="steps",
        save_steps=10,
        save_total_limit=5,
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=3e-5,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        eval_strategy="no",
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
            "labels": torch.stack([b["labels"] for b in batch]),
        },
    )
    trainer.train()

    print("💾 Saving fine-tuned model...")
    model.save_pretrained("./fine_tuned_pegasus")
    tokenizer.save_pretrained("./fine_tuned_pegasus")

if __name__ == "__main__":
    main()
