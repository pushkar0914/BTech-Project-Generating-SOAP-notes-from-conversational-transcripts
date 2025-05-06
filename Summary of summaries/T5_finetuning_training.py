import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

# ─── Device Setup ───────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("🖥️  Training on:", device)

# ─── Dataset ─────────────────────────────────────────────────────────────────────
class T5SoapSummarizationDataset(Dataset):
    def __init__(self, input_folder, target_folder, tokenizer,
                 max_source_length=512, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = []

        # group CSVs by patient prefix
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
                    if v.lower() != "\"nothing reported\"":
                        parts.append(f"{col}: {v}")
                sessions.append(f"Session {idx}:\n" + "\n".join(parts))

            # combine sessions
            src = "summarize: " + "\n\n".join(sessions)
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

# ─── Fine‐Tuning ─────────────────────────────────────────────────────────────────
def main():
    input_folder  = "TRAINING/sessions"
    target_folder = "TRAINING/target_summaries"
    model_name    = "t5-large"

    print("🔁 Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model     = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    print("📚 Building dataset...")
    ds = T5SoapSummarizationDataset(
        input_folder, target_folder,
        tokenizer,
        max_source_length=512,
        max_target_length=128
    )
    print(f"   → total examples: {len(ds)}")

    training_args = TrainingArguments(
        output_dir="./fine_tuned_t5",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # effective batch size = 4
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        learning_rate=3e-5,
        fp16=True,
        remove_unused_columns=False,
        eval_strategy="no",
        no_cuda=True
    )

    print("🚀 Starting training...")
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
    model.save_pretrained("./fine_tuned_t5")
    tokenizer.save_pretrained("./fine_tuned_t5")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# ###############################################################################
# #  T5 SOAP-note summarisation — AMP + gradient-checkpointing on 8 GB GPU
# ###############################################################################

# import os
# # ── allocator hints *before* torch is imported ────────────────────────────────
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
#     "expandable_segments:True,max_split_size_mb:64"

# import gc
# import glob
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from transformers import (
#     T5Tokenizer,
#     T5ForConditionalGeneration,
#     Trainer,
#     TrainingArguments,
#     TrainerCallback
# )

# # ─── Callback: quick GPU-memory dashboard each step ───────────────────────────
# class MemReport(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         alloc   = torch.cuda.memory_allocated()  / 2**20
#         reserve = torch.cuda.memory_reserved()   / 2**20
#         print(f"step {state.global_step:>4}: "
#               f"allocated {alloc:7.1f} MB | "
#               f"reserved {reserve:7.1f} MB", end="\r")
#         torch.cuda.empty_cache(); gc.collect()
#         return control


# # ─── Dataset definition ───────────────────────────────────────────────────────
# class T5SoapSummarizationDataset(Dataset):
#     def __init__(self, input_folder, target_folder, tokenizer,
#                  max_source_length=512, max_target_length=128):
#         self.tokenizer = tokenizer
#         self.max_source_length = max_source_length
#         self.max_target_length = max_target_length
#         self.data = []

#         # group CSVs by patient ID prefix
#         csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
#         groups = {}
#         for path in csv_files:
#             pid = os.path.basename(path).split("_")[0]
#             groups.setdefault(pid, []).append(path)

#         for pid, paths in groups.items():
#             sessions = []
#             for idx, p in enumerate(sorted(paths), 1):
#                 df = pd.read_csv(p).iloc[0]
#                 parts = [
#                     f"{col}: {str(val).strip()}"
#                     for col, val in df.items()
#                     if str(val).strip().lower() != "\"nothing reported\""
#                 ]
#                 sessions.append(f"Session {idx}:\n" + "\n".join(parts))

#             src = "summarize: " + "\n\n".join(sessions)
#             tgt_path = os.path.join(target_folder, f"{pid}.txt")
#             if os.path.exists(tgt_path):
#                 tgt = open(tgt_path, encoding="utf-8").read().strip()
#                 self.data.append((src, tgt))
#             else:
#                 print(f"⚠️  No summary for patient {pid}")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         src, tgt = self.data[idx]
#         enc = self.tokenizer(
#             src,
#             max_length=self.max_source_length,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt"
#         )
#         dec = self.tokenizer(
#             tgt,
#             max_length=self.max_target_length,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt"
#         )
#         labels = dec.input_ids.squeeze()
#         labels[labels == self.tokenizer.pad_token_id] = -100
#         return {
#             "input_ids":      enc.input_ids.squeeze(),
#             "attention_mask": enc.attention_mask.squeeze(),
#             "labels":         labels
#         }


# # ─── Main fine-tuning routine ─────────────────────────────────────────────────
# def main():
#     input_folder  = "TRAINING/sessions"
#     target_folder = "TRAINING/target_summaries"
#     model_name    = "t5-large"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("🖥️  Training on:", device)

#     # 1) tokenizer
#     tokenizer = T5Tokenizer.from_pretrained(model_name)

#     # 2) model in *FP32* (let AMP handle casting) ⬅️
#     print("🔁 Loading model ...")
#     model = T5ForConditionalGeneration.from_pretrained(
#         model_name,
#         low_cpu_mem_usage=True
#     )
#     model.gradient_checkpointing_enable()
#     model.config.use_cache = False
#     model.to(device)

#     # 3) dataset
#     print("📚 Building dataset ...")
#     ds = T5SoapSummarizationDataset(
#         input_folder, target_folder, tokenizer,
#         max_source_length=512, max_target_length=128
#     )
#     print(f"   → total examples: {len(ds)}")

#     # 4) training arguments
#     training_args = TrainingArguments(
#         output_dir="./fine_tuned_t5",
#         num_train_epochs=3,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,      # effective batch = 4
#         learning_rate=3e-5,
#         fp16=True,                          # AMP mixed precision ⬅️
#         save_strategy="epoch",
#         save_total_limit=2,
#         logging_strategy="steps",
#         logging_steps=50,
#         remove_unused_columns=False,
#         eval_strategy="no"
#     )

#     # 5) Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=ds,
#         data_collator=lambda batch: {
#             "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
#             "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
#             "labels":         torch.stack([b["labels"]         for b in batch]),
#         },
#         callbacks=[MemReport()]
#         # default AdamW + GradScaler are fine now ⬅️
#     )

#     # 6) train
#     print("🚀 Starting training ...")
#     trainer.train()

#     # 7) save
#     print("\n💾 Saving fine-tuned model ...")
#     model.save_pretrained("./fine_tuned_t5")
#     tokenizer.save_pretrained("./fine_tuned_t5")

#     # 8) final clean-up
#     torch.cuda.empty_cache()
#     gc.collect()


# if __name__ == "__main__":
#     main()
