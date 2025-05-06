# GPU COMPATIBLE CODE WITH PRINT STATEMENTS_TO_RUN

import os
import glob
import math
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate

# ─── Device Setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Paths ──────────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "TESTING\\sessions"
TARGET_FOLDER = "TESTING\\target_summaries"

# ─── Load Model & Tokenizer ─────────────────────────────────────────────────────
model_name = "t5-large"   # or your fine‑tuned checkpoint
tokenizer  = T5Tokenizer.from_pretrained(model_name)
model      = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# ─── Load Evaluation Metrics ────────────────────────────────────────────────────
rouge   = evaluate.load("rouge")
bleu    = evaluate.load("bleu")
meteor  = evaluate.load("meteor")

# ─── Helper: Summarize One Patient ───────────────────────────────────────────────
def summarize_patient(prefix):
    print(f"\n>> Summarizing patient {prefix}")
    # 1) Collect session files
    session_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"{prefix}_TRANSCRIPT*.csv")))
    print(f"   Found {len(session_paths)} session files")
    session_texts = []
    for idx, path in enumerate(session_paths, start=1):
        df = pd.read_csv(path)
        row = df.iloc[0]
        parts = []
        for col in df.columns:
            v = str(row[col]).strip()
            if v.lower() != "\"nothing reported\"":
                parts.append(f"{v}")
        session_texts.append(f"Session {idx}:\n" + "\n".join(parts))

    # 2) Build hierarchical input
    combined   = "\n".join(session_texts)
    input_text = "summarize: " + combined
    print(f"   Input length (chars): {len(input_text)}")
    input_ids  = tokenizer.encode(input_text, return_tensors="pt", truncation=False).to(device)

    # 3) First‑level chunking
    chunk_size, overlap = 256, 100
    total_tokens    = input_ids.shape[1]
    num_chunks      = math.ceil((total_tokens - overlap) / (chunk_size - overlap))
    print(f"   First-level chunking: total_tokens={total_tokens}, chunk_size={chunk_size}, overlap={overlap}, num_chunks≈{num_chunks}")
    chunk_summaries = []

    with torch.no_grad():
        for chunk_idx, start in enumerate(range(0, total_tokens, chunk_size - overlap), start=1):
            end       = min(start + chunk_size, total_tokens)
            print(f"      Chunk {chunk_idx}/{num_chunks}: tokens {start}-{end}")
            chunk_ids = input_ids[:, start:end]

            out = model.generate(
                chunk_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            summary_chunk = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"         → chunk summary (len={len(summary_chunk.split())} tokens)")
            chunk_summaries.append(summary_chunk)

        # 4) Final summarization
        print(f"   Combining {len(chunk_summaries)} chunk summaries for final pass")
        final_input = "summarize: " + " ".join(chunk_summaries)
        final_ids   = tokenizer.encode(
            final_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        summary_ids = model.generate(
            final_ids,
            max_length=300,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    print(f"   Final summary length: {len(final_summary.split())} tokens")
    return final_summary

# ─── Run Over Entire Dataset ─────────────────────────────────────────────────────
predictions = []
references  = []

# Find unique patient prefixes
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
prefixes  = sorted({os.path.basename(f).split("_")[0] for f in csv_files})
print(f"\nTotal patients to process: {len(prefixes)}")

for idx, prefix in enumerate(prefixes, start=1):
    print(f"\n=== [{idx}/{len(prefixes)}] Processing patient {prefix} ===")
    pred = summarize_patient(prefix)

    ref_path = os.path.join(TARGET_FOLDER, f"{prefix}.txt")
    if not os.path.exists(ref_path):
        print(f"⚠️ Missing reference for {prefix}, skipping")
        continue
    ref = open(ref_path, encoding="utf-8").read().strip()

    predictions.append(pred)
    references.append(ref)

print("\nAll summaries generated. Computing metrics...")

# ─── Compute ROUGE ────────────────────────────────────────────────────────────────
rouge_res = rouge.compute(predictions=predictions, references=references)
# print(f"\nROUGE-1   : {rouge_res['rouge1'].mid.fmeasure:.4f}")
# print(f"ROUGE-2   : {rouge_res['rouge2'].mid.fmeasure:.4f}")
# print(f"ROUGE-L   : {rouge_res['rougeL'].mid.fmeasure:.4f}")

print(f"ROUGE-1   : {rouge_res['rouge1']:.4f}")
print(f"ROUGE-2   : {rouge_res['rouge2']:.4f}")
print(f"ROUGE-L   : {rouge_res['rougeL']:.4f}")

# ─── Compute BLEU ────────────────────────────────────────────────────────────────
# pred_tokens = [pred.split() for pred in predictions]
# ref_tokens  = [[[ref.split()]] for ref in references]
# bleu_res    = bleu.compute(predictions=pred_tokens, references=ref_tokens)
# print(f"\nBLEU      : {bleu_res['bleu']:.4f}")
bleu_res = bleu.compute(predictions=predictions, references=references)
print(f"BLEU      : {bleu_res['bleu']:.4f}")


# pred_tokens = [p.split()       for p in predictions]   # List[List[str]]
# ref_tokens  = [[r.split()]     for r in references]    # List[List[List[str]]]

# bleu_res = bleu.compute(
#     predictions=pred_tokens,
#     references=ref_tokens
# )
# print(f"BLEU      : {bleu_res['bleu']:.4f}")

# ─── Compute METEOR ──────────────────────────────────────────────────────────────
meteor_res = meteor.compute(predictions=predictions, references=references)
print(f"\nMETEOR    : {meteor_res['meteor']:.4f}")
