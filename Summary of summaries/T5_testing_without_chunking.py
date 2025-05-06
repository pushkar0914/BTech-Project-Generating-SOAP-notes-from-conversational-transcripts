import os
import glob
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate

# ─── Device Setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Paths & Model ──────────────────────────────────────────────────────────────
INPUT_FOLDER  = "TESTING/sessions"
TARGET_FOLDER = "TESTING/target_summaries"
MODEL_NAME    = "t5-large"           # or "./fine_tuned_t5" if using your checkpoint

print("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir="D:\\Sharvari_btech_project\\cache")
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir="D:\\Sharvari_btech_project\\cache").to(device)
model.eval()

# ─── Metrics ────────────────────────────────────────────────────────────────────
rouge  = evaluate.load("rouge")
# bleu   = evaluate.load("bleu")
bertscore  = evaluate.load("bertscore")
meteor = evaluate.load("meteor")

# ─── Helper: One-Shot Summarization ──────────────────────────────────────────────
def summarize_patient(prefix):
    print(f"\n>> Summarizing patient {prefix}")
    # 1) Read and combine sessions
    session_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"{prefix}_TRANSCRIPT*.csv")))
    session_texts = []
    for idx, path in enumerate(session_paths, start=1):
        df = pd.read_csv(path).iloc[0]
        parts = [
            f"{col}: {val}".strip()
            for col, val in df.items()
            if str(val).strip().lower() != "\"nothing reported\""
        ]
        session_texts.append(f"Session {idx}:\n" + "\n".join(parts))

    combined = "\n\n".join(session_texts)
    input_text = "summarize: " + combined
    print(f"   Input length (chars): {len(input_text)}")

    # 2) Tokenize + truncate to 512
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length"
    ).to(device)

    # 3) Generate in one shot
    with torch.no_grad():
        out_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=300,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    summary = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    print(f"   Generated summary ({len(summary.split())} tokens)")
    return summary

# ─── Run on All Test Patients ───────────────────────────────────────────────────
preds, refs = [], []
prefixes = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))})
print(f"\nTotal patients: {len(prefixes)}")

for i, pid in enumerate(prefixes, 1):
    print(f"[{i}/{len(prefixes)}] {pid}", end=" ")
    summary = summarize_patient(pid)
    pred = summary
    ref_path = os.path.join(TARGET_FOLDER, f"{pid}.txt")
    if not os.path.exists(ref_path):
        print("⚠️ no ref")
        continue
    reference = open(ref_path, encoding="utf-8").read().strip()

    preds.append(pred)
    refs.append(reference)
    print("✓")

print("\nAll done. Computing metrics for T5…")

# ─── Compute BERTScore ───────────────────────────────────────────────────────────
# returns precision, recall, f1 for each example; take the mean f1
bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")
avg_f1 = sum(bertscore_res["f1"]) / len(bertscore_res["f1"])
print(f"BERTScore (F1): {avg_f1:.4f}")

# ─── ROUGE ───────────────────────────────────────────────────────────────────────
rouge_res = rouge.compute(predictions=preds, references=refs)
print(f"ROUGE-1 : {rouge_res['rouge1']:.4f}")
print(f"ROUGE-2 : {rouge_res['rouge2']:.4f}")
print(f"ROUGE-L : {rouge_res['rougeL']:.4f}")

# ─── METEOR ──────────────────────────────────────────────────────────────────────
meteor_res = meteor.compute(predictions=preds, references=refs)
print(f"METEOR  : {meteor_res['meteor']:.4f}")


