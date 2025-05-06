import os
import glob
import torch
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import evaluate

# ─── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Paths & Model ──────────────────────────────────────────────────────────────
INPUT_FOLDER  = "TESTING/sessions"           # your test CSV folder
TARGET_FOLDER = "TESTING/target_summaries"   # your test summary txts
MODEL_DIR     = "./fine_tuned_pegasus"       # path to your fine-tuned Pegasus

print("Loading fine-tuned Pegasus model...")
tokenizer = PegasusTokenizer.from_pretrained(MODEL_DIR)
model     = PegasusForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ─── Metrics ────────────────────────────────────────────────────────────────────
rouge  = evaluate.load("rouge")
# bleu   = evaluate.load("bleu")
meteor = evaluate.load("meteor")
bertscore  = evaluate.load("bertscore")

# ─── Gather patient prefixes ─────────────────────────────────────────────────────
all_csv  = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
prefixes = sorted({os.path.basename(f).split("_")[0] for f in all_csv})
print(f"Total test patients: {len(prefixes)}\n")

# ─── Summarization Helper ────────────────────────────────────────────────────────
def summarize_patient(prefix):
    # collect session CSVs
    paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"{prefix}_TRANSCRIPT*.csv")))
    segments = []
    for idx, p in enumerate(paths, 1):
        df = pd.read_csv(p).iloc[0]
        parts = [
            f"{col}: {val}".strip()
            for col, val in df.items()
            if str(val).strip().lower() != "nothing reported"
        ]
        segments.append(f"[Session {idx}]\n" + "\n".join(parts))
    combined = "\n\n".join(segments)

    # tokenize + truncate to 1024 tokens
    inputs = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length"
    ).to(device)

    # generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=300,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

# ─── Run on All Test Patients ───────────────────────────────────────────────────
preds, refs = [], []
for i, pid in enumerate(prefixes, 1):
    print(f"[{i}/{len(prefixes)}] Summarizing patient {pid}…", end=" ")
    summary = summarize_patient(pid)
    preds.append(summary)

    ref_path = os.path.join(TARGET_FOLDER, f"{pid}.txt")
    if not os.path.exists(ref_path):
        print("missing reference, skipping")
        continue
    refs.append(open(ref_path, encoding="utf-8").read().strip())
    print("✓")

print("\nGeneration complete – computing metrics…")

# ─── Compute ROUGE ───────────────────────────────────────────────────────────────
rouge_res = rouge.compute(predictions=preds, references=refs)
print(f"ROUGE-1 : {rouge_res['rouge1']:.4f}")
print(f"ROUGE-2 : {rouge_res['rouge2']:.4f}")
print(f"ROUGE-L : {rouge_res['rougeL']:.4f}")

# ─── Compute BLEU ────────────────────────────────────────────────────────────────
# bleu_res = bleu.compute(predictions=preds, references=refs)
# print(f"BLEU    : {bleu_res['bleu']:.4f}")

# ─── Compute METEOR ──────────────────────────────────────────────────────────────
meteor_res = meteor.compute(predictions=preds, references=refs)
print(f"METEOR  : {meteor_res['meteor']:.4f}")

# ─── Compute BERTScore ───────────────────────────────────────────────────────────
# returns precision, recall, f1 for each example; take the mean f1
bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")
avg_f1 = sum(bertscore_res["f1"]) / len(bertscore_res["f1"])
print(f"BERTScore (F1): {avg_f1:.4f}")
