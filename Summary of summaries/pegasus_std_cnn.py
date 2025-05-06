import os
import glob
import torch
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import evaluate

# ─── Device Setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Paths & Model ──────────────────────────────────────────────────────────────
INPUT_FOLDER  = "TESTING/sessions"
TARGET_FOLDER = "TESTING/target_summaries"
# model_name    = "google/pegasus-cnn_dailymail"
# model_name = "google/pegasus-pubmed"
model_name = "google/pegasus-large"

print("Loading Pegasus model and tokenizer...")
tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir="D:\\Sharvari_btech_project\\cache")
model     = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir="D:\\Sharvari_btech_project\\cache").to(device)
model.eval()

# ─── Load Metrics ────────────────────────────────────────────────────────────────
rouge      = evaluate.load("rouge")
meteor     = evaluate.load("meteor")
bertscore  = evaluate.load("bertscore")

# ─── Summarization Function ─────────────────────────────────────────────────────
def summarize_sessions(prefix):
    # collect session CSVs
    paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"{prefix}_TRANSCRIPT*.csv")))
    session_texts = []
    for idx, path in enumerate(paths, 1):
        df = pd.read_csv(path).iloc[0]
        parts = [
            f"{col}: {val}".strip()
            for col, val in df.items()
            if str(val).strip().lower() != "\"nothing reported\""
        ]
        session_texts.append(f"[Session {idx}]\n" + "\n".join(parts))

    combined = "\n\n".join(session_texts)

    # tokenize + truncate
    inputs = tokenizer(
        combined,
        max_length=2048,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    ).to(device)

    # generate
    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

# ─── Run on All Test Patients ───────────────────────────────────────────────────
prefixes = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))})
print(f"Found {len(prefixes)} patients.\n")

predictions = []
references  = []
sample_printed = False

for idx, pid in enumerate(prefixes, 1):
    print(f"[{idx}/{len(prefixes)}] Summarizing patient {pid}…", end=" ")
    pred = summarize_sessions(pid)
    print("done.")
    predictions.append(pred)

    # load reference
    ref_path = os.path.join(TARGET_FOLDER, f"{pid}.txt")
    if os.path.exists(ref_path):
        ref = open(ref_path, encoding="utf-8").read().strip()
        references.append(ref)
    else:
        print(f"  ⚠️ Missing reference for {pid}, skipping evaluation for this instance.")
        continue

    # print one sample
    if not sample_printed:
        print("\n=== Sample Summary ===")
        print("Generated Summary:\n", pred)
        print("Reference Summary:\n", ref)
        print("======================\n")
        sample_printed = True

# ─── Compute & Print Metrics ─────────────────────────────────────────────────────
print("\nComputing ROUGE, METEOR & BERTScore for Pegasus (pegasus-large)…\n")
rouge_res  = rouge.compute(predictions=predictions, references=references)
meteor_res = meteor.compute(predictions=predictions, references=references)

print(f"ROUGE-1 : {rouge_res['rouge1']:.4f}")
print(f"ROUGE-2 : {rouge_res['rouge2']:.4f}")
print(f"ROUGE-L : {rouge_res['rougeL']:.4f}")
print(f"METEOR  : {meteor_res['meteor']:.4f}")

# ─── Compute BERTScore ───────────────────────────────────────────────────────────
bertscore_res = bertscore.compute(predictions=predictions, references=references, lang="en")
# precision = sum(bertscore_res["precision"]) / len(bertscore_res["precision"])
# recall    = sum(bertscore_res["recall"])    / len(bertscore_res["recall"])
f1        = sum(bertscore_res["f1"])        / len(bertscore_res["f1"])
print(f"BERTScore (F1): {f1:.4f}")
