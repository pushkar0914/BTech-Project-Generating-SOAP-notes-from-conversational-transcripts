import os
import glob
import torch
import pandas as pd
from transformers import LEDTokenizerFast, LEDForConditionalGeneration
import evaluate

# ─── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Paths & Model ──────────────────────────────────────────────────────────────
INPUT_FOLDER  = "TESTING/sessions"           # your test CSV folder
TARGET_FOLDER = "TESTING/target_summaries"   # your test summary txts
MODEL_DIR     = "./fine_tuned_led"           # your fine-tuned LED

print("Loading fine-tuned model...")
tokenizer = LEDTokenizerFast.from_pretrained(MODEL_DIR)
model     = LEDForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ─── Metrics ────────────────────────────────────────────────────────────────────
rouge  = evaluate.load("rouge")
# bleu   = evaluate.load("bleu")
bertscore  = evaluate.load("bertscore")
meteor = evaluate.load("meteor")

# ─── Gather Prefixes ─────────────────────────────────────────────────────────────
all_csv  = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
prefixes = sorted({os.path.basename(f).split("_")[0] for f in all_csv})
print(f"Total test patients: {len(prefixes)}")

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
            if str(val).strip().lower() != "\"nothing reported\""
        ]
        segments.append(f"<Session {idx}>\n" + "\n".join(parts))
    combined = "\n\n".join(segments)

    # build input
    input_text = (
        "Summarize the following therapy sessions in one paragraph, "
        "in chronological order:\n\n"
        + combined
    )
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding="max_length"
    ).to(device)

    # global attention mask
    input_ids = enc.input_ids
    gam       = torch.zeros_like(input_ids)
    sid       = tokenizer.encode("<Session", add_special_tokens=False)[0]
    for i, t in enumerate(input_ids[0]):
        if t == sid or i == 0:
            gam[0, i] = 1

    # generate
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=enc.attention_mask,
            global_attention_mask=gam,
            max_length=300,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

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

# ─── Compute BERTScore ───────────────────────────────────────────────────────────
# returns precision, recall, f1 for each example; take the mean f1
bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")
avg_f1 = sum(bertscore_res["f1"]) / len(bertscore_res["f1"])
print(f"BERTScore (F1): {avg_f1:.4f}")

# ─── Compute ROUGE ───────────────────────────────────────────────────────────────
rouge_res = rouge.compute(predictions=preds, references=refs)
print(f"ROUGE-1 : {rouge_res['rouge1']:.4f}")
print(f"ROUGE-2 : {rouge_res['rouge2']:.4f}")
print(f"ROUGE-L : {rouge_res['rougeL']:.4f}")

# ─── Compute BLEU ────────────────────────────────────────────────────────────────
# pred_tokens = [p.split() for p in preds]
# ref_tokens  = [[[r.split()]] for r in refs]
# bleu_res    = bleu.compute(predictions=pred_tokens, references=ref_tokens)
# print(f"BLEU    : {bleu_res['bleu']:.4f}")

# ─── Compute METEOR ──────────────────────────────────────────────────────────────
meteor_res = meteor.compute(predictions=preds, references=refs)
print(f"METEOR  : {meteor_res['meteor']:.4f}")
