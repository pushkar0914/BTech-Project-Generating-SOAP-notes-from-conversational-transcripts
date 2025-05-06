# TO RUN NORMAL LED

import os
import glob
import torch
import pandas as pd
from transformers import LEDTokenizerFast, LEDForConditionalGeneration
import evaluate

# ─── Device Setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)

# ─── Paths ──────────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "TESTING\\sessions"
TARGET_FOLDER = "TESTING\\target_summaries"

# ─── Load Model & Tokenizer ─────────────────────────────────────────────────────
print("Loading LED model and tokenizer...")
model_name = "allenai/led-large-16384-arxiv"
tokenizer  = LEDTokenizerFast.from_pretrained(model_name)
model      = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# ─── Load Evaluation Metrics ────────────────────────────────────────────────────
print("Loading evaluation metrics...")
rouge  = evaluate.load("rouge")
# bleu   = evaluate.load("bleu")
bertscore  = evaluate.load("bertscore")
meteor = evaluate.load("meteor")

# ─── Inference Helper ────────────────────────────────────────────────────────────
def summarize_sessions_for(prefix):
    # 1) Read and combine sessions
    paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"{prefix}_TRANSCRIPT*.csv")))
    session_texts = []
    for idx, p in enumerate(paths, start=1):
        df = pd.read_csv(p).iloc[0]
        parts = [
            f"{col}: {str(val).strip()}"
            for col, val in df.items()
            if str(val).strip().lower() != "\"nothing reported\""
        ]
        session_texts.append(f"<Session {idx}>\n" + "\n".join(parts))

    # 2) Build input string
    input_text = (
        "Summarize and paraphrase the following therapy sessions in one paragraph, "
        "in chronological order:\n\n"
        + "\n\n".join(session_texts)
    )

    # 3) Tokenize
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=2048
    ).to(device)

    # 4) Build global attention mask
    input_ids = encoded.input_ids
    gam = torch.zeros_like(input_ids)
    session_tok_id = tokenizer.encode("<Session", add_special_tokens=False)[0]
    for i, tok in enumerate(input_ids[0]):
        if tok == session_tok_id or i == 0:
            gam[0, i] = 1

    # 5) Generate
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=encoded.attention_mask,
            global_attention_mask=gam,
            decoder_start_token_id=tokenizer.bos_token_id,
            num_beams=6,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            early_stopping=True,
            max_length=512,
            num_beam_groups=2,
            diversity_penalty=0.5
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

# ─── Run Testing ─────────────────────────────────────────────────────────────────
predictions = []
references  = []

# gather patient prefixes
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
prefixes  = sorted({os.path.basename(f).split("_")[0] for f in csv_files})
print(f"Found {len(prefixes)} patients.")

for i, pid in enumerate(prefixes, start=1):
    print(f"[{i}/{len(prefixes)}] Summarizing patient {pid}...")
    pred = summarize_sessions_for(pid)

    ref_path = os.path.join(TARGET_FOLDER, f"{pid}.txt")
    if not os.path.exists(ref_path):
        print(f"Reference missing for {pid}, skipping.")
        continue
    ref = open(ref_path, encoding="utf-8").read().strip()

    predictions.append(pred)
    references.append(ref)

print("\nAll done. Computing metrics…")

# ─── ROUGE ───────────────────────────────────────────────────────────────────────
rouge_res = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1 : {rouge_res['rouge1']:.4f}")
print(f"ROUGE-2 : {rouge_res['rouge2']:.4f}")
print(f"ROUGE-L : {rouge_res['rougeL']:.4f}")

# ─── METEOR ──────────────────────────────────────────────────────────────────────
meteor_res = meteor.compute(predictions=predictions, references=references)
print(f"METEOR  : {meteor_res['meteor']:.4f}")

# ─── Compute BERTScore ───────────────────────────────────────────────────────────
# returns precision, recall, f1 for each example; take the mean f1
bertscore_res = bertscore.compute(predictions=predictions, references=references, lang="en")
avg_f1 = sum(bertscore_res["f1"]) / len(bertscore_res["f1"])
print(f"BERTScore (F1): {avg_f1:.4f}")

# ─── BLEU ────────────────────────────────────────────────────────────────────────
# pred_tokens = [p.split() for p in predictions]
# ref_tokens  = [[[r.split()]] for r in references]
# bleu_res    = bleu.compute(predictions=pred_tokens, references=ref_tokens)
# print(f"BLEU    : {bleu_res['bleu']:.4f}")


