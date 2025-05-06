# evaluate_classification.py

import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics import (
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from models.noteworthy_extractor import NoteworthyExtractor

# --- SETTINGS (must match your training code) ---
MODEL_PATH        = "models/noteworthy_extractor.pth"
SPLITS_JSON       = "data/session_splits.json"
CLASSIFIED_FOLDER = "data/classified_utterances"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 15 subsections, same order/spelling as in train code
SUBSECTIONS = [
    "Presenting Problem / Chief Complaint",
    "Trauma History",
    "Substance Use History",
    "History of Present Illness (HPI)",
    "Medical and Psychiatric History",
    "Psychosocial History",
    "Risk Assessment",
    "Mental Health Observations",
    "Physiological Observations",
    "Current Functional Status",
    "Diagnostic Impressions",
    "Progress Evaluation",
    "Medications",
    "Therapeutic Interventions",
    "Next Steps"
]
SUB2IDX = {s: i for i, s in enumerate(SUBSECTIONS)}

def label_to_multihot(label_str: str) -> np.ndarray:
    """Convert comma-separated labels into a 15-dim multi-hot vector."""
    vec = np.zeros(len(SUBSECTIONS), dtype=int)
    for lbl in [x.strip() for x in label_str.split(",")]:
        idx = SUB2IDX.get(lbl)
        if idx is not None:
            vec[idx] = 1
    return vec

def tokenize(text: str, tokenizer: BertTokenizer):
    """Tokenize one utterance exactly as in training."""
    if not isinstance(text, str) or not text.strip():
        text = "Nothing reported"
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

def main():
    # 1) Load test split
    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)
    test_files = splits.get("test", [])
    if not test_files:
        raise RuntimeError(f"No test files found in {SPLITS_JSON}")

    # 2) Load model & tokenizer
    print(f"\nUsing device: {DEVICE}\n")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = NoteworthyExtractor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3) Prepare accumulators
    y_true_list   = []
    y_pred_list   = []
    total_utts    = 0
    total_correct = 0
    num_files     = len(test_files)

    # 4) Iterate & predict
    for i, fp in enumerate(sorted(test_files), 1):
        base = os.path.basename(fp)
        if not os.path.exists(fp):
            print(f"[{i}/{num_files}] ⚠️  Missing file, skipping: {base}")
            continue

        df = pd.read_csv(fp, encoding="utf-8-sig")
        df = df[df["Classified_Subsection"] != "Insignificant"]
        file_utts    = len(df)
        file_correct = 0

        print(f"[{i}/{num_files}] Processing `{base}` — {file_utts} utterances")

        for _, row in df.iterrows():
            gt_vec = label_to_multihot(row["Classified_Subsection"])
            input_ids, attn_mask = tokenize(row["Utterance"], tokenizer)
            with torch.no_grad():
                logits = model(input_ids, attn_mask)
                pred_vec = (torch.sigmoid(logits) > 0.7).int().cpu().numpy().squeeze()

            # exact-match?
            if np.array_equal(gt_vec, pred_vec):
                file_correct += 1

            # accumulate for global metrics
            y_true_list.append(gt_vec)
            y_pred_list.append(pred_vec)

        total_utts    += file_utts
        total_correct += file_correct
        print(f"    → Correct in this file: {file_correct}/{file_utts}\n")

    # 5) Exact-match accuracy
    overall_acc = total_correct / total_utts if total_utts else 0.0
    print("✅ Exact-match evaluation complete\n")
    print(f"  Total utterances : {total_utts}")
    print(f"  Exactly correct  : {total_correct}")
    print(f"  Accuracy (exact) : {overall_acc:.4f}\n")

    # 6) Convert lists to arrays for other metrics
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # 7) Hamming loss
    hl = hamming_loss(y_true, y_pred)
    print(f"=== Hamming loss ===\n{hl:.4f}\n")

    # 8) Micro-averaged precision, recall, F1
    p_micro = precision_score(y_true, y_pred, average="micro")
    r_micro = recall_score(   y_true, y_pred, average="micro")
    f1_micro= f1_score(      y_true, y_pred, average="micro")
    print("=== Micro-avg Precision, Recall, F1 ===")
    print(f"Precision: {p_micro:.4f}")
    print(f"Recall   : {r_micro:.4f}")
    print(f"F1       : {f1_micro:.4f}\n")

    # 9) Detailed per-label report
    print("=== Classification report (per-label) ===")
    print(classification_report(
        y_true,
        y_pred,
        target_names=SUBSECTIONS,
        zero_division=0
    ))

if __name__ == "__main__":
    main()
