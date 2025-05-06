#!/usr/bin/env python3
"""
generate_single_summary_bart.py

Usage:
    python generate_single_summary_bart.py SESSION_ID

Example:
    python generate_single_summary_bart.py 300_TRANSCRIPT
"""

import os
import sys
import torch
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration

# ——— Configuration ——————————————————————————————————————————————————————————
TEST_INPUT_DIR = "data/test_intermediate_files"
TARGET_DIR     = "data/Soap_notes"
MODEL_PATH     = "models/bart_summarizer_session_level.pth"

SUBSECTION_LIST = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History",
    "History of Present Illness (HPI)", "Medical and Psychiatric History", "Psychosocial History",
    "Risk Assessment", "Mental Health Observations", "Physiological Observations",
    "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_input_text(session_id: str) -> str:
    """Read the grouped-utterances CSV and build the model input string."""
    path = os.path.join(TEST_INPUT_DIR, f"{session_id}_utterances_grouping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    lines = []
    for subsec in SUBSECTION_LIST:
        mask = df["Subsection"] == subsec
        if mask.any():
            utt = df.loc[mask, "Grouped_Utterances"].iloc[0].strip()
            text = utt if utt else "Nothing reported"
        else:
            text = "Nothing reported"
        lines.append(f"{subsec}: {text}")
    return "\n".join(lines)


def build_reference_text(session_id: str) -> str:
    """Read the reference CSV and build the ground-truth summary string."""
    path = os.path.join(TARGET_DIR, f"{session_id}_output.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    lines = []
    for subsec in SUBSECTION_LIST:
        if subsec in df.columns:
            val = str(df[subsec].iloc[0]).strip()
            text = val if val and val != "nan" else "Nothing reported"
        else:
            text = "Nothing reported"
        lines.append(f"{subsec}: {text}")
    return "\n".join(lines)


def generate_summary(input_text: str,
                     tokenizer: BartTokenizer,
                     model: BartForConditionalGeneration) -> str:
    """Tokenize, generate with BART, and decode the output."""
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=1024
    ).to(DEVICE)

    output_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_length=256,
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True,
        min_length=125,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    session_id = sys.argv[1]
    print(f"\n→ Session: {session_id}\n")

    # 1) Build input & reference strings
    input_text     = build_input_text(session_id)
    reference_text = build_reference_text(session_id)

    # 2) Load tokenizer & model
    print(f"Loading BART model to {DEVICE}...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # 3) Generate
    summary = generate_summary(input_text, tokenizer, model)

    # 4) Print
    print("=== Input (truncated) ===")
    print(input_text[:500] + ("..." if len(input_text) > 500 else ""))
    print("\n=== Generated Summary ===")
    print(summary)
    print("\n=== Reference Summary ===")
    print(reference_text)
    print("\n")

if __name__ == "__main__":
    main()
