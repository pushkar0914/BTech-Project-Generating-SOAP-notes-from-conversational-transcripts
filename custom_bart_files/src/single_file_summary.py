#!/usr/bin/env python3
"""
generate_single_summary_custom_bart.py

Usage:
    python generate_single_summary_custom_bart.py SESSION_ID

Example:
    python generate_single_summary_custom_bart.py 300_TRANSCRIPT
"""

import os
import sys
import logging
import torch
import pandas as pd
from transformers import BartConfig, AutoTokenizer
from .custom_bart_model import CustomBartForConditionalGeneration
from ner_module.ner import load_model, perform_ner_on_tokens

# ——— Configuration ——————————————————————————————————————————————————————————
TEST_INPUT_DIR = "data/test_intermediate_files"
TARGET_DIR     = "data/Soap_notes"
CHECKPOINT     = "checkpoints/model_epoch5.pt"   # adjust if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

# Section & subsection maps (must match your training)
SECTION_MAPPING = {
    "Presenting Problem / Chief Complaint": 0,
    "Trauma History": 0,
    "Substance Use History": 0,
    "History of Present Illness (HPI)": 0,
    "Medical and Psychiatric History": 0,
    "Psychosocial History": 0,
    "Risk Assessment": 0,
    "Mental Health Observations": 1,
    "Physiological Observations": 1,
    "Current Functional Status": 1,
    "Diagnostic Impressions": 2,
    "Progress Evaluation": 2,
    "Medications": 3,
    "Therapeutic Interventions": 3,
    "Next Steps": 3
}
SUBSECTION_MAPPING = {name: idx for idx, name in enumerate(SECTION_MAPPING.keys())}

SUBSECTION_LIST = list(SECTION_MAPPING.keys())

# ——— Helpers —————————————————————————————————————————————————————————————
def adjust_input_filename(fn: str) -> str:
    return fn.replace(".csv", "_utterances_grouping.csv") if "utterances_grouping" not in fn else fn

def get_target_filename(fn: str) -> str:
    return fn.replace(".csv", "_output.csv") if "output" not in fn else fn

def pad_list(lst, target_length, pad_value):
    return lst + [pad_value] * (target_length - len(lst)) if len(lst) < target_length else lst[:target_length]

def process_session_file(input_path, tokenizer, ner_model, sec_map, subsec_map, max_length=1024):
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    all_tokens, all_ner, all_sec_ids, all_sub_ids = [], [], [], []
    entities = set()
    for _, row in df.iterrows():
        subsec = row["Subsection"]
        utt   = row["Grouped_Utterances"].strip()
        if utt.lower() == "nothing reported":
            utt = ""
        toks = tokenizer.tokenize(utt)
        toks = [t.lstrip("Ġ") or "[UNK]" for t in toks]
        if toks:
            flags, ents = perform_ner_on_tokens(ner_model, toks, return_entities=True)
            flags = [1 if ((isinstance(f, str) and f!="O") or (isinstance(f,int) and f!=0)) else 0 for f in flags]
        else:
            flags, ents = [], set()
        all_tokens.extend(toks)
        all_ner.extend(flags)
        sid = sec_map.get(subsec, 0)
        ssid= subsec_map.get(subsec, 0)
        all_sec_ids.extend([sid]*len(toks))
        all_sub_ids.extend([ssid]*len(toks))
        entities |= ents

    # Convert tokens → IDs, pad/truncate
    ids = tokenizer.convert_tokens_to_ids(all_tokens)
    ids = pad_list(ids, max_length, tokenizer.pad_token_id)
    input_ids = torch.tensor([ids], device=DEVICE)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    sec_ids = pad_list(all_sec_ids, max_length, 0)
    sub_ids = pad_list(all_sub_ids,  max_length, 0)
    ner_mask = pad_list(all_ner,     max_length, 0)
    return (
        input_ids,
        attention_mask,
        torch.tensor([sec_ids], device=DEVICE),
        torch.tensor([sub_ids], device=DEVICE),
        torch.tensor([ner_mask],device=DEVICE),
        entities
    )

def process_target_file(target_path, tokenizer, max_len=512):
    df = pd.read_csv(target_path, encoding="utf-8-sig")
    text = ""
    for col in df.columns:
        text += f"{col}: {df[col].iloc[0]}\n"
    text = text.replace("Ġ", "")
    enc = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length")
    ids = enc["input_ids"].squeeze(0)
    # ensure EOS at end
    eos = tokenizer.eos_token_id
    if eos not in ids:
        ids[-1] = eos
    return ids.unsqueeze(0).to(DEVICE)

# ——— Main ————————————————————————————————————————————————————————————————
def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    session = sys.argv[1]
    base_fn = session + ".csv"

    # Build file paths
    in_fn  = adjust_input_filename(base_fn)
    tgt_fn = get_target_filename(base_fn)
    input_path  = os.path.join(TEST_INPUT_DIR, in_fn)
    target_path = os.path.join(TARGET_DIR,   tgt_fn)

    # Load tokenizer, NER & model
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    ner_model = load_model()
    config    = BartConfig.from_pretrained("facebook/bart-base")
    model     = CustomBartForConditionalGeneration(config).to(DEVICE)
    ckpt      = CHECKPOINT
    if not os.path.exists(ckpt):
        logging.error(f"Checkpoint not found: {ckpt}")
        return
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.tokenizer = tokenizer
    model.ner_model = ner_model
    model.eval()

    # Process inputs
    inp_ids, attn_mask, sec_ids, sub_ids, ner_mask, ents = process_session_file(
        input_path, tokenizer, ner_model, SECTION_MAPPING, SUBSECTION_MAPPING
    )
    tgt_ids = process_target_file(target_path, tokenizer)

    # Generate
    logging.info(f"Generating summary for {session} (entities: {len(ents)})")
    summary = model.generate_text(
        input_ids=inp_ids,
        attention_mask=attn_mask,
        section_ids=sec_ids,
        subsection_ids=sub_ids,
        ner_mask=ner_mask,
        max_length=512
    )

    # Decode reference
    ref = tokenizer.decode(tgt_ids.squeeze(0).tolist(), skip_special_tokens=True)

    # Print results
    print("\n=== Generated Summary ===\n")
    print(summary)
    print("\n=== Reference Summary ===\n")
    print(ref)
    print("\n")

if __name__ == "__main__":
    main()
