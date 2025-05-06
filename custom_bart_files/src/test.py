'''import os
import json
import time
import logging
import torch
from torch.utils.data import DataLoader
from transformers import BartConfig, AutoTokenizer
from .custom_bart_model import CustomBartForConditionalGeneration
from .train_utils import compute_ner_penalty
from ner_module.ner import load_model, perform_ner_on_tokens
from rouge_score import rouge_scorer
import pandas as pd

logging.basicConfig(level=logging.INFO)

# --- Helper Functions for Filename Adjustment ---
def adjust_input_filename(filename):
    if "utterances_grouping" not in filename:
        return filename.replace(".csv", "_utterances_grouping.csv")
    return filename

def get_target_filename(filename):
    if "utterances_grouping" in filename:
        return filename.replace("utterances_grouping", "output")
    else:
        return filename.replace(".csv", "_output.csv")

# --- Utility Function to Pad/Truncate Lists ---
def pad_list(lst, target_length, pad_value=0):
    if len(lst) < target_length:
        return lst + [pad_value] * (target_length - len(lst))
    else:
        return lst[:target_length]

# --- Processing Functions for a Single Session ---
def process_session_file(input_path, tokenizer, ner_model, section_mapping, subsection_mapping, max_length=1024):
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Error reading input CSV {input_path}: {e}")
        raise e
    all_tokens = []
    all_ner_flags = []
    all_section_ids = []
    all_subsection_ids = []
    input_entity_set = set()
    for idx, row in df.iterrows():
        subsection = row['Subsection']
        utterance = row['Grouped_Utterances']
        if utterance.strip().lower() == "nothing reported":
            utterance = ""
        tokens = tokenizer.tokenize(utterance)
        tokens = [token.lstrip("Ġ") for token in tokens]
        tokens = [token if token.strip() != "" else "[UNK]" for token in tokens]
        if tokens:
            try:
                ner_flags, entity_set = perform_ner_on_tokens(ner_model, tokens, return_entities=True)
                ner_flags = [1 if ((isinstance(f, str) and f != "O") or (isinstance(f, int) and f != 0)) else 0 for f in ner_flags]
            except Exception as e:
                logging.error(f"Error processing NER for row {idx} in {input_path}: {e}")
                ner_flags, entity_set = [], set()
        else:
            ner_flags, entity_set = [], set()
        all_tokens.extend(tokens)
        sec_id = section_mapping.get(subsection, 0)
        subsec_id = subsection_mapping.get(subsection, 0)
        all_section_ids.extend([sec_id] * len(tokens))
        all_subsection_ids.extend([subsec_id] * len(tokens))
        all_ner_flags.extend(ner_flags)
        input_entity_set = input_entity_set.union(entity_set)
    logging.info(f"Processed {input_path}. Total tokens: {len(all_tokens)}")
    token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    token_ids = pad_list(token_ids, max_length, tokenizer.pad_token_id)
    input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
    section_ids_tensor = torch.tensor(pad_list(all_section_ids, max_length, 0), dtype=torch.long)
    subsection_ids_tensor = torch.tensor(pad_list(all_subsection_ids, max_length, 0), dtype=torch.long)
    ner_mask_tensor = torch.tensor(pad_list(all_ner_flags, max_length, 0), dtype=torch.long)
    return input_ids_tensor, attention_mask_tensor, section_ids_tensor, subsection_ids_tensor, ner_mask_tensor, input_entity_set

def process_target_file(target_path, tokenizer, max_target_length=512):
    #print("in process target file ")
    """
    Updated to read from all 15 columns. For each column, this function concatenates the column header, ": ", and its first-row value,
    then adds a newline. Finally, it strips off any 'Ġ' markers for consistency.
    """
    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        logging.error(f"Error reading target CSV {target_path}: {e}")
        raise e
    final_summary = ""
    
    for col in df.columns:
        value = str(df[col].iloc[0])
        final_summary += f"{col}: {value}\n"
    # Remove Ġ markers
    final_summary = final_summary.replace("Ġ", "")
    target_encoding = tokenizer(final_summary, return_tensors="pt", max_length=max_target_length, truncation=True, padding="max_length")
    target_ids = target_encoding["input_ids"].squeeze(0)
    if target_ids[0].item() == tokenizer.bos_token_id:
       target_ids = target_ids[1:]
       #print(f"target_ids shape: {target_ids.shape}")
       target_ids = torch.cat([target_ids, torch.tensor([tokenizer.pad_token_id], device=target_ids.device)], dim=0) 
    eos_token = tokenizer.eos_token_id
    if eos_token not in target_ids:
        target_ids[-1] = eos_token
    else:
        eos_pos = (target_ids == eos_token).nonzero(as_tuple=True)[0]
        if eos_pos[-1] != target_ids.size(0) - 1:
            # move EOS to end
            target_ids[eos_pos[-1]] = target_ids[-1]
            target_ids[-1] = eos_token  
    return target_ids

# --- PyTorch Dataset Class ---
class SessionDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, tokenizer, ner_model, section_mapping, subsection_mapping, input_dir, target_dir, max_length=1024, max_target_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.ner_model = ner_model
        self.section_mapping = section_mapping
        self.subsection_mapping = subsection_mapping
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.max_length = max_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        base_filename = os.path.basename(self.file_paths[idx])
        input_filename = adjust_input_filename(base_filename)
        target_filename = get_target_filename(base_filename)
        input_file = os.path.join(self.input_dir, input_filename)
        target_file = os.path.join(self.target_dir, target_filename)
        input_ids, attn_mask, section_ids, subsection_ids, ner_mask, input_entity_set = process_session_file(
            input_file, self.tokenizer, self.ner_model, self.section_mapping, self.subsection_mapping, self.max_length
        )
        target_ids = process_target_file(target_file, self.tokenizer, self.max_target_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "section_ids": section_ids,
            "subsection_ids": subsection_ids,
            "ner_mask": ner_mask,
            "target_ids": target_ids,
            "input_entity_set": list(input_entity_set)
        }

def evaluate_model(model, dataloader, device, max_length=512):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            section_ids = batch["section_ids"].to(device)
            subsection_ids = batch["subsection_ids"].to(device)
            ner_mask = batch["ner_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            generated_text = model.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                section_ids=section_ids,
                subsection_ids=subsection_ids,
                ner_mask=ner_mask,
                max_length=max_length
            )
            if batch_idx < 3:
                target_text = model.tokenizer.decode(target_ids.squeeze(0).tolist(), skip_special_tokens=True)
                generated_token_ids = model.tokenizer.encode(generated_text)
                logging.info(f"DEBUG: Sample {batch_idx} target text:\n{target_text}")
                logging.info(f"DEBUG: Sample {batch_idx} generated text:\n{generated_text}")
                logging.info(f"DEBUG: Sample {batch_idx} generated token ids: {generated_token_ids}")
            target_text = model.tokenizer.decode(target_ids.squeeze(0).tolist(), skip_special_tokens=True)
            scores = scorer.score(target_text, generated_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Evaluated {batch_idx+1}/{len(dataloader)} test sessions.")
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    return avg_rouge1, avg_rouge2, avg_rougeL

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    splits_path = os.path.join("data", "session_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_files = splits["test"]

    test_input_dir = os.path.join("data", "test_intermediate_files")
    target_dir = os.path.join("data", "Soap_notes")

    section_mapping = {
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
    subsection_mapping = {name: idx for idx, name in enumerate(section_mapping.keys())}

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    logging.info("Loaded pretrained BART tokenizer.")
    ner_model = load_model()
    logging.info("Loaded custom NER model.")

    test_dataset = SessionDataset(test_files, tokenizer, ner_model, section_mapping, subsection_mapping,
                                  input_dir=test_input_dir, target_dir=target_dir, max_length=1024, max_target_length=512)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    checkpoint_path = os.path.join("checkpoints", "model_epoch10.pt")
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint {checkpoint_path} not found.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.tokenizer = tokenizer
    model.ner_model = ner_model
    model.to(device)
    logging.info("Loaded model from checkpoint and moved to device.")

    avg_rouge1, avg_rouge2, avg_rougeL = evaluate_model(model, test_loader, device)
    logging.info(f"Test ROUGE scores: ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")

if __name__ == "__main__":
    main()
'''
'''import os
import json
import time
import logging
import torch
from torch.utils.data import DataLoader
from transformers import BartConfig, AutoTokenizer
from .custom_bart_model import CustomBartForConditionalGeneration
from .train_utils import compute_ner_penalty
from ner_module.ner import load_model, perform_ner_on_tokens
from rouge_score import rouge_scorer
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


logging.basicConfig(level=logging.INFO)

# --- Helper Functions for Filename Adjustment ---
def adjust_input_filename(filename):
    if "utterances_grouping" not in filename:
        return filename.replace(".csv", "_utterances_grouping.csv")
    return filename

def get_target_filename(filename):
    if "utterances_grouping" in filename:
        return filename.replace("utterances_grouping", "output")
    else:
        return filename.replace(".csv", "_output.csv")

# --- Utility Function to Pad/Truncate Lists ---
def pad_list(lst, target_length, pad_value=0):
    if len(lst) < target_length:
        return lst + [pad_value] * (target_length - len(lst))
    else:
        return lst[:target_length]

# --- Processing Functions for a Single Session ---
def process_session_file(input_path, tokenizer, ner_model, section_mapping, subsection_mapping, max_length=1024):
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Error reading input CSV {input_path}: {e}")
        raise e
    all_tokens = []
    all_ner_flags = []
    all_section_ids = []
    all_subsection_ids = []
    input_entity_set = set()
    for idx, row in df.iterrows():
        subsection = row['Subsection']
        utterance = row['Grouped_Utterances']
        if utterance.strip().lower() == "nothing reported":
            utterance = ""
        tokens = tokenizer.tokenize(utterance)
        tokens = [token.lstrip("Ġ") for token in tokens]
        tokens = [token if token.strip() != "" else "[UNK]" for token in tokens]
        if tokens:
            try:
                ner_flags, entity_set = perform_ner_on_tokens(ner_model, tokens, return_entities=True)
                ner_flags = [1 if ((isinstance(f, str) and f != "O") or (isinstance(f, int) and f != 0)) else 0 for f in ner_flags]
            except Exception as e:
                logging.error(f"Error processing NER for row {idx} in {input_path}: {e}")
                ner_flags, entity_set = [], set()
        else:
            ner_flags, entity_set = [], set()
        all_tokens.extend(tokens)
        sec_id = section_mapping.get(subsection, 0)
        subsec_id = subsection_mapping.get(subsection, 0)
        all_section_ids.extend([sec_id] * len(tokens))
        all_subsection_ids.extend([subsec_id] * len(tokens))
        all_ner_flags.extend(ner_flags)
        input_entity_set = input_entity_set.union(entity_set)
    logging.info(f"Processed {input_path}. Total tokens: {len(all_tokens)}")
    token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    token_ids = pad_list(token_ids, max_length, tokenizer.pad_token_id)
    input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
    section_ids_tensor = torch.tensor(pad_list(all_section_ids, max_length, 0), dtype=torch.long)
    subsection_ids_tensor = torch.tensor(pad_list(all_subsection_ids, max_length, 0), dtype=torch.long)
    ner_mask_tensor = torch.tensor(pad_list(all_ner_flags, max_length, 0), dtype=torch.long)
    return input_ids_tensor, attention_mask_tensor, section_ids_tensor, subsection_ids_tensor, ner_mask_tensor, input_entity_set

def process_target_file(target_path, tokenizer, max_target_length=512):
    #print("in process target file ")
    """
    Updated to read from all 15 columns. For each column, this function concatenates the column header, ": ", and its first-row value,
    then adds a newline. Finally, it strips off any 'Ġ' markers for consistency.
    """
    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        logging.error(f"Error reading target CSV {target_path}: {e}")
        raise e
    final_summary = ""
    
    for col in df.columns:
        value = str(df[col].iloc[0])
        final_summary += f"{col}: {value}\n"
    # Remove Ġ markers
    final_summary = final_summary.replace("Ġ", "")
    target_encoding = tokenizer(final_summary, return_tensors="pt", max_length=max_target_length, truncation=True, padding="max_length")
    target_ids = target_encoding["input_ids"].squeeze(0)
    if target_ids[0].item() == tokenizer.bos_token_id:
       target_ids = target_ids[1:]
       #print(f"target_ids shape: {target_ids.shape}")
       target_ids = torch.cat([target_ids, torch.tensor([tokenizer.pad_token_id], device=target_ids.device)], dim=0) 
    eos_token = tokenizer.eos_token_id
    if eos_token not in target_ids:
        target_ids[-1] = eos_token
    else:
        eos_pos = (target_ids == eos_token).nonzero(as_tuple=True)[0]
        if eos_pos[-1] != target_ids.size(0) - 1:
            # move EOS to end
            target_ids[eos_pos[-1]] = target_ids[-1]
            target_ids[-1] = eos_token  
    return target_ids

# --- PyTorch Dataset Class ---
class SessionDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, tokenizer, ner_model, section_mapping, subsection_mapping, input_dir, target_dir, max_length=1024, max_target_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.ner_model = ner_model
        self.section_mapping = section_mapping
        self.subsection_mapping = subsection_mapping
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.max_length = max_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        base_filename = os.path.basename(self.file_paths[idx])
        input_filename = adjust_input_filename(base_filename)
        target_filename = get_target_filename(base_filename)
        input_file = os.path.join(self.input_dir, input_filename)
        target_file = os.path.join(self.target_dir, target_filename)
        input_ids, attn_mask, section_ids, subsection_ids, ner_mask, input_entity_set = process_session_file(
            input_file, self.tokenizer, self.ner_model, self.section_mapping, self.subsection_mapping, self.max_length
        )
        target_ids = process_target_file(target_file, self.tokenizer, self.max_target_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "section_ids": section_ids,
            "subsection_ids": subsection_ids,
            "ner_mask": ner_mask,
            "target_ids": target_ids,
            "input_entity_set": list(input_entity_set)
        }
def evaluate_model(model, dataloader, device, max_length=512):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    meteor_scores = []

    smooth = SmoothingFunction().method4

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            section_ids = batch["section_ids"].to(device)
            subsection_ids = batch["subsection_ids"].to(device)
            ner_mask = batch["ner_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            generated_text = model.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                section_ids=section_ids,
                subsection_ids=subsection_ids,
                ner_mask=ner_mask,
                max_length=max_length
            )

            target_text = model.tokenizer.decode(target_ids.squeeze(0).tolist(), skip_special_tokens=True)

            # ROUGE
            scores = scorer.score(target_text, generated_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            # BLEU
            ref_tokens = target_text.lower().split()
            gen_tokens = generated_text.lower().split()
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth)
            bleu_scores.append(bleu)

            # METEOR
            meteor = meteor_score([ref_tokens], gen_tokens)
            meteor_scores.append(meteor)

            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Evaluated {batch_idx+1}/{len(dataloader)} test sessions.")

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    return avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu, avg_meteor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    splits_path = os.path.join("data", "session_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_files = splits["test"]

    test_input_dir = os.path.join("data", "test_intermediate_files")
    target_dir = os.path.join("data", "Soap_notes")

    section_mapping = {
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
    subsection_mapping = {name: idx for idx, name in enumerate(section_mapping.keys())}

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    logging.info("Loaded pretrained BART tokenizer.")
    ner_model = load_model()
    logging.info("Loaded custom NER model.")

    test_dataset = SessionDataset(test_files, tokenizer, ner_model, section_mapping, subsection_mapping,
                                  input_dir=test_input_dir, target_dir=target_dir, max_length=1024, max_target_length=512)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    checkpoint_path = os.path.join("checkpoints", "model_epoch10.pt")
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint {checkpoint_path} not found.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.tokenizer = tokenizer
    model.ner_model = ner_model
    model.to(device)
    logging.info("Loaded model from checkpoint and moved to device.")

    avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu, avg_meteor = evaluate_model(model, test_loader, device)
    logging.info(f"Test ROUGE scores: ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")
    logging.info(f"Test BLEU: {avg_bleu:.4f}")
    logging.info(f"Test METEOR: {avg_meteor:.4f}")

if __name__ == "__main__":
    main()'''

'''import os
import json
import time
import logging
import torch
from torch.utils.data import DataLoader
from transformers import BartConfig, AutoTokenizer
from .custom_bart_model import CustomBartForConditionalGeneration
from .train_utils import compute_ner_penalty
from ner_module.ner import load_model, perform_ner_on_tokens
from rouge_score import rouge_scorer
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score  # ← Add this at the top


logging.basicConfig(level=logging.INFO)

# --- Helper Functions for Filename Adjustment ---
def adjust_input_filename(filename):
    if "utterances_grouping" not in filename:
        return filename.replace(".csv", "_utterances_grouping.csv")
    return filename

def get_target_filename(filename):
    if "utterances_grouping" in filename:
        return filename.replace("utterances_grouping", "output")
    else:
        return filename.replace(".csv", "_output.csv")

# --- Utility Function to Pad/Truncate Lists ---
def pad_list(lst, target_length, pad_value=0):
    if len(lst) < target_length:
        return lst + [pad_value] * (target_length - len(lst))
    else:
        return lst[:target_length]

# --- Processing Functions for a Single Session ---
def process_session_file(input_path, tokenizer, ner_model, section_mapping, subsection_mapping, max_length=1024):
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Error reading input CSV {input_path}: {e}")
        raise e
    all_tokens = []
    all_ner_flags = []
    all_section_ids = []
    all_subsection_ids = []
    input_entity_set = set()
    for idx, row in df.iterrows():
        subsection = row['Subsection']
        utterance = row['Grouped_Utterances']
        if utterance.strip().lower() == "nothing reported":
            utterance = ""
        tokens = tokenizer.tokenize(utterance)
        tokens = [token.lstrip("Ġ") for token in tokens]
        tokens = [token if token.strip() != "" else "[UNK]" for token in tokens]
        if tokens:
            try:
                ner_flags, entity_set = perform_ner_on_tokens(ner_model, tokens, return_entities=True)
                ner_flags = [1 if ((isinstance(f, str) and f != "O") or (isinstance(f, int) and f != 0)) else 0 for f in ner_flags]
            except Exception as e:
                logging.error(f"Error processing NER for row {idx} in {input_path}: {e}")
                ner_flags, entity_set = [], set()
        else:
            ner_flags, entity_set = [], set()
        all_tokens.extend(tokens)
        sec_id = section_mapping.get(subsection, 0)
        subsec_id = subsection_mapping.get(subsection, 0)
        all_section_ids.extend([sec_id] * len(tokens))
        all_subsection_ids.extend([subsec_id] * len(tokens))
        all_ner_flags.extend(ner_flags)
        input_entity_set = input_entity_set.union(entity_set)
    logging.info(f"Processed {input_path}. Total tokens: {len(all_tokens)}")
    token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    token_ids = pad_list(token_ids, max_length, tokenizer.pad_token_id)
    input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
    section_ids_tensor = torch.tensor(pad_list(all_section_ids, max_length, 0), dtype=torch.long)
    subsection_ids_tensor = torch.tensor(pad_list(all_subsection_ids, max_length, 0), dtype=torch.long)
    ner_mask_tensor = torch.tensor(pad_list(all_ner_flags, max_length, 0), dtype=torch.long)
    return input_ids_tensor, attention_mask_tensor, section_ids_tensor, subsection_ids_tensor, ner_mask_tensor, input_entity_set

def process_target_file(target_path, tokenizer, max_target_length=512):
    #print("in process target file ")
    """
    Updated to read from all 15 columns. For each column, this function concatenates the column header, ": ", and its first-row value,
    then adds a newline. Finally, it strips off any 'Ġ' markers for consistency.
    """
    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        logging.error(f"Error reading target CSV {target_path}: {e}")
        raise e
    final_summary = ""
    
    for col in df.columns:
        value = str(df[col].iloc[0])
        final_summary += f"{col}: {value}\n"
    # Remove Ġ markers
    final_summary = final_summary.replace("Ġ", "")
    target_encoding = tokenizer(final_summary, return_tensors="pt", max_length=max_target_length, truncation=True, padding="max_length")
    target_ids = target_encoding["input_ids"].squeeze(0)
    if target_ids[0].item() == tokenizer.bos_token_id:
       target_ids = target_ids[1:]
       #print(f"target_ids shape: {target_ids.shape}")
       target_ids = torch.cat([target_ids, torch.tensor([tokenizer.pad_token_id], device=target_ids.device)], dim=0) 
    eos_token = tokenizer.eos_token_id
    if eos_token not in target_ids:
        target_ids[-1] = eos_token
    else:
        eos_pos = (target_ids == eos_token).nonzero(as_tuple=True)[0]
        if eos_pos[-1] != target_ids.size(0) - 1:
            # move EOS to end
            target_ids[eos_pos[-1]] = target_ids[-1]
            target_ids[-1] = eos_token  
    return target_ids

# --- PyTorch Dataset Class ---
class SessionDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, tokenizer, ner_model, section_mapping, subsection_mapping, input_dir, target_dir, max_length=1024, max_target_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.ner_model = ner_model
        self.section_mapping = section_mapping
        self.subsection_mapping = subsection_mapping
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.max_length = max_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        base_filename = os.path.basename(self.file_paths[idx])
        input_filename = adjust_input_filename(base_filename)
        target_filename = get_target_filename(base_filename)
        input_file = os.path.join(self.input_dir, input_filename)
        target_file = os.path.join(self.target_dir, target_filename)
        input_ids, attn_mask, section_ids, subsection_ids, ner_mask, input_entity_set = process_session_file(
            input_file, self.tokenizer, self.ner_model, self.section_mapping, self.subsection_mapping, self.max_length
        )
        target_ids = process_target_file(target_file, self.tokenizer, self.max_target_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "section_ids": section_ids,
            "subsection_ids": subsection_ids,
            "ner_mask": ner_mask,
            "target_ids": target_ids,
            "input_entity_set": list(input_entity_set)
        }
def evaluate_model(model, dataloader, device, max_length=512):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    meteor_scores = []

    all_preds, all_refs = [] , []

    smooth = SmoothingFunction().method4

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            section_ids = batch["section_ids"].to(device)
            subsection_ids = batch["subsection_ids"].to(device)
            ner_mask = batch["ner_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            generated_text = model.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                section_ids=section_ids,
                subsection_ids=subsection_ids,
                ner_mask=ner_mask,
                max_length=max_length
            )

            target_text = model.tokenizer.decode(target_ids.squeeze(0).tolist(), skip_special_tokens=True)

            # Store for BERTScore
            all_preds.append(generated_text)
            all_refs.append(target_text)

            # ROUGE
            scores = scorer.score(target_text, generated_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            # BLEU
            ref_tokens = target_text.lower().split()
            gen_tokens = generated_text.lower().split()
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth)
            bleu_scores.append(bleu)

            # METEOR
            meteor = meteor_score([ref_tokens], gen_tokens)
            meteor_scores.append(meteor)

            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Evaluated {batch_idx+1}/{len(dataloader)} test sessions.")

    # BERTScore
    P, R, F1 = bert_score(all_preds, all_refs, lang="en", model_type="roberta-large", device=device, rescale_with_baseline=True)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    avg_bleu   = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_bertscore = F1.mean().item()

    return avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu, avg_meteor, avg_bertscore


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    splits_path = os.path.join("data", "session_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_files = splits["test"]

    test_input_dir = os.path.join("data", "test_intermediate_files")
    target_dir = os.path.join("data", "Soap_notes")

    section_mapping = {
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
    subsection_mapping = {name: idx for idx, name in enumerate(section_mapping.keys())}

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    logging.info("Loaded pretrained BART tokenizer.")
    ner_model = load_model()
    logging.info("Loaded custom NER model.")

    test_dataset = SessionDataset(test_files, tokenizer, ner_model, section_mapping, subsection_mapping,
                                  input_dir=test_input_dir, target_dir=target_dir, max_length=1024, max_target_length=512)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    checkpoint_path = os.path.join("checkpoints", "model_epoch3.pt")
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint {checkpoint_path} not found.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.tokenizer = tokenizer
    model.ner_model = ner_model
    model.to(device)
    logging.info("Loaded model from checkpoint and moved to device.")

    avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu, avg_meteor, avg_bertscore = evaluate_model(model, test_loader, device)
    logging.info(f"Test ROUGE scores: ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")
    logging.info(f"Test BLEU: {avg_bleu:.4f}")
    logging.info(f"Test METEOR: {avg_meteor:.4f}")
    logging.info(f"Test BERTScore-F1: {avg_bertscore:.4f}")

if __name__ == "__main__":
    main()'''

import os
import json
import time
import logging
import torch
from torch.utils.data import DataLoader
from transformers import BartConfig, AutoTokenizer
from .custom_bart_model import CustomBartForConditionalGeneration
from .train_utils import compute_ner_penalty
from ner_module.ner import load_model, perform_ner_on_tokens
from rouge_score import rouge_scorer
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score  # ← Add this at the top


logging.basicConfig(level=logging.INFO)

# --- Helper Functions for Filename Adjustment ---
def adjust_input_filename(filename):
    if "utterances_grouping" not in filename:
        return filename.replace(".csv", "_utterances_grouping.csv")
    return filename

def get_target_filename(filename):
    if "utterances_grouping" in filename:
        return filename.replace("utterances_grouping", "output")
    else:
        return filename.replace(".csv", "_output.csv")

# --- Utility Function to Pad/Truncate Lists ---
def pad_list(lst, target_length, pad_value=0):
    if len(lst) < target_length:
        return lst + [pad_value] * (target_length - len(lst))
    else:
        return lst[:target_length]

# --- Processing Functions for a Single Session ---
def process_session_file(input_path, tokenizer, ner_model, section_mapping, subsection_mapping, max_length=1024):
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Error reading input CSV {input_path}: {e}")
        raise e
    all_tokens = []
    all_ner_flags = []
    all_section_ids = []
    all_subsection_ids = []
    input_entity_set = set()
    for idx, row in df.iterrows():
        subsection = row['Subsection']
        utterance = row['Grouped_Utterances']
        if utterance.strip().lower() == "nothing reported":
            utterance = ""
        tokens = tokenizer.tokenize(utterance)
        tokens = [token.lstrip("Ġ") for token in tokens]
        tokens = [token if token.strip() != "" else "[UNK]" for token in tokens]
        if tokens:
            try:
                ner_flags, entity_set = perform_ner_on_tokens(ner_model, tokens, return_entities=True)
                ner_flags = [1 if ((isinstance(f, str) and f != "O") or (isinstance(f, int) and f != 0)) else 0 for f in ner_flags]
            except Exception as e:
                logging.error(f"Error processing NER for row {idx} in {input_path}: {e}")
                ner_flags, entity_set = [], set()
        else:
            ner_flags, entity_set = [], set()
        all_tokens.extend(tokens)
        sec_id = section_mapping.get(subsection, 0)
        subsec_id = subsection_mapping.get(subsection, 0)
        all_section_ids.extend([sec_id] * len(tokens))
        all_subsection_ids.extend([subsec_id] * len(tokens))
        all_ner_flags.extend(ner_flags)
        input_entity_set = input_entity_set.union(entity_set)
    logging.info(f"Processed {input_path}. Total tokens: {len(all_tokens)}")
    token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    token_ids = pad_list(token_ids, max_length, tokenizer.pad_token_id)
    input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
    section_ids_tensor = torch.tensor(pad_list(all_section_ids, max_length, 0), dtype=torch.long)
    subsection_ids_tensor = torch.tensor(pad_list(all_subsection_ids, max_length, 0), dtype=torch.long)
    ner_mask_tensor = torch.tensor(pad_list(all_ner_flags, max_length, 0), dtype=torch.long)
    return input_ids_tensor, attention_mask_tensor, section_ids_tensor, subsection_ids_tensor, ner_mask_tensor, input_entity_set

def process_target_file(target_path, tokenizer, max_target_length=512):
    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        logging.error(f"Error reading target CSV {target_path}: {e}")
        raise e
    final_summary = ""
    for col in df.columns:
        value = str(df[col].iloc[0])
        final_summary += f"{col}: {value}\n"
    final_summary = final_summary.replace("Ġ", "")
    target_encoding = tokenizer(final_summary, return_tensors="pt", max_length=max_target_length, truncation=True, padding="max_length")
    target_ids = target_encoding["input_ids"].squeeze(0)
    if target_ids[0].item() == tokenizer.bos_token_id:
       target_ids = target_ids[1:]
       target_ids = torch.cat([target_ids, torch.tensor([tokenizer.pad_token_id], device=target_ids.device)], dim=0) 
    eos_token = tokenizer.eos_token_id
    if eos_token not in target_ids:
        target_ids[-1] = eos_token
    else:
        eos_pos = (target_ids == eos_token).nonzero(as_tuple=True)[0]
        if eos_pos[-1] != target_ids.size(0) - 1:
            target_ids[eos_pos[-1]] = target_ids[-1]
            target_ids[-1] = eos_token  
    return target_ids

# --- PyTorch Dataset Class ---
class SessionDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, tokenizer, ner_model, section_mapping, subsection_mapping, input_dir, target_dir, max_length=1024, max_target_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.ner_model = ner_model
        self.section_mapping = section_mapping
        self.subsection_mapping = subsection_mapping
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.max_length = max_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        base_filename = os.path.basename(self.file_paths[idx])
        input_filename = adjust_input_filename(base_filename)
        target_filename = get_target_filename(base_filename)
        input_file = os.path.join(self.input_dir, input_filename)
        target_file = os.path.join(self.target_dir, target_filename)
        input_ids, attn_mask, section_ids, subsection_ids, ner_mask, input_entity_set = process_session_file(
            input_file, self.tokenizer, self.ner_model, self.section_mapping, self.subsection_mapping, self.max_length
        )
        target_ids = process_target_file(target_file, self.tokenizer, self.max_target_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "section_ids": section_ids,
            "subsection_ids": subsection_ids,
            "ner_mask": ner_mask,
            "target_ids": target_ids,
            "input_entity_set": list(input_entity_set)
        }

def evaluate_model(model, dataloader, device, max_length=512):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    meteor_scores = []

    all_preds, all_refs = [] , []

    smooth = SmoothingFunction().method4
    start_time = time.time()  # Start time tracker for the entire evaluation process

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            section_ids = batch["section_ids"].to(device)
            subsection_ids = batch["subsection_ids"].to(device)
            ner_mask = batch["ner_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            generated_text = model.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                section_ids=section_ids,
                subsection_ids=subsection_ids,
                ner_mask=ner_mask,
                max_length=max_length
            )

            target_text = model.tokenizer.decode(target_ids.squeeze(0).tolist(), skip_special_tokens=True)

            # Store for BERTScore
            all_preds.append(generated_text)
            all_refs.append(target_text)

            # ROUGE
            scores = scorer.score(target_text, generated_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            # BLEU
            ref_tokens = target_text.lower().split()
            gen_tokens = generated_text.lower().split()
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth)
            bleu_scores.append(bleu)

            # METEOR
            meteor = meteor_score([ref_tokens], gen_tokens)
            meteor_scores.append(meteor)

            # Print generated text vs target text for sample files
            if batch_idx < 5:  # Print for first 5 samples
                logging.info(f"Sample {batch_idx+1} - Generated: {generated_text}")
                logging.info(f"Sample {batch_idx+1} - Ground Truth: {target_text}")

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                logging.info(f"Evaluated {batch_idx+1}/{len(dataloader)} test sessions. Elapsed: {elapsed:.2f}s")

    # BERTScore
    P, R, F1 = bert_score(all_preds, all_refs, lang="en", model_type="roberta-large", device=device, rescale_with_baseline=True)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    avg_bleu   = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_bertscore = F1.mean().item()

    return avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu, avg_meteor, avg_bertscore


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    splits_path = os.path.join("data", "session_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_files = splits["test"]

    test_input_dir = os.path.join("data", "test_intermediate_files")
    target_dir = os.path.join("data", "Soap_notes")

    section_mapping = {
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
    subsection_mapping = {name: idx for idx, name in enumerate(section_mapping.keys())}

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    logging.info("Loaded pretrained BART tokenizer.")
    ner_model = load_model()
    logging.info("Loaded custom NER model.")

    test_dataset = SessionDataset(test_files, tokenizer, ner_model, section_mapping, subsection_mapping,
                                  input_dir=test_input_dir, target_dir=target_dir, max_length=1024, max_target_length=512)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    checkpoint_path = os.path.join("checkpoints", "model_epoch5.pt")
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint {checkpoint_path} not found.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.tokenizer = tokenizer
    model.ner_model = ner_model
    model.to(device)
    logging.info("Loaded model from checkpoint and moved to device.")

    avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu, avg_meteor, avg_bertscore = evaluate_model(model, test_loader, device)
    logging.info(f"Test ROUGE scores: ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")
    logging.info(f"Test BLEU: {avg_bleu:.4f}")
    logging.info(f"Test METEOR: {avg_meteor:.4f}")
    logging.info(f"Test BERTScore-F1: {avg_bertscore:.4f}")

if __name__ == "__main__":
    main()
