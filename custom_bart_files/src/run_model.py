'''
import os
import sys
import logging
import pandas as pd
import torch
from transformers import BartConfig, AutoTokenizer
from custom_bart_model import CustomBartForConditionalGeneration
from ner_module.ner import load_model, perform_ner_on_tokens

# -------------------------------
# CSV Processing Functions
# -------------------------------
def simple_tokenizer(text):
    """Simple whitespace tokenizer as a fallback."""
    return text.strip().split()

def process_row(subsection, utterance, ner_model, tokenizer):
    """Process one CSV row: tokenize utterance using the pretrained tokenizer,
       get the NER mask, and also extract the named entity set.
    """
    if utterance.strip().lower() == "nothing reported":
        utterance = ""
    # Tokenize using the pretrained tokenizer
    tokens = tokenizer.tokenize(utterance)
    tokens = [token.lstrip("Ġ") for token in tokens]
    tokens = [token if token.strip() != "" else "[UNK]" for token in tokens]

    if not tokens:
        return [], [], set()
    try:
        ner_flags, entity_set = perform_ner_on_tokens(ner_model, tokens, return_entities=True)
        # Convert any non-zero flag (or non-"O") to 1
        ner_flags = [1 if ((isinstance(f, str) and f != "O") or (isinstance(f, int) and f != 0)) else 0 for f in ner_flags]
    except Exception as e:
        logging.error(f"Error processing NER for subsection {subsection}: {e}")
        tokens, ner_flags, entity_set = [], [], set()
    
    print(f"\n--- Processing Row ---")
    print(f"Subsection: {subsection}")
    print(f"Utterance: {utterance}")
    print(f"Tokens: {tokens}")
    print(f"NER Mask: {ner_flags}")
    print(f"entity set :{entity_set}")
    return tokens, ner_flags, entity_set

def process_csv(csv_path, ner_model, tokenizer, section_mapping, subsection_mapping):
    """Read CSV and process each row, accumulating tokens, ID lists, and entity sets."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        sys.exit(1)
    all_tokens = []
    all_ner_flags = []
    all_section_ids = []
    all_subsection_ids = []
    input_entity_set = set()  # Accumulate union of entities
    structured_input = ""
    for idx, row in df.iterrows():
        subsection = row['Subsection']
        utterance = row['Grouped_Utterances']
        tokens, ner_flags, entity_set = process_row(subsection, utterance, ner_model, tokenizer)
        structured_input += f"[SUBSECTION] {subsection}: {utterance}\n"
        all_tokens.extend(tokens)
        sec_id = section_mapping.get(subsection, 0)
        subsec_id = subsection_mapping.get(subsection, 0)
        all_section_ids.extend([sec_id] * len(tokens))
        all_subsection_ids.extend([subsec_id] * len(tokens))
        all_ner_flags.extend(ner_flags)
        input_entity_set = input_entity_set.union(entity_set)
    print(f"input entity set : {input_entity_set}")
    print("\n--- Combined Structured Input ---")
    print(structured_input)
    return all_tokens, all_section_ids, all_subsection_ids, all_ner_flags, input_entity_set

def pad_list(lst, target_length, pad_value=0):
    """Utility function to pad or truncate a list to target_length."""
    if len(lst) < target_length:
        return lst + [pad_value] * (target_length - len(lst))
    else:
        return lst[:target_length]

# -------------------------------
# Main Run Function
# -------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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
    print("Loaded pretrained BART tokenizer.")

    ner_model = load_model()
    print("Loaded custom NER model.")

    csv_path = os.path.join("data", "364_TRANSCRIPT_utterances_grouping.csv")
    tokens_list, section_ids_list, subsection_ids_list, ner_flag_list, input_entity_set = process_csv(
        csv_path, ner_model, tokenizer, section_mapping, subsection_mapping
    )
    
    max_length = 1024
    if len(tokens_list) <= max_length:
        token_ids = tokenizer.convert_tokens_to_ids(tokens_list)
        token_ids = pad_list(token_ids, max_length, tokenizer.pad_token_id)
        # Fix: Remove extra list wrapping
        input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
        section_ids_tensor = torch.tensor(pad_list(section_ids_list, max_length, 0), dtype=torch.long)
        subsection_ids_tensor = torch.tensor(pad_list(subsection_ids_list, max_length, 0), dtype=torch.long)
        ner_mask_tensor = torch.tensor(pad_list(ner_flag_list, max_length, 0), dtype=torch.long)
    else:
        structured_text = " ".join(tokens_list)
        encoding = tokenizer(structured_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids_tensor = encoding["input_ids"]
        attention_mask_tensor = encoding["attention_mask"]
        section_ids_tensor = torch.tensor(section_ids_list[:max_length], dtype=torch.long)
        subsection_ids_tensor = torch.tensor(subsection_ids_list[:max_length], dtype=torch.long)
        ner_mask_tensor = torch.tensor(ner_flag_list[:max_length], dtype=torch.long)
    
    print("Input IDs tensor shape:", input_ids_tensor.shape)
    print("Attention mask tensor shape:", attention_mask_tensor.shape)
    print("Section IDs tensor shape:", section_ids_tensor.shape)
    print("Subsection IDs tensor shape:", subsection_ids_tensor.shape)
    print("NER mask tensor shape:", ner_mask_tensor.shape)
    
    soap_note = ("Subjective: Patient reports chest pain and shortness of breath. Denies fever.\n"
                 "Objective: Vital signs stable. Lungs clear.\n"
                 "Assessment: Likely angina.\n"
                 "Plan: Obtain EKG, prescribe nitroglycerin, follow-up in one week.")
    target_encoding = tokenizer(soap_note, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
    decoder_input_ids = target_encoding["input_ids"]
    decoder_attention_mask = target_encoding["attention_mask"]
    labels = decoder_input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    print("Target input shape:", decoder_input_ids.shape)
    
    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    model.load_pretrained_weights("facebook/bart-base")
    model.tokenizer = tokenizer
    # Attach the accumulated input entity set and NER model to the custom model
    model.input_entity_set = input_entity_set
    model.ner_model = ner_model

    print("\n=== Running Forward Pass Through the Custom BART Model ===")
    outputs = model(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        ner_mask=ner_mask_tensor,
        section_ids=section_ids_tensor,
        subsection_ids=subsection_ids_tensor
    )
    print("Model outputs:", outputs)
    if "loss" in outputs:
        print("Loss:", outputs["loss"].item())'''


'''import os
import sys
import logging
import pandas as pd
import torch
from transformers import BartConfig, AutoTokenizer
from .custom_bart_model import CustomBartForConditionalGeneration
from ner_module.ner import load_model, perform_ner_on_tokens

logging.basicConfig(level=logging.INFO)

# -------------------------------
# CSV Processing Functions
# -------------------------------
def simple_tokenizer(text):
    """Simple whitespace tokenizer as a fallback."""
    return text.strip().split()

def process_row(subsection, utterance, ner_model, tokenizer):
    """Process one CSV row: tokenize utterance using the pretrained tokenizer,
       get the NER mask, and also extract the named entity set.
    """
    if utterance.strip().lower() == "nothing reported":
        utterance = ""
    # Tokenize using the pretrained tokenizer
    tokens = tokenizer.tokenize(utterance)
    tokens = [token.lstrip("Ġ") for token in tokens]
    tokens = [token if token.strip() != "" else "[UNK]" for token in tokens]

    if not tokens:
        return [], [], set()
    try:
        ner_flags, entity_set = perform_ner_on_tokens(ner_model, tokens, return_entities=True)
        # Convert any non-zero flag (or non-"O") to 1
        ner_flags = [1 if ((isinstance(f, str) and f != "O") or (isinstance(f, int) and f != 0)) else 0 for f in ner_flags]
    except Exception as e:
        logging.error(f"Error processing NER for subsection {subsection}: {e}")
        tokens, ner_flags, entity_set = [], [], set()
    
    print(f"\n--- Processing Row ---")
    print(f"Subsection: {subsection}")
    print(f"Utterance: {utterance}")
    print(f"Tokens: {tokens}")
    print(f"NER Mask: {ner_flags}")
    print(f"Entity set: {entity_set}")
    return tokens, ner_flags, entity_set

def process_csv(csv_path, ner_model, tokenizer, section_mapping, subsection_mapping):
    """Read CSV and process each row, accumulating tokens, ID lists, and entity sets."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        sys.exit(1)
    all_tokens = []
    all_ner_flags = []
    all_section_ids = []
    all_subsection_ids = []
    input_entity_set = set()  # Accumulate union of entities
    structured_input = ""
    for idx, row in df.iterrows():
        subsection = row['Subsection']
        utterance = row['Grouped_Utterances']
        tokens, ner_flags, entity_set = process_row(subsection, utterance, ner_model, tokenizer)
        structured_input += f"[SUBSECTION] {subsection}: {utterance}\n"
        all_tokens.extend(tokens)
        sec_id = section_mapping.get(subsection, 0)
        subsec_id = subsection_mapping.get(subsection, 0)
        all_section_ids.extend([sec_id] * len(tokens))
        all_subsection_ids.extend([subsec_id] * len(tokens))
        all_ner_flags.extend(ner_flags)
        input_entity_set = input_entity_set.union(entity_set)
    print(f"Input entity set : {input_entity_set}")
    print("\n--- Combined Structured Input ---")
    print(structured_input)
    return all_tokens, all_section_ids, all_subsection_ids, all_ner_flags, input_entity_set

def pad_list(lst, target_length, pad_value=0):
    """Utility function to pad or truncate a list to target_length."""
    if len(lst) < target_length:
        return lst + [pad_value] * (target_length - len(lst))
    else:
        return lst[:target_length]

# -------------------------------
# Main Inference and Debug Function
# -------------------------------
def main():
    # Allow CSV file path to be passed as a command-line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.join("data", "364_TRANSCRIPT_utterances_grouping.csv")
    logging.info(f"Using CSV file: {csv_path}")

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
    print("Loaded pretrained BART tokenizer.")

    ner_model = load_model()
    print("Loaded custom NER model.")

    # Process CSV file to obtain tokens and related info.
    tokens_list, section_ids_list, subsection_ids_list, ner_flag_list, input_entity_set = process_csv(
        csv_path, ner_model, tokenizer, section_mapping, subsection_mapping
    )
    
    max_length = 1024
    # Process tokens for a single sample.
    if len(tokens_list) <= max_length:
        token_ids = tokenizer.convert_tokens_to_ids(tokens_list)
        token_ids = pad_list(token_ids, max_length, tokenizer.pad_token_id)
        input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        if input_ids_tensor.dim() == 1:
            input_ids_tensor = input_ids_tensor.unsqueeze(0)
        attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
        section_ids_tensor = torch.tensor(pad_list(section_ids_list, max_length, 0), dtype=torch.long)
        if section_ids_tensor.dim() == 1:
            section_ids_tensor = section_ids_tensor.unsqueeze(0)
        subsection_ids_tensor = torch.tensor(pad_list(subsection_ids_list, max_length, 0), dtype=torch.long)
        if subsection_ids_tensor.dim() == 1:
            subsection_ids_tensor = subsection_ids_tensor.unsqueeze(0)
        ner_mask_tensor = torch.tensor(pad_list(ner_flag_list, max_length, 0), dtype=torch.long)
        if ner_mask_tensor.dim() == 1:
            ner_mask_tensor = ner_mask_tensor.unsqueeze(0)
    else:
        # If tokens exceed max_length, re-tokenize the joined text.
        structured_text = " ".join(tokens_list)
        encoding = tokenizer(structured_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids_tensor = encoding["input_ids"]
        attention_mask_tensor = encoding["attention_mask"]
        section_ids_tensor = torch.tensor(section_ids_list[:max_length], dtype=torch.long).unsqueeze(0)
        subsection_ids_tensor = torch.tensor(subsection_ids_list[:max_length], dtype=torch.long).unsqueeze(0)
        ner_mask_tensor = torch.tensor(ner_flag_list[:max_length], dtype=torch.long).unsqueeze(0)
    
    print("Input IDs tensor shape:", input_ids_tensor.shape)
    print("Attention mask tensor shape:", attention_mask_tensor.shape)
    print("Section IDs tensor shape:", section_ids_tensor.shape)
    print("Subsection IDs tensor shape:", subsection_ids_tensor.shape)
    print("NER mask tensor shape:", ner_mask_tensor.shape)

    # Set device and move input tensors to same device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids_tensor = input_ids_tensor.to(device)
    attention_mask_tensor = attention_mask_tensor.to(device)
    section_ids_tensor = section_ids_tensor.to(device)
    subsection_ids_tensor = subsection_ids_tensor.to(device)
    ner_mask_tensor = ner_mask_tensor.to(device)

    # Load model and checkpoint.
    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    checkpoint_path = os.path.join("checkpoints", "model_epoch3.pt")
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
    model.load_pretrained_weights("facebook/bart-base")
    model.tokenizer = tokenizer
    model.input_entity_set = input_entity_set
    model.ner_model = ner_model
    model.to(device)
    logging.info("Loaded model from checkpoint and moved to device.")

    # -------------------------------
    # Debug: Run one decoding step to print next-token logits.
    # -------------------------------
    model.eval()
    bos_token_id = config.bos_token_id or config.decoder_start_token_id
    current_seq = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    dec_mask = (current_seq != config.pad_token_id).long()
    enc_attn_mask = (1.0 - attention_mask_tensor[:, None, None, :].float()) * -1e9 if attention_mask_tensor is not None else None

    encoder_output = model.encoder(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor,
        section_ids=section_ids_tensor,
        subsection_ids=subsection_ids_tensor,
        ner_mask=ner_mask_tensor
    )
    decoder_outputs = model.decoder(
        decoder_input_ids=current_seq,
        decoder_attention_mask=dec_mask,
        encoder_hidden_states=encoder_output,
        encoder_attention_mask=enc_attn_mask
    )
    hidden_states = decoder_outputs  # [1, seq_len, d_model]
    logits = model.lm_head(hidden_states)  # [1, seq_len, vocab_size]
    next_token_logits = logits[:, -1, :]  # [1, vocab_size]
    print("DEBUG: Next-token logits (first 10 values):", next_token_logits[0][:25].tolist())

    # -------------------------------
    # Full Inference: Generate text using generate_text
    # -------------------------------
    generated_text = model.generate_text(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor,
        section_ids=section_ids_tensor,
        subsection_ids=subsection_ids_tensor,
        ner_mask=ner_mask_tensor,
        max_length=256,
        beam_size=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        length_penalty=1.0,
        early_stopping=True,
        min_length=5
    )
    print("\n=== Generated Text ===")
    print(generated_text)

if __name__ == "__main__":
    main()
'''
import os
import json
import time
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartConfig, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.models.bart.modeling_bart import shift_tokens_right
from .custom_bart_model import CustomBartForConditionalGeneration
from .train_utils import compute_ner_penalty
from ner_module.ner import load_model, perform_ner_on_tokens
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO)

# --- Helper Functions for Filename Adjustment --`  
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
    return lst + [pad_value] * (target_length - len(lst)) if len(lst) < target_length else lst[:target_length]

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

def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_ner_penalty = 0.0  # To accumulate NER penalty losses

    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        section_ids = batch["section_ids"].to(device)
        subsection_ids = batch["subsection_ids"].to(device)
        ner_mask = batch["ner_mask"].to(device)
        labels = batch["target_ids"].to(device)

        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.bos_token_id 
        )

        decoder_attention_mask = (decoder_input_ids != model.tokenizer.pad_token_id).long()

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            ner_mask=ner_mask,
            section_ids=section_ids,
            subsection_ids=subsection_ids
        )
        loss = outputs["loss"]
        ner_penalty = compute_ner_penalty(input_ids, labels, model)
        
        # Combine the main loss and NER penalty
        total_loss += loss.item()
        total_ner_penalty += ner_penalty.item()  # Accumulate NER penalty

        loss += ner_penalty  # Add the NER penalty to the main loss for backward pass
        loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logging.info(f"Epoch batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}, Elapsed: {elapsed:.2f}s")
            start_time = time.time()

    avg_loss = total_loss / len(dataloader)
    avg_ner_penalty = total_ner_penalty / len(dataloader)
    logging.info(f"Average NER Penalty for the epoch: {avg_ner_penalty:.4f}")
    
    return avg_loss, avg_ner_penalty

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load session splits
    splits_path = os.path.join("data", "session_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    train_files = splits["train"]

    train_input_dir = os.path.join("data", "train_intermediate_files")
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

    train_dataset = SessionDataset(train_files, tokenizer, ner_model, section_mapping, subsection_mapping,
                                   input_dir=train_input_dir, target_dir=target_dir, max_length=1024, max_target_length=512)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    model.load_pretrained_weights("facebook/bart-base")
    model.tokenizer = tokenizer
    model.ner_model = ner_model
    model.to(device)
    logging.info("Initialized model and moved to device.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * 17  # 17 epochs
    warmup_steps = int(0.05 * total_steps)  # 5% of total steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    num_epochs = 17
    gradient_accumulation_steps = 4  # Example: Simulate a batch size of 4 with accumulation

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        avg_loss, avg_ner_penalty = train_epoch(model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps)
        logging.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}, Average NER Penalty: {avg_ner_penalty:.4f}. Time taken: {time.time() - epoch_start:.2f}s")
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    logging.info("Training completed.")


if __name__ == "__main__":
    main()
