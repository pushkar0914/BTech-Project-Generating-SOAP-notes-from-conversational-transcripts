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
        # Remove Ġ prefix for consistency (for NER reasons)
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
        #print(f"tokens:{tokens}")    
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
    #print(f"pad token id :{tokenizer.pad_token_id}")
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
    # Build final summary similar to test side (one line per column)
    final_summary = ""
    for col in df.columns:
        value = str(df[col].iloc[0])
        final_summary += f"{col}: {value}\n"
    # Strip off any 'Ġ' markers for consistency
    #print(f"final summary :{final_summary}")
    final_summary = final_summary.replace("Ġ", "")
    #print(f"final summary :{final_summary}")
    target_encoding = tokenizer(final_summary, return_tensors="pt", max_length=max_target_length, truncation=True, padding="max_length")
    target_ids = target_encoding["input_ids"].squeeze(0)
    #print(f"target_ids shape: {target_ids.shape}")


    # drop the first <s>
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

    #print("First token ID:", target_ids[0].item())
    #print("BOS token ID:", tokenizer.bos_token_id)
    #print("2 token ID:", target_ids[1].item())
    #print("3 token ID:", target_ids[2].item())
    #print("4 token ID:", target_ids[3].item())
    #print("Last token ID:", target_ids[-1].item())
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

# --- Validation Evaluation Function ---
def evaluate_model(model, dataloader, device, max_length=256):
    model.eval()
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
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
            scores = scorer.score(target_text, generated_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    return avg_rouge1, avg_rouge2, avg_rougeL

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        section_ids = batch["section_ids"].to(device)
        subsection_ids = batch["subsection_ids"].to(device)
        ner_mask = batch["ner_mask"].to(device)
        labels = batch["target_ids"].to(device)

        # Shift labels to create correct decoder inputs
        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.bos_token_id 
        )
        #print(f"decoder input ids after right shift : {decoder_input_ids} ")
        #print(f"decoder_input_ids shape : {decoder_input_ids.shape}")

        # Build decoder attention mask
        decoder_attention_mask = (decoder_input_ids != model.tokenizer.pad_token_id).long()

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,  # now correctly shifted
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            ner_mask=ner_mask,
            section_ids=section_ids,
            subsection_ids=subsection_ids
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logging.info(f"Epoch batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}, Elapsed: {elapsed:.2f}s")
            start_time = time.time()

    return total_loss / len(dataloader)


# --- Main Training Function with Periodic Validation Evaluation ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load session splits
    splits_path = os.path.join("data", "session_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    train_files = splits["train"]
    #val_files = splits["val"]

    # Define directories
    train_input_dir = os.path.join("data", "train_intermediate_files")
    # New: Use validation_intermediate_files directory for validation inputs.
    #val_input_dir = os.path.join("data", "validation_intermediate_files")
    target_dir = os.path.join("data", "Soap_notes")

    # Define mappings
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

    # Create training dataset and loader.
    train_dataset = SessionDataset(train_files, tokenizer, ner_model, section_mapping, subsection_mapping,
                                   input_dir=train_input_dir, target_dir=target_dir, max_length=1024, max_target_length=512)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Create validation dataset and loader.
    #val_dataset = SessionDataset(val_files, tokenizer, ner_model, section_mapping, subsection_mapping,
                                 #input_dir=val_input_dir, target_dir=target_dir, max_length=1024, max_target_length=256)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = CustomBartForConditionalGeneration(config)
    model.load_pretrained_weights("facebook/bart-base")
    model.tokenizer = tokenizer
    model.ner_model = ner_model
    model.to(device)
    logging.info("Initialized model and moved to device.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * 10  # 10 epochs (you can update this if running more epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    num_epochs = 10
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logging.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}. Time taken: {time.time() - epoch_start:.2f}s")
        # Save checkpoint after each epoch
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

        # --- Periodic Validation Evaluation ---
        #val_rouge1, val_rouge2, val_rougeL = evaluate_model(model, val_loader, device)
        #logging.info(f"Validation ROUGE scores at epoch {epoch+1}: ROUGE-1: {val_rouge1:.4f}, ROUGE-2: {val_rouge2:.4f}, ROUGE-L: {val_rougeL:.4f}")

    logging.info("Training completed.")

if __name__ == "__main__":
    main()'''



'''import os
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
        # Remove Ġ prefix for consistency (for NER reasons)
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

def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
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
    total_steps = len(train_loader) * 15  # 15 epochs
    warmup_steps = int(0.05 * total_steps)  # 5% of total steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    num_epochs = 15
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
    learning schd and grad clipping 
'''
#below best till now 
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

# --- Helper Functions for Filename Adjustment --  
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
        # Remove Ġ prefix for consistency (for NER reasons)
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

def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
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
    total_steps = len(train_loader) * 5  # 10 epochs
    warmup_steps = int(0.05 * total_steps)  # 5% of total steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    num_epochs = 5
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