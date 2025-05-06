'''import os
import torch
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import logging

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Should print "Using device: cuda" if available

# Setup logging
logging.basicConfig(filename="summarization_training.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Load session split info (we need only training sessions for summarization training)
with open("data/session_splits.json", "r") as f:
    splits = json.load(f)
train_files = splits["train"]
print("check1")

# Folder paths for intermediate files and summary files
train_intermediate_folder = "data/train_intermediate_files"
soap_notes_folder = "data/Soap_notes"

# Define valid subsections (same as before)
subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]
print("check2")

# Build summarization data from training sessions only
summarization_data = []
for file_path in train_files:
    base_name = os.path.basename(file_path).replace(".csv", "")
    intermediate_file = os.path.join(train_intermediate_folder, f"{base_name}_utterances_grouping.csv")
    summary_file = os.path.join(soap_notes_folder, f"{base_name}_output.csv")
    
    interm_df = pd.read_csv(intermediate_file)
    summary_df = pd.read_csv(summary_file)
    summary_row = summary_df.iloc[0]
    
    for subsec in subsection_list:
        grouped_text_series = interm_df.loc[interm_df["Subsection"] == subsec, "Grouped_Utterances"]
        if not grouped_text_series.empty:
            input_text = grouped_text_series.values[0]
        else:
            input_text = "Nothing reported"
        target_summary = summary_row.get(subsec, "Nothing reported")
        summarization_data.append([subsec, input_text, target_summary])

print("check3")
# Convert to DataFrame
summarization_df = pd.DataFrame(summarization_data, columns=["Subsection", "Input_Text", "Target_Summary"])
print("check4")

# Fix: Use DataFrame slicing instead of `random_split`
total = len(summarization_df)
train_count = int(0.8 * total)
val_count = int(0.1 * total)
test_count = total - train_count - val_count

# Perform DataFrame-based splitting
train_df = summarization_df.iloc[:train_count].reset_index(drop=True)
val_df = summarization_df.iloc[train_count:train_count+val_count].reset_index(drop=True)
test_df = summarization_df.iloc[train_count+val_count:].reset_index(drop=True)
print("check5")

# Load Bart tokenizer and model
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
print("check6")

def process_text(text, section):
    """Format and tokenize the input text for Bart with section conditioning."""
    input_str = f"section: {section} text: {text}"
    return bart_tokenizer(input_str, return_tensors="pt", padding=True, truncation=True, max_length=512)
print("check7")

# Custom Dataset for summarization
class SummarizationDataset(Dataset):
    def __init__(self, df):
        self.data = df.reset_index(drop=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = process_text(row["Input_Text"], row["Subsection"])
        # Convert Target_Summary to string & handle NaNs
        target_summary = str(row["Target_Summary"]) if pd.notna(row["Target_Summary"]) else "Nothing reported"
        target_encoding = bart_tokenizer(target_summary, return_tensors="pt", padding=True, truncation=True, max_length=128)
        return {
            "input_ids": encoding['input_ids'].squeeze(),
            "attention_mask": encoding['attention_mask'].squeeze(),
            "labels": target_encoding['input_ids'].squeeze(),
            "subsection": row["Subsection"]
        }
print("check8")

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad all sequences to the same length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=bart_tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)  # Pad attention mask with 0
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Ignore padding in loss computation

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }

# Convert DataFrame splits into PyTorch Datasets
train_dataset = SummarizationDataset(train_df)
val_dataset = SummarizationDataset(val_df)
test_dataset = SummarizationDataset(test_df)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False, collate_fn=collate_fn)
print("check9")

def train_bart_model():
    print("check10")
    """Train the Bart summarization model on training sessions."""
    bart_model.train()
    optimizer = AdamW(bart_model.parameters(), lr=1e-4)
    
    for epoch in range(3):
        print("check11")
        print(f"Epoch {epoch+1} started...")
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            if i % 10 == 0:  # Print every 10 batches
                print(f"Processing batch {i}/{len(train_loader)}...")
            optimizer.zero_grad()
            outputs = bart_model(input_ids=batch['input_ids'].to(device),
                                 attention_mask=batch['attention_mask'].to(device),
                                 labels=batch['labels'].to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    torch.save(bart_model.state_dict(), "models/bart_summarizer.pth")
    print("Bart Summarization Model Saved!")

if __name__ == "__main__":
    train_bart_model()
'''

import os
import torch
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from torch.nn.utils.rnn import pad_sequence
import logging

# Logging setup
logging.basicConfig(filename="summarization_training.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Paths and setup
with open("data/session_splits.json", "r") as f:
    splits = json.load(f)
train_files = splits["train"]

train_input_dir = "data/train_intermediate_files"
target_dir = "data/Soap_notes"

subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256

def build_training_pair(input_path, target_path):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)

    input_lines = []
    for subsec in subsection_list:
        uttr = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        value = uttr.values[0] if not uttr.empty else "Nothing reported"
        input_lines.append(f"{subsec}: {value}")
    input_text = "\n".join(input_lines)

    target_text = "\n".join([
        f"{col}: {str(target_df[col].iloc[0])}" if pd.notna(target_df[col].iloc[0]) else f"{col}: Nothing reported"
        for col in subsection_list
    ])
    return input_text, target_text

class SessionLevelDataset(Dataset):
    def __init__(self, file_list):
        self.pairs = []
        for file_path in file_list:
            base = os.path.basename(file_path).replace(".csv", "")
            input_file = os.path.join(train_input_dir, f"{base}_utterances_grouping.csv")
            target_file = os.path.join(target_dir, f"{base}_output.csv")
            try:
                src, tgt = build_training_pair(input_file, target_file)
                self.pairs.append((src, tgt))
            except Exception as e:
                logging.warning(f"Skipping file {file_path} due to error: {e}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return {
            "input_text": src,
            "target_text": tgt
        }

def collate_fn(batch):
    input_enc = tokenizer([item["input_text"] for item in batch],
                          return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
    target_enc = tokenizer([item["target_text"] for item in batch],
                           return_tensors="pt", padding=True, truncation=True, max_length=MAX_TARGET_LENGTH).to(device)
    labels = target_enc["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": labels
    }

def train():
    train_dataset = SessionLevelDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    model.train()
    for epoch in range(3):
        print(f"Epoch {epoch+1} started...")
        total_loss = 0
        for i, batch in enumerate(train_loader):
            if i % 10 == 0:  # Print every 10 batches
                print(f"Processing batch {i}/{len(train_loader)}...")
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bart_summarizer_session_level.pth")
    logging.info("Session-level BART model saved.")

if __name__ == "__main__":
    train()
