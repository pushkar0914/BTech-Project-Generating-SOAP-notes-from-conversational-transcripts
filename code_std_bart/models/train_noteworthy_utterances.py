import os
import sys
import json
import torch
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from noteworthy_extractor import NoteworthyExtractor
import logging
import csv
# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Should print "Using device: cuda"

# Setup logging
logging.basicConfig(filename="noteworthy_training.log", level=logging.INFO,
                    format='%(asctime)s - %(message)s', force=True)

# Define folder paths
classified_folder = "data/classified_utterances"   # contains session CSVs (e.g., 300_TRANSCRIPT.csv, etc.)
train_intermediate_folder = "data/train_intermediate_files"   # for ground-truth grouping (training sessions)
test_intermediate_folder = "data/test_intermediate_files"     # for predicted grouping (test sessions)
os.makedirs(train_intermediate_folder, exist_ok=True)
os.makedirs(test_intermediate_folder, exist_ok=True)

# List all session files from the classified folder
all_session_files = [os.path.join(classified_folder, f) for f in os.listdir(classified_folder) if f.endswith(".csv")]
all_session_files.sort()  # sort for reproducibility

# Decide if we force a new split
force_split = "--force-split" in sys.argv

split_file = "data/session_splits.json"
if os.path.exists(split_file) and not force_split:
    with open(split_file, "r") as f:
        session_splits = json.load(f)
    print("Session splits loaded from file.")
else:
    random.seed(42)
    random.shuffle(all_session_files)
    num_sessions = len(all_session_files)
    train_count = int(0.8 * num_sessions)
    val_count = int(0.1 * num_sessions)
    test_count = num_sessions - train_count - val_count

    train_files = all_session_files[:train_count]
    val_files = all_session_files[train_count:train_count+val_count]
    test_files = all_session_files[train_count+val_count:]

    session_splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    with open(split_file, "w") as f:
        json.dump(session_splits, f, indent=4)
    print("Session splits saved.")

# Use loaded splits
train_files = session_splits["train"]
val_files = session_splits["val"]
test_files = session_splits["test"]

# Define subsection mapping (15 valid subsections)
subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]
subsection_to_index = {subsec: i for i, subsec in enumerate(subsection_list)}
print("check1")

def label_to_multihot(label_str):
    """Convert a comma-separated label string to a multi-hot vector of length 15."""
    labels = [l.strip() for l in label_str.split(",")]
    multihot = [0] * len(subsection_list)
    for l in labels:
        if l in subsection_to_index:
            multihot[subsection_to_index[l]] = 1
    return torch.tensor(multihot, dtype=torch.float)
print("check2")

# For training the extractor, combine all training session files
train_df_list = []
for file in train_files:
    temp_df = pd.read_csv(file, encoding="utf-8-sig")
    # Filter out rows with "Insignificant" labels
    temp_df = temp_df[temp_df['Classified_Subsection'] != "Insignificant"]
    train_df_list.append(temp_df)
train_df = pd.concat(train_df_list, ignore_index=True)
print("check3")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def process_text(text):
    """Tokenize text with fixed max length."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        text = "Nothing reported"
    return tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
print("check4")

# Dataset class for extraction training
class NoteworthyDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['Utterance'].tolist()
        self.labels = dataframe['Classified_Subsection'].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = process_text(self.texts[idx])
        multihot_label = label_to_multihot(self.labels[idx])
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': multihot_label,
            'text': self.texts[idx]
        }

train_dataset = NoteworthyDataset(train_df)
# (For simplicity, we won’t use val/test splits here for extraction training)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print("check5")

# Initialize extraction model, optimizer, and multi-label loss
noteworthy_model = NoteworthyExtractor().to(device)
optimizer = torch.optim.Adam(noteworthy_model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()
print("check6")

def train_noteworthy_model():
    """Train the extraction model on the training sessions."""
    for epoch in range(3):
        print(f"Epoch {epoch+1} started...")
        noteworthy_model.train()
        for i, batch in enumerate(train_loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(train_loader)}...")
            optimizer.zero_grad()
            outputs = noteworthy_model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            # Concatenate labels and move them to device
            target = torch.cat([label.unsqueeze(0) if label.dim() == 1 else label for label in batch['label']], dim=0).to(device)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")
    torch.save(noteworthy_model.state_dict(), "models/noteworthy_extractor.pth")
    print("Extraction model saved.")
print("check7")

def generate_intermediate_files(mode="train"):
    """
    Generate intermediate grouping CSVs for each session.
    For mode "train": use the original (ground-truth) classifications.
    For mode "test": run the trained model to classify utterances.
    The resulting CSV groups utterances by subsection.
    """
    print("check9")
    # Choose file list and output folder based on mode
    if mode == "train":
        files = train_files  # use training session files
        output_folder = train_intermediate_folder
        use_ground_truth = True
    elif mode == "test":
        files = test_files   # use test session files
        output_folder = test_intermediate_folder
        use_ground_truth = False
    else:
        raise ValueError("Mode must be 'train' or 'test'")
    
    # If using model predictions, ensure the model is in eval mode
    if not use_ground_truth:
        noteworthy_model.eval()
    
    for file_path in files:
        session_df = pd.read_csv(file_path, encoding="utf-8-sig")
        session_df = session_df[session_df['Classified_Subsection'] != "Insignificant"]
        session_results = []
        for idx, row in session_df.iterrows():
            text = row['Utterance']
            if use_ground_truth:
                # Use original classifications
                labels = [l.strip() for l in row['Classified_Subsection'].split(",")]
            else:
                # Use model predictions: ensure text is tokenized and moved to device
                encoding = process_text(text)
                outputs = noteworthy_model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
                preds = (torch.sigmoid(outputs) > 0.5).int()
                # Convert predicted indices to labels
                labels = [subsection_list[j] for j, flag in enumerate(preds[0]) if flag.item() == 1]
                if not labels:
                    labels = ["Nothing reported"]
            for label in labels:
                session_results.append([text, label])
        if session_results:
            results_df = pd.DataFrame(session_results, columns=["Utterance", "Subsection"])
            grouped = results_df.groupby("Subsection")["Utterance"].apply(lambda x: " ".join(x)).to_dict()
        else:
            grouped = {}
        for subsec in subsection_list:
            if subsec not in grouped:
                grouped[subsec] = "Nothing reported"
        grouping_data = [[subsec, grouped[subsec]] for subsec in subsection_list]
        grouping_df = pd.DataFrame(grouping_data, columns=["Subsection", "Grouped_Utterances"])
        base_name = os.path.basename(file_path).replace(".csv", "")
        intermediate_file = os.path.join(output_folder, f"{base_name}_utterances_grouping.csv")
        grouping_df.to_csv(intermediate_file, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved {mode} intermediate grouping for {base_name} to {intermediate_file}")

if __name__ == "__main__":
    import sys
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode_arg == "train":
        print("DEBUG: Starting training of Noteworthy Extractor...")
        train_noteworthy_model()
    elif mode_arg == "test":
        # Load the model from file in test mode
        noteworthy_model.load_state_dict(torch.load("models/noteworthy_extractor.pth", map_location=device))
        noteworthy_model.to(device)
        noteworthy_model.eval()
    else:
        raise ValueError("Argument must be 'train' or 'test'")
    
    # Generate intermediate files based on mode
    generate_intermediate_files(mode=mode_arg)
