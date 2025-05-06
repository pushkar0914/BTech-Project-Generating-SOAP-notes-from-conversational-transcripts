'''import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer

# 1) Choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_intermediate_folder = "data/test_intermediate_files"   # predicted intermediate files for test sessions
soap_notes_folder = "data/Soap_notes"  # original summary files
# For storing generated summaries for test sessions (optional)
generated_output_folder = "data/test_generated_soap_notes"
os.makedirs(generated_output_folder, exist_ok=True)

# Load session split info to get test session file names
with open("data/session_splits.json", "r") as f:
    splits = json.load(f)
test_files = splits["test"]

# Define valid subsections
subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

# Build summarization test data from test sessions
summarization_data = []
for file_path in test_files:
    base_name = os.path.basename(file_path).replace(".csv", "")
    # Use predicted intermediate file from test_intermediate_folder
    intermediate_file = os.path.join(test_intermediate_folder, f"{base_name}_utterances_grouping.csv")
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
        summarization_data.append([subsec, input_text, target_summary, base_name])

test_summarization_df = pd.DataFrame(summarization_data, columns=["Subsection", "Input_Text", "Target_Summary", "SessionID"])

# Create a test dataset for summarization evaluation
class SummarizationDataset(Dataset):
    def __init__(self, df):
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_str = f"section: {row['Subsection']} text: {row['Input_Text']}"
        return {
            "input_str": input_str,
            "target_summary": row["Target_Summary"],
            "SessionID": row["SessionID"],
            "Subsection": row["Subsection"]
        }

test_dataset = SummarizationDataset(test_summarization_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 2) Load T5 model and tokenizer (fine-tuned model) onto GPU
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load the fine-tuned weights
t5_model.load_state_dict(
    torch.load("models/t5_summarizer.pth", map_location=device)
)

# Move model to GPU (if available)
t5_model.to(device)
t5_model.eval()

def generate_summary(input_str):
    # Tokenize inputs and move them to GPU
    encoding = t5_tokenizer(input_str, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    summary_ids = t5_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_rouge1, all_rouge2, all_rougeL = [], [], []

# Evaluate over the test set
results = []
with torch.no_grad():
    for batch in test_loader:
        input_str = batch["input_str"][0]
        target_summary = batch["target_summary"][0]

        # -- Fix: If target_summary or input_str is a list or tensor, convert to string
        if isinstance(target_summary, list):
            target_summary = " ".join(map(str, target_summary))
        elif isinstance(target_summary, torch.Tensor):
            # If it's a single-element tensor
            if target_summary.dim() == 0:
                target_summary = str(target_summary.item())
            else:
                # If multi-element, handle appropriately
                target_summary = " ".join(map(str, target_summary.tolist()))

        if isinstance(input_str, list):
            input_str = " ".join(map(str, input_str))
        elif isinstance(input_str, torch.Tensor):
            if input_str.dim() == 0:
                input_str = str(input_str.item())
            else:
                input_str = " ".join(map(str, input_str.tolist()))

        # Generate the summary
        generated = generate_summary(input_str)

        # Also check if generated is a tensor or list
        if isinstance(generated, list):
            generated = " ".join(map(str, generated))
        elif isinstance(generated, torch.Tensor):
            if generated.dim() == 0:
                generated = str(generated.item())
            else:
                generated = " ".join(map(str, generated.tolist()))

        scores = scorer.score(target_summary, generated)

        all_rouge1.append(scores['rouge1'].fmeasure)
        all_rouge2.append(scores['rouge2'].fmeasure)
        all_rougeL.append(scores['rougeL'].fmeasure)

        results.append({
            "SessionID": batch["SessionID"][0],
            "Subsection": batch["Subsection"][0],
            "Input": input_str,
            "Generated": generated,
            "Target": target_summary,
            "ROUGE": scores
        })

        print(f"Session {batch['SessionID'][0]} - {batch['Subsection'][0]}")
        print("Generated:", generated)
        print("Target:", target_summary)
        print("Scores:", scores)
        print("-----")

avg_rouge1 = sum(all_rouge1) / len(all_rouge1) if all_rouge1 else 0
avg_rouge2 = sum(all_rouge2) / len(all_rouge2) if all_rouge2 else 0
avg_rougeL = sum(all_rougeL) / len(all_rougeL) if all_rougeL else 0

print("Average ROUGE-1:", avg_rouge1)
print("Average ROUGE-2:", avg_rouge2)
print("Average ROUGE-L:", avg_rougeL)

# Optionally, save the detailed results to a CSV file for further analysis
results_df = pd.DataFrame(results)
results_df.to_csv("data/test_generated_soap_notes/evaluation_results.csv", index=False)
print("Evaluation results saved to data/test_generated_soap_notes/evaluation_results.csv!")
'''

'''import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder paths
test_input_dir = "data/test_intermediate_files"
target_dir = "data/Soap_notes"

# Load session split info
with open("data/session_splits.json", "r") as f:
    test_files = json.load(f)["test"]

# Subsection list (same order used during training)
subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

# Utility to merge subsections from one session file
def build_test_pair(input_path, target_path):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)

    merged_input = []
    for subsec in subsection_list:
        uttr = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        value = uttr.values[0] if not uttr.empty else "Nothing reported"
        merged_input.append(f"{subsec}: {value}")
    input_text = "\n".join(merged_input)

    target_text = "\n".join([
        f"{col}: {str(target_df[col].iloc[0])}" if pd.notna(target_df[col].iloc[0]) else f"{col}: Nothing reported"
        for col in subsection_list
    ])
    return input_text, target_text

# Dataset for full-session test evaluation
class SessionLevelTestDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        for file_path in file_list:
            base = os.path.basename(file_path).replace(".csv", "")
            input_file = os.path.join(test_input_dir, f"{base}_utterances_grouping.csv")
            target_file = os.path.join(target_dir, f"{base}_output.csv")
            try:
                input_text, target_text = build_test_pair(input_file, target_file)
                self.data.append((input_text, target_text, base))
            except Exception as e:
                print(f"Skipping {base} due to error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_text": self.data[idx][0],
            "target_text": self.data[idx][1],
            "session_id": self.data[idx][2]
        }

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.load_state_dict(torch.load("models/t5_summarizer_session_level.pth", map_location=device))
model.to(device)
model.eval()

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load test dataset
test_dataset = SessionLevelTestDataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate
all_rouge1, all_rouge2, all_rougeL = [], [], []
results = []

with torch.no_grad():
    for batch in test_loader:
        input_text = batch["input_text"][0]
        target_text = batch["target_text"][0]
        session_id = batch["session_id"][0]

        enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=256  # Match training target length
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        scores = scorer.score(target_text, generated_text)
        all_rouge1.append(scores["rouge1"].fmeasure)
        all_rouge2.append(scores["rouge2"].fmeasure)
        all_rougeL.append(scores["rougeL"].fmeasure)

        results.append({
            "SessionID": session_id,
            "Generated": generated_text,
            "Target": target_text,
            "ROUGE": scores
        })

# Compute average scores
avg_rouge1 = sum(all_rouge1) / len(all_rouge1)
avg_rouge2 = sum(all_rouge2) / len(all_rouge2)
avg_rougeL = sum(all_rougeL) / len(all_rougeL)

print("Average ROUGE-1:", avg_rouge1)
print("Average ROUGE-2:", avg_rouge2)
print("Average ROUGE-L:", avg_rougeL)

# Optionally save detailed results
pd.DataFrame(results).to_csv("data/test_generated_soap_notes/evaluation_session_level.csv", index=False)
'''
'''import os
import json
import torch
import nltk
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer

# 👇 NEW
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder paths
test_input_dir = "data/test_intermediate_files"
target_dir = "data/Soap_notes"

# Load session split info
with open("data/session_splits.json", "r") as f:
    test_files = json.load(f)["test"]

# Subsection list (same order used during training)
subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

# Utility to merge subsections from one session file
def build_test_pair(input_path, target_path):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)

    merged_input = []
    for subsec in subsection_list:
        uttr = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        value = uttr.values[0] if not uttr.empty else "Nothing reported"
        merged_input.append(f"{subsec}: {value}")
    input_text = "\n".join(merged_input)

    target_text = "\n".join([
        f"{col}: {str(target_df[col].iloc[0])}" if pd.notna(target_df[col].iloc[0]) else f"{col}: Nothing reported"
        for col in subsection_list
    ])
    return input_text, target_text

# Dataset for full-session test evaluation
class SessionLevelTestDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        for file_path in file_list:
            base = os.path.basename(file_path).replace(".csv", "")
            input_file = os.path.join(test_input_dir, f"{base}_utterances_grouping.csv")
            target_file = os.path.join(target_dir, f"{base}_output.csv")
            try:
                input_text, target_text = build_test_pair(input_file, target_file)
                self.data.append((input_text, target_text, base))
            except Exception as e:
                print(f"Skipping {base} due to error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_text": self.data[idx][0],
            "target_text": self.data[idx][1],
            "session_id": self.data[idx][2]
        }

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.load_state_dict(torch.load("models/t5_summarizer_session_level.pth", map_location=device))
model.to(device)
model.eval()

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load test dataset
test_dataset = SessionLevelTestDataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- Score Storage ---
all_rouge1, all_rouge2, all_rougeL = [], [], []
all_bleu, all_meteor = [], []
results = []

# Evaluation
with torch.no_grad():
    for batch in test_loader:
        input_text = batch["input_text"][0]
        target_text = batch["target_text"][0]
        session_id = batch["session_id"][0]

        enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=256  # Match training target length
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ROUGE
        scores = scorer.score(target_text, generated_text)
        all_rouge1.append(scores["rouge1"].fmeasure)
        all_rouge2.append(scores["rouge2"].fmeasure)
        all_rougeL.append(scores["rougeL"].fmeasure)

        # --- BLEU + METEOR ---
        reference = target_text.split()
        hypothesis = generated_text.split()
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
        meteor = meteor_score([target_text.split()], generated_text.split())
        all_bleu.append(bleu)
        all_meteor.append(meteor)

        results.append({
            "SessionID": session_id,
            "Generated": generated_text,
            "Target": target_text,
            "ROUGE": scores,
            "BLEU": bleu,
            "METEOR": meteor
        })

# --- Average Scores ---
avg_rouge1 = sum(all_rouge1) / len(all_rouge1)
avg_rouge2 = sum(all_rouge2) / len(all_rouge2)
avg_rougeL = sum(all_rougeL) / len(all_rougeL)
avg_bleu = sum(all_bleu) / len(all_bleu)
avg_meteor = sum(all_meteor) / len(all_meteor)

print("Average ROUGE-1:", avg_rouge1)
print("Average ROUGE-2:", avg_rouge2)
print("Average ROUGE-L:", avg_rougeL)
print("Average BLEU:", avg_bleu)
print("Average METEOR:", avg_meteor)

# Save detailed results
pd.DataFrame(results).to_csv("data/test_generated_soap_notes/evaluation_session_level_with_bleu_meteor.csv", index=False)
'''
import os
import json
import torch
import nltk
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score  # 👈 NEW

# NLTK downloads
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
test_input_dir = "data/test_intermediate_files"
target_dir = "data/Soap_notes"

with open("data/session_splits.json", "r") as f:
    test_files = json.load(f)["test"]

subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

def build_test_pair(input_path, target_path):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)

    merged_input = []
    for subsec in subsection_list:
        uttr = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        value = uttr.values[0] if not uttr.empty else "Nothing reported"
        merged_input.append(f"{subsec}: {value}")
    input_text = "\n".join(merged_input)

    target_text = "\n".join([
        f"{col}: {str(target_df[col].iloc[0])}" if pd.notna(target_df[col].iloc[0]) else f"{col}: Nothing reported"
        for col in subsection_list
    ])
    return input_text, target_text

class SessionLevelTestDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        for file_path in file_list:
            base = os.path.basename(file_path).replace(".csv", "")
            input_file = os.path.join(test_input_dir, f"{base}_utterances_grouping.csv")
            target_file = os.path.join(target_dir, f"{base}_output.csv")
            try:
                input_text, target_text = build_test_pair(input_file, target_file)
                self.data.append((input_text, target_text, base))
            except Exception as e:
                print(f"Skipping {base} due to error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_text": self.data[idx][0],
            "target_text": self.data[idx][1],
            "session_id": self.data[idx][2]
        }

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.load_state_dict(torch.load("models/t5_summarizer_session_level.pth", map_location=device))
model.to(device)
model.eval()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
test_dataset = SessionLevelTestDataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

all_rouge1, all_rouge2, all_rougeL = [], [], []
all_bleu, all_meteor, all_bertscore = [], [], []
results = []

with torch.no_grad():
    for batch in test_loader:
        input_text = batch["input_text"][0]
        target_text = batch["target_text"][0]
        session_id = batch["session_id"][0]

        enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=256
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ROUGE
        scores = scorer.score(target_text, generated_text)
        all_rouge1.append(scores["rouge1"].fmeasure)
        all_rouge2.append(scores["rouge2"].fmeasure)
        all_rougeL.append(scores["rougeL"].fmeasure)

        # BLEU & METEOR
        reference = target_text.split()
        hypothesis = generated_text.split()
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
        meteor = meteor_score([reference], hypothesis)
        all_bleu.append(bleu)
        all_meteor.append(meteor)

        # BERTScore (using RoBERTa-large from local cache)
        P, R, F1 = bert_score([generated_text], [target_text], model_type="roberta-large", lang="en", device=device, rescale_with_baseline=True)
        all_bertscore.append(F1[0].item())

        results.append({
            "SessionID": session_id,
            "Generated": generated_text,
            "Target": target_text,
            "ROUGE": scores,
            "BLEU": bleu,
            "METEOR": meteor,
            "BERTScore-F1": F1[0].item()
        })

# --- Print averages
print("Average ROUGE-1:", sum(all_rouge1) / len(all_rouge1))
print("Average ROUGE-2:", sum(all_rouge2) / len(all_rouge2))
print("Average ROUGE-L:", sum(all_rougeL) / len(all_rougeL))
print("Average BLEU:   ", sum(all_bleu) / len(all_bleu))
print("Average METEOR: ", sum(all_meteor) / len(all_meteor))
print("Average BERTScore-F1:", sum(all_bertscore) / len(all_bertscore))

# Save results
pd.DataFrame(results).to_csv("data/test_generated_soap_notes/evaluation_session_level_full_metrics.csv", index=False)
