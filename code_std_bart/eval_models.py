'''import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
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

# 2) Load Bart model and tokenizer (fine-tuned model) onto GPU
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Load the fine-tuned weights
bart_model.load_state_dict(torch.load("models/bart_summarizer.pth", map_location=device))

# Move model to GPU (if available)
bart_model.to(device)
bart_model.eval()

def generate_summary(input_str):
    # Tokenize inputs and move them to GPU
    encoding = bart_tokenizer(input_str, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    summary_ids = bart_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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
            if target_summary.dim() == 0:
                target_summary = str(target_summary.item())
            else:
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
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
test_intermediate_folder = "data/test_intermediate_files"
soap_notes_folder = "data/Soap_notes"
generated_output_folder = "data/test_generated_soap_notes"
os.makedirs(generated_output_folder, exist_ok=True)

# Load test session IDs
with open("data/session_splits.json", "r") as f:
    test_files = json.load(f)["test"]

subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

# Build input-output session pairs
session_data = []
for file_path in test_files:
    base = os.path.basename(file_path).replace(".csv", "")
    input_file = os.path.join(test_intermediate_folder, f"{base}_utterances_grouping.csv")
    target_file = os.path.join(soap_notes_folder, f"{base}_output.csv")

    try:
        input_df = pd.read_csv(input_file)
        target_df = pd.read_csv(target_file)
    except Exception as e:
        print(f"Skipping session {base} due to error: {e}")
        continue

    # Build full input and output
    input_lines = []
    target_lines = []
    for subsec in subsection_list:
        val = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        input_text = val.values[0] if not val.empty else "Nothing reported"
        input_lines.append(f"{subsec}: {input_text}")

        tval = target_df[subsec].iloc[0] if subsec in target_df.columns and pd.notna(target_df[subsec].iloc[0]) else "Nothing reported"
        target_lines.append(f"{subsec}: {tval}")

    session_data.append((base, "\n".join(input_lines), "\n".join(target_lines)))

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.load_state_dict(torch.load("models/bart_summarizer_session_level.pth", map_location=device))
model.to(device)
model.eval()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
results = []

# Evaluate
with torch.no_grad():
    for session_id, input_text, target_text in session_data:
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024).to(device)
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        scores = scorer.score(target_text, generated)

        results.append({
            "SessionID": session_id,
            "Generated": generated,
            "Target": target_text,
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-L": scores["rougeL"].fmeasure
        })

        print(f"[Session {session_id}]")
        print("Generated:\n", generated)
        print("Target:\n", target_text)
        print("Scores:", scores)
        print("--------")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(generated_output_folder, "bart_eval_session_level.csv"), index=False)

# Print average scores
print("\nFINAL AVERAGE ROUGE SCORES:")
print("ROUGE-1:", df["ROUGE-1"].mean())
print("ROUGE-2:", df["ROUGE-2"].mean())
print("ROUGE-L:", df["ROUGE-L"].mean())
'''
'''import os
import json
import torch
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# Helper: merge utterances
def build_test_pair(input_path, target_path):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)

    input_lines = []
    for subsec in subsection_list:
        uttr = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        value = uttr.values[0] if not uttr.empty else "Nothing reported"
        input_lines.append(f"{subsec}: {value}")
    input_text = "\n".join(input_lines)

    target_lines = []
    for subsec in subsection_list:
        val = str(target_df[subsec].iloc[0]) if pd.notna(target_df[subsec].iloc[0]) else "Nothing reported"
        target_lines.append(f"{subsec}: {val}")
    target_text = "\n".join(target_lines)

    return input_text, target_text

# Dataset
class SessionDataset(Dataset):
    def __init__(self, file_list):
        self.examples = []
        for path in file_list:
            base = os.path.basename(path).replace(".csv", "")
            input_path = os.path.join(test_input_dir, f"{base}_utterances_grouping.csv")
            target_path = os.path.join(target_dir, f"{base}_output.csv")
            try:
                src, tgt = build_test_pair(input_path, target_path)
                self.examples.append((src, tgt, base))
            except Exception as e:
                print(f"Skipping {base} due to error: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src, tgt, sid = self.examples[idx]
        return {"input_text": src, "target_text": tgt, "session_id": sid}

# Load tokenizer & model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.load_state_dict(torch.load("models/bart_summarizer_session_level.pth", map_location=device))
model.to(device)
model.eval()

# Load test data
dataset = SessionDataset(test_files)
dataloader = DataLoader(dataset, batch_size=1)

# Metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method4
all_scores = []

# Inference loop
with torch.no_grad():
    for batch in dataloader:
        input_text = batch["input_text"][0]
        target_text = batch["target_text"][0]
        session_id = batch["session_id"][0]

        enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=256
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        rouge = scorer.score(target_text, generated)
        bleu = sentence_bleu([target_text.split()], generated.split(), smoothing_function=smooth_fn)
        meteor = meteor_score([target_text.split()], generated.split())

        all_scores.append({
            "SessionID": session_id,
            "Input": input_text,
            "Target": target_text,
            "Generated": generated,
            "ROUGE-1": rouge["rouge1"].fmeasure,
            "ROUGE-2": rouge["rouge2"].fmeasure,
            "ROUGE-L": rouge["rougeL"].fmeasure,
            "BLEU": bleu,
            "METEOR": meteor
        })

# Convert to DataFrame
results_df = pd.DataFrame(all_scores)
results_df.to_csv("data/test_generated_soap_notes/bart_eval_fullsession_metrics.csv", index=False)

# Print averages
print("Average ROUGE-1:", results_df["ROUGE-1"].mean())
print("Average ROUGE-2:", results_df["ROUGE-2"].mean())
print("Average ROUGE-L:", results_df["ROUGE-L"].mean())
print("Average BLEU:", results_df["BLEU"].mean())
print("Average METEOR:", results_df["METEOR"].mean())
'''

'''import os
import json
import torch
import nltk
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# Helper: merge utterances
def build_test_pair(input_path, target_path):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)

    input_lines = []
    for subsec in subsection_list:
        uttr = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        value = uttr.values[0] if not uttr.empty else "Nothing reported"
        input_lines.append(f"{subsec}: {value}")
    input_text = "\n".join(input_lines)

    target_lines = []
    for subsec in subsection_list:
        val = str(target_df[subsec].iloc[0]) if pd.notna(target_df[subsec].iloc[0]) else "Nothing reported"
        target_lines.append(f"{subsec}: {val}")
    target_text = "\n".join(target_lines)

    return input_text, target_text

# Dataset
class SessionDataset(Dataset):
    def __init__(self, file_list):
        self.examples = []
        for path in file_list:
            base = os.path.basename(path).replace(".csv", "")
            input_path = os.path.join(test_input_dir, f"{base}_utterances_grouping.csv")
            target_path = os.path.join(target_dir, f"{base}_output.csv")
            try:
                src, tgt = build_test_pair(input_path, target_path)
                self.examples.append((src, tgt, base))
            except Exception as e:
                print(f"Skipping {base} due to error: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src, tgt, sid = self.examples[idx]
        return {"input_text": src, "target_text": tgt, "session_id": sid}

# Load tokenizer & model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.load_state_dict(torch.load("models/bart_summarizer_session_level.pth", map_location=device))
model.to(device)
model.eval()

# Load test data
dataset = SessionDataset(test_files)
dataloader = DataLoader(dataset, batch_size=1)

# Metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method4
all_scores = []

# Inference loop
all_preds, all_refs = [], []
with torch.no_grad():
    for batch in dataloader:
        input_text = batch["input_text"][0]
        target_text = batch["target_text"][0]
        session_id = batch["session_id"][0]

        enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=256
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        rouge = scorer.score(target_text, generated)
        bleu = sentence_bleu([target_text.split()], generated.split(), smoothing_function=smooth_fn)
        meteor = meteor_score([target_text.split()], generated.split())

        all_preds.append(generated)
        all_refs.append(target_text)

        all_scores.append({
            "SessionID": session_id,
            "Input": input_text,
            "Target": target_text,
            "Generated": generated,
            "ROUGE-1": rouge["rouge1"].fmeasure,
            "ROUGE-2": rouge["rouge2"].fmeasure,
            "ROUGE-L": rouge["rougeL"].fmeasure,
            "BLEU": bleu,
            "METEOR": meteor
        })

# Compute BERTScore (outside loop for efficiency)
P, R, F1 = bert_score(all_preds, all_refs, lang="en", rescale_with_baseline=True)
results_df = pd.DataFrame(all_scores)
results_df["BERTScore_P"] = P.tolist()
results_df["BERTScore_R"] = R.tolist()
results_df["BERTScore_F1"] = F1.tolist()

# Save results
results_df.to_csv("data/test_generated_soap_notes/bart_eval_fullsession_metrics.csv", index=False)

# Print averages
print("Average ROUGE-1:", results_df["ROUGE-1"].mean())
print("Average ROUGE-2:", results_df["ROUGE-2"].mean())
print("Average ROUGE-L:", results_df["ROUGE-L"].mean())
print("Average BLEU:", results_df["BLEU"].mean())
print("Average METEOR:", results_df["METEOR"].mean())
print("Average BERTScore-F1:", results_df["BERTScore_F1"].mean())
'''
import os
import json
import torch
import nltk
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

nltk.download('wordnet')
nltk.download('omw-1.4')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# Helper: build session-level input and target
def build_test_pair(input_path, target_path):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)

    input_lines = []
    for subsec in subsection_list:
        uttr = input_df.loc[input_df["Subsection"] == subsec, "Grouped_Utterances"]
        value = uttr.values[0] if not uttr.empty else "Nothing reported"
        input_lines.append(f"{subsec}: {value}")
    input_text = "\n".join(input_lines)

    target_lines = []
    for subsec in subsection_list:
        val = str(target_df[subsec].iloc[0]) if pd.notna(target_df[subsec].iloc[0]) else "Nothing reported"
        target_lines.append(f"{subsec}: {val}")
    target_text = "\n".join(target_lines)

    return input_text, target_text

# Dataset class
class SessionDataset(Dataset):
    def __init__(self, file_list):
        self.examples = []
        for path in file_list:
            base = os.path.basename(path).replace(".csv", "")
            input_path = os.path.join(test_input_dir, f"{base}_utterances_grouping.csv")
            target_path = os.path.join(target_dir, f"{base}_output.csv")
            try:
                src, tgt = build_test_pair(input_path, target_path)
                self.examples.append((src, tgt, base))
            except Exception as e:
                print(f"Skipping {base} due to error: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src, tgt, sid = self.examples[idx]
        return {"input_text": src, "target_text": tgt, "session_id": sid}

# Load tokenizer & model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.load_state_dict(torch.load("models/bart_summarizer_session_level.pth", map_location=device))
model.to(device)
model.eval()

# Load test data
dataset = SessionDataset(test_files)
dataloader = DataLoader(dataset, batch_size=1)

# Scorers
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method4

# Score containers
all_scores = []
all_preds, all_refs = [], []

# Inference and metric collection
with torch.no_grad():
    for batch in dataloader:
        input_text = batch["input_text"][0]
        target_text = batch["target_text"][0]
        session_id = batch["session_id"][0]

        enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=256
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ROUGE
        rouge = scorer.score(target_text, generated)
        # BLEU
        bleu = sentence_bleu([target_text.split()], generated.split(), smoothing_function=smooth_fn)
        # METEOR
        meteor = meteor_score([target_text.split()], generated.split())

        all_preds.append(generated)
        all_refs.append(target_text)

        all_scores.append({
            "SessionID": session_id,
            "Input": input_text,
            "Target": target_text,
            "Generated": generated,
            "ROUGE-1": rouge["rouge1"].fmeasure,
            "ROUGE-2": rouge["rouge2"].fmeasure,
            "ROUGE-L": rouge["rougeL"].fmeasure,
            "BLEU": bleu,
            "METEOR": meteor
        })

# --- Compute BERTScore using local RoBERTa-large ---
P, R, F1 = bert_score(
    all_preds,
    all_refs,
    model_type="roberta-large",
    lang="en",
    device=device,
    rescale_with_baseline=True
)

# Combine results
results_df = pd.DataFrame(all_scores)
results_df["BERTScore_P"] = P.tolist()
results_df["BERTScore_R"] = R.tolist()
results_df["BERTScore_F1"] = F1.tolist()

# Save results
results_df.to_csv("data/test_generated_soap_notes/bart_eval_fullsession_metrics.csv", index=False)

# Print averages
print("Average ROUGE-1:", results_df["ROUGE-1"].mean())
print("Average ROUGE-2:", results_df["ROUGE-2"].mean())
print("Average ROUGE-L:", results_df["ROUGE-L"].mean())
print("Average BLEU:", results_df["BLEU"].mean())
print("Average METEOR:", results_df["METEOR"].mean())
print("Average BERTScore-F1:", results_df["BERTScore_F1"].mean())
