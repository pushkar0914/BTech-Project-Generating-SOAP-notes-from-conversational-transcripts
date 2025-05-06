import os
import json
import pandas as pd

# Define subsection mapping (15 valid subsections) - Ensure this is consistent with your main script
subsection_list = [
    "Presenting Problem / Chief Complaint", "Trauma History", "Substance Use History", "History of Present Illness (HPI)",
    "Medical and Psychiatric History", "Psychosocial History", "Risk Assessment", "Mental Health Observations",
    "Physiological Observations", "Current Functional Status", "Diagnostic Impressions", "Progress Evaluation",
    "Medications", "Therapeutic Interventions", "Next Steps"
]

def generate_intermediate_files_ground_truth(folder_path, output_folder):
    """
    Generate intermediate grouping CSVs for all session files in a given folder
    using the ground-truth classifications.

    Args:
        folder_path (str): Path to the folder containing the session CSV files.
        output_folder (str): Path to the folder where the intermediate CSV files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    all_files.sort()  # Sort for consistent processing

    for file_path in all_files:
        session_df = pd.read_csv(file_path, encoding="utf-8-sig")
        session_df = session_df[session_df['Classified_Subsection'] != "Insignificant"].copy() # Avoid SettingWithCopyWarning
        session_results = []
        for idx, row in session_df.iterrows():
            text = row['Utterance']
            labels = [l.strip() for l in row['Classified_Subsection'].split(",")]
            for label in labels:
                session_results.append([text, label])

        # Group utterances by subsection
        if session_results:
            results_df = pd.DataFrame(session_results, columns=["Utterance", "Subsection"])
            grouped = results_df.groupby("Subsection")["Utterance"].apply(lambda x: " ".join(x)).to_dict()
        else:
            grouped = {}

        grouping_data = []
        for subsec in subsection_list:
            grouped_utterances = grouped.get(subsec, "Nothing reported")
            grouping_data.append([subsec, grouped_utterances])

        grouping_df = pd.DataFrame(grouping_data, columns=["Subsection", "Grouped_Utterances"])
        base_name = os.path.basename(file_path).replace(".csv", "")
        intermediate_file = os.path.join(output_folder, f"{base_name}_utterances_grouping.csv")
        grouping_df.to_csv(intermediate_file, encoding="utf-8-sig", index=False)
        print(f"Saved ground-truth intermediate grouping for {base_name} to {intermediate_file}")

if __name__ == "__main__":
    session_splits_file = "session_splits.json"
    classified_folder = "classified_utterances"
    validation_output_folder = "validation_intermediate_files"
    validation_files_folder = "temp_validation_files"  # Temporary folder to hold validation files

    os.makedirs(validation_output_folder, exist_ok=True)
    os.makedirs(validation_files_folder, exist_ok=True)

    try:
        with open(session_splits_file, "r") as f:
            session_splits = json.load(f)
            validation_file_paths = session_splits.get("val", [])

        print(f"Validation files from session splits: {validation_file_paths}")

        if not validation_file_paths:
            print("No validation files found in session splits.")
        else:
            # Copy validation files to a temporary folder
            for val_file_path in validation_file_paths:
                source_path = os.path.join(classified_folder, os.path.basename(val_file_path))
                destination_path = os.path.join(validation_files_folder, os.path.basename(source_path))
                try:
                    # Check if the source file exists before attempting to copy
                    if os.path.exists(source_path):
                        with open(source_path, 'rb') as src, open(destination_path, 'wb') as dst:
                            dst.write(src.read())
                        print(f"Copied: {os.path.basename(source_path)} to {validation_files_folder}")
                    else:
                        print(f"Warning: Validation file not found at: {source_path}")
                except Exception as e:
                    print(f"Error copying {os.path.basename(source_path)}: {e}")

            # Generate intermediate files for the copied validation files
            generate_intermediate_files_ground_truth(validation_files_folder, validation_output_folder)

            # Clean up the temporary folder (optional)
            for file_name in os.listdir(validation_files_folder):
                file_path = os.path.join(validation_files_folder, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            os.rmdir(validation_files_folder)
            print(f"Cleaned up temporary folder: {validation_files_folder}")

    except FileNotFoundError:
        print(f"Error: Session splits file not found at {session_splits_file}")
    except Exception as e:
        print(f"An error occurred: {e}")