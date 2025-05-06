import os
import glob
import torch
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# ─── Device & Model Loading ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_DIR = "./fine_tuned_pegasus"   # your Pegasus fine-tuned checkpoint
tokenizer = PegasusTokenizer.from_pretrained(MODEL_DIR)
model     = PegasusForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ─── Summarization Function for Pegasus ───────────────────────────────────────────
def summarize_sessions_pegasus(session_texts,
                               max_input_length=1024,
                               max_output_length=300):
    """
    session_texts: list of strings, each like "Session 1:\nHPI: ...\n..."
    returns: one‐paragraph abstractive summary
    """
    # build the single input string
    input_str = (
        "summarize: "
        + "\n\n".join(session_texts)
    )

    # tokenize + truncate
    inputs = tokenizer(
        input_str,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    ).to(device)

    # generate
    summary_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

# ─── Convenience: Summarize by CSV prefix ────────────────────────────────────────
def summarize_patient_from_csv_pegasus(prefix, csv_folder="TESTING/sessions"):
    paths = sorted(glob.glob(os.path.join(csv_folder, f"{prefix}_TRANSCRIPT*.csv")))
    session_texts = []
    for idx, path in enumerate(paths, 1):
        df = pd.read_csv(path).iloc[0]
        parts = [
            f"{col}: {val}".strip()
            for col, val in df.items()
            if str(val).strip().lower() != "\"nothing reported\""
        ]
        session_texts.append(f"Session {idx}:\n" + "\n".join(parts))

    return summarize_sessions_pegasus(session_texts)

# ─── Example Usage ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example 1: Manual session texts
    example_sessions = [
        '''Session 1:\nThe patient reports it took some time to adjust to living in Los Angeles, but overall found it somewhat easy. 
        - They mentioned feeling frustrated during a recent argument and have been good at controlling their temper. 
        - The patient expressed some regret about going to college right after high school, feeling they could have been further along in their career by now. 
        - They shared that they felt happy recently due to nice weather and had a memorable experience graduating high school. 
        - The patient also indicated they have been annoyed recently by a friend and chose to cut off that relationship
        The patient has not been diagnosed with depression or PTSD and has not served in the military.
        The patient is originally from Atlanta, Georgia, and enjoys living in Los Angeles for its weather and opportunities, although they dislike the congestion. 
                - They studied business and administration and plan to return to school next semester. 
                - The patient enjoys reading, cooking, and exercising to relax. 
                - They consider themselves somewhat local and visit their hometown once a year. 
                - The patient has a close relationship with their family, particularly their grandparents and parents, who have been positive influences in their life. 
                - They enjoy sports, going out with friends and family, and playing games for fun.
        The patient finds it easy to get a good night's sleep but feels irritated and lazy when they do not sleep well.
        ''',
        '''Session 2:\nThe patient is contemplating moving out but feels uncertain about readiness and financial stability. 
       - They are considering applying for management positions to improve their financial situation. 
       - Cooking and exercising are identified as coping mechanisms, though the patient struggles to maintain their exercise routine due to a busy schedule. 
       - The patient is also trying to express feelings of annoyance by writing things down before discussing them with others.
        The patient has been busy with schoolwork and has not kept up with their exercise routine as much as they would like. 
            - They are working on polishing their resume to apply for jobs.
        The patient will engage in a journaling exercise to articulate feelings better, particularly moments of annoyance. 
            - They will also set goals for job applications and exercise routines.
        The patient aims to apply for two to three jobs and to run twice this week. 
            - They will check in on these goals in the next session.''',
        '''Session 3:\nThe patient expressed feelings of anxiety regarding job applications and concerns about finding a fulfilling job. They are looking for roles that allow creativity and interaction with others.
        The patient has applied to three jobs and feels a mix of nervousness and relief. They are trying to be patient while waiting for responses. They also dealt with a disagreement with a friend and found writing about it helpful.
        The patient has been cooking and exercising, feeling accomplished and energized from these activities. They managed to cook a pasta recipe and went for two runs, which helped clear their mind.
        The patient has shown progress in communication skills, expressing feelings better after a disagreement and wanting to continue improving in this area.
        The patient is encouraged to network and reach out to people in fields of interest, setting a goal to do so in the coming week.'''
    ]
    print("Manual summary:\n", summarize_sessions_pegasus(example_sessions))

    # Example 2: From CSV folder
    patient_id = "300"
    summary = summarize_patient_from_csv_pegasus(patient_id, csv_folder="TESTING/sessions")
    print(f"\nSummary for patient {patient_id}:\n", summary)
