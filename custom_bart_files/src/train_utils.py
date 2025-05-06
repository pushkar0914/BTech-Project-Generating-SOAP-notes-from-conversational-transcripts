import torch
from transformers import AutoTokenizer

def extract_entities_from_tokens(token_ids, tokenizer):
    """
    Extract set of entity token strings from a sequence of token IDs using simple heuristics.
    This dummy implementation considers tokens starting with a capital letter (after decoding) as entities.
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    words = text.split()
    entities = set()
    for w in words:
        # Simple heuristic: if capitalized and longer than 3 or contains digits, consider it an entity
        if (len(w) > 3 and w[0].isupper()) or any(ch.isdigit() for ch in w):
            entities.add(w.strip(".,:;"))
    return entities

def compute_ner_penalty(input_ids: torch.Tensor, output_ids: torch.Tensor, model, penalty_weight=0.05):
    """
    Compute NER penalty based on input vs output entities.
    - Penalizes hallucinated entities (in output not in input) and missed entities (in input not in output).
    This version uses the input entity set attached to the model and extracts the output entity set by decoding
    the output token IDs after filtering out negative values.
    """
    # Use model's tokenizer if available, otherwise AutoTokenizer
    try:
        tokenizer = model.tokenizer
    except AttributeError:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    
    # Get the input entity set from the model (if it has been attached)
    try:
        input_entity_set = model.input_entity_set
    except AttributeError:
        # Fallback: decode input_ids and extract entities using a simple heuristic
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        input_entity_set = set(input_text.split())  # not ideal; should have been provided
    # For output, filter out negative tokens (like -100) before decoding
    out_ids = [tok for tok in output_ids[0].tolist() if tok >= 0]
    output_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    # Use the attached NER model (assumed to be available as model.ner_model)
    output_doc = model.ner_model(output_text)
    output_entity_set = set(ent.text for ent in output_doc.ents)
    #print(f"output set : {output_entity_set}")
    missed = input_entity_set.difference(output_entity_set)
    hallucinated = output_entity_set.difference(input_entity_set)
    penalty = penalty_weight * (len(missed) + len(hallucinated))
    penalty_tensor = torch.tensor(penalty, device=input_ids.device, dtype=torch.float32)
    #print(f"ner penalty : {penalty}")
    # Average penalty per batch element (here batch size is assumed to be 1)
    return penalty_tensor
