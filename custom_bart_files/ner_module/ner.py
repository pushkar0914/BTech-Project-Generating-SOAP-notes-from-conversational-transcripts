# ner.py
'''import spacy
from spacy.tokens import Doc

def load_model(model_path: str = "./model/model-last"):
    """
    Load a spaCy model.
    
    Parameters:
        model_path (str): Path to a custom model. If None, loads the default 'en_core_web_sm'.
    
    Returns:
        nlp: The loaded spaCy language model.
    """
    if model_path:
        nlp = spacy.load(model_path)
        print(2)
    else:
        # Load the default English model. Make sure to install it if you haven't:
        # python -m spacy download en_core_web_sm
        print(1)
        nlp = spacy.load("en_core_web_sm")
    return nlp

def perform_ner_on_tokens(nlp, tokens):
    """
    Process a list of tokens through the NER pipeline.
    
    Parameters:
        nlp: The spaCy language model.
        tokens (List[str]): A list of token strings.
    
    Returns:
        List[str]: A list of entity labels corresponding to each token. If a token is
                   part of an entity, its entity type is returned; otherwise, an empty string.
    
    Notes:
        This function creates a Doc object from the tokens using the spaCy vocab,
        then runs the NER component. It relies on token-level attributes (ent_iob_ and ent_type_).
        If your original pipeline uses custom tokenization, ensure that this method
        of constructing a Doc aligns with that process.
    """
    # Construct a Doc object from tokens.
    doc = Doc(nlp.vocab, words=tokens)
    
    # Process the doc to get NER annotations.
    doc = nlp(doc)
    
    # For each token, return the entity type if it belongs to an entity, else return an empty string.
    ner_tags = []
    for token in doc:
        if token.ent_iob_ != 'O':
           # ner_tags.append(token.ent_type_)
            ner_tags.append(1)
        else:
            ner_tags.append(0)
    return ner_tags '''
import spacy
from spacy.tokens import Doc
import os

def load_model(model_path: str = "./ner_module/model/model-last", ruler_path: str = "./ner_module/annotation_ner/entity_ruler"):
    """
    Load a spaCy model from model_path and add an entity_ruler component with patterns
    from ruler_path. If no model_path is provided, load the default 'en_core_web_sm'.
    """
    if model_path:
        nlp = spacy.load(model_path)
        print("Loaded custom model from", model_path)
    else:
        print("Loading default model 'en_core_web_sm'")
        nlp = spacy.load("en_core_web_sm")
    
    # Remove existing entity_ruler if present to reload fresh patterns.
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    
    # Add entity_ruler before the ner component.
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Load patterns from the specified directory, if it exists.
    if os.path.exists(ruler_path):
        try:
            ruler.from_disk(ruler_path)
            print("EntityRuler patterns loaded from", ruler_path)
        except Exception as e:
            print("Error loading EntityRuler patterns:", e)
    else:
        print(f"WARNING: Ruler path '{ruler_path}' not found. No patterns loaded.")
    
    return nlp

def perform_ner_on_tokens(nlp, tokens, return_entities=False):
    """
    Process a list of tokens through the NER pipeline.
    
    Parameters:
        nlp: The spaCy language model (with entity_ruler and ner).
        tokens (List[str]): A list of token strings.
        return_entities (bool): If True, also return a set of named entities detected.
    
    Returns:
        If return_entities is False:
           List[str]: A list of per-token tags (entity type if part of an entity, or 0 otherwise).
        Else:
           Tuple[List[str], Set[str]]: (per-token tags, set of named entity strings)
    """
    # Create a Doc from the provided tokens (bypassing default tokenization).
    doc = Doc(nlp.vocab, words=tokens)
    # Process the Doc through the pipeline.
    doc = nlp(doc)
    ner_tags = []
    for token in doc:
        if token.ent_iob_ != 'O':
            ner_tags.append(token.ent_type_)
        else:
            ner_tags.append(0)
    if return_entities:
        entity_set = set(ent.text for ent in doc.ents)
        return ner_tags, entity_set
    return ner_tags
