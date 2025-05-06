# example.py
'''from ner import load_model, perform_ner_on_tokens

def main():
    # Load a spaCy model. You can pass a custom model path if needed.
    nlp = load_model()  # This loads the default model "en_core_web_sm".
    
    # Example list of tokens.
    tokens = ["Depression", "Dementia", "OCD", "Paris", "anxiety", "anorexia", "OCD"]
    
    # Get NER outputs: for tokens that are part of an entity, the entity type is returned.
    ner_results = perform_ner_on_tokens(nlp, tokens)
    
    # Print the tokens with their corresponding NER tags.
    print("Tokens:   ", tokens)
    print("NER Tags: ", ner_results)

if __name__ == "__main__":
    main()'''
from ner import load_model, perform_ner_on_tokens

def main():
    # Load the spaCy model along with the entity_ruler.
    nlp = load_model()  # Uses model_last and adds patterns from "./annotation_ner/entity_ruler"
    
    # Example list of tokens (pre-tokenized). The order matters for context-dependent models,
    # but the entity_ruler should match known patterns regardless of position.
    tokens = ["Depression", "Dementia", "harmaline", "Paris", "anxiety", "anorexia", "OCD"]
    
    # Get NER outputs: flag each token as 1 if it is recognized as an entity, else 0.
    ner_results = perform_ner_on_tokens(nlp, tokens)
    
    # Print the tokens with their corresponding NER flags.
    print("Tokens:   ", tokens)
    print("NER Flags:", ner_results)
    
    # Print the pipeline components to verify the order.
    #print("Pipeline components:", nlp.pipe_names)

if __name__ == "__main__":
    main()

