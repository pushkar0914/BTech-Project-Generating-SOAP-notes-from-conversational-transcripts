import spacy
from spacy.tokens import Doc

nlp = spacy.load("./model/model-best")  # Loads the trained model

# Add an entity_ruler component BEFORE the ner component
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.from_disk("./annotation_ner/entity_ruler")  # Adjust the path to your patterns folder

print(nlp.pipe_names)