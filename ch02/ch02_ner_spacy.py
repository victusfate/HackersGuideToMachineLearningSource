import spacy
from spacy import displacy

# Load a SpaCy model (in this case, the English model)
nlp = spacy.load("en_core_web_sm")

# Input text
text = "Google LLC is an American multinational technology company that specializes in Internet-related services and products."

# Process the text
doc = nlp(text)

# Print the entities in the text
for ent in doc.ents:
    print('ent',ent)

#You can also visualize the entities in the text using SpaCy's built-in visualizer:
displacy.render(doc, style='ent')
