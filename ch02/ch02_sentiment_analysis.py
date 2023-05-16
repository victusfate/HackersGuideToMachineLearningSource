from transformers import pipeline

# Initialize the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis")

# Input text
text = "I love the new design of your website!"

# Perform sentiment analysis
result = nlp(text)[0]

print(f"label: {result['label']}, with score: {result['score']}")
