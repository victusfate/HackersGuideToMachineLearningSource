from transformers import pipeline

# Load the text summarization pipeline
summarizer = pipeline("summarization")

# Example text to be summarized
text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vestibulum felis id massa fermentum consequat. 
Vivamus auctor posuere turpis, at placerat arcu blandit ac. Sed lobortis congue tortor, ut eleifend orci cursus eu. 
Phasellus accumsan lacus eget sem faucibus, vel molestie tortor ultrices. Aenean eu viverra nisl. 
"""

# Generate a summary of the text
summary = summarizer(text, max_length=50, min_length=20, do_sample=True)[0]['summary_text']

# Print the generated summary
print("Summary:", summary)
