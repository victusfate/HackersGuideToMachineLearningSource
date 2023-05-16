from transformers import BartForQuestionAnswering, BartTokenizer

# Initializing a BART base style configuration
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForQuestionAnswering.from_pretrained('facebook/bart-large')

# The text that you want to extract the answer from
context = "The Eiffel Tower is located in Paris."

# The question that you want answered
question = "Where is the Eiffel Tower located?"

inputs = tokenizer(question, context, return_tensors='pt')

output = model(**inputs)

# Get the most probable start and end tokens
answer_start = torch.argmax(output.start_logits) 
answer_end = torch.argmax(output.end_logits) + 1 

# Get the answer
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

print(answer)
