import openai

openai.api_key = 'your-api-key'

def chatbot(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "text-davinci-003" depending on the model available
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )
    return response['choices'][0]['message']['content']

while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    response = chatbot(message)
    print("Bot: ", response)