
# https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj
transformers_version = "v4.29.0" #@param ["main", "v4.29.0"] {allow-input: true}
print(f"Setting up everything with transformers version {transformers_version}")

# !pip install huggingface_hub>=0.14.1 git+https://github.com/huggingface/transformers@$transformers_version -q diffusers accelerate datasets torch soundfile sentencepiece opencv-python openai


agent_name = "OpenAI (API Key)" #@param ["StarCoder (HF Token)", "OpenAssistant (HF Token)", "OpenAI (API Key)"]
import getpass

from transformers.tools import OpenAiAgent
pswd = getpass.getpass('OpenAI API key:')
agent = OpenAiAgent(model="text-davinci-003", api_key=pswd)
print("OpenAI is initialized 💪")

from huggingface_hub import notebook_login
notebook_login()

boat = agent.run("Generate an image of a boat in the water")
boat
# a pretty image of a boat in the water

caption = agent.run("Can you caption the `boat_image`?", boat_image=boat)
caption
# 'a boat floating in the ocean with a small island in the background'