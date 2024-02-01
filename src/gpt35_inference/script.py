#Note: The openai-python library support for Azure OpenAI is in preview.
      #Note: This code sample requires OpenAI Python library version 0.28.1 or lower.
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://met-openai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("9bb3c2b3e1aa4d2799fd35373fe8807e")

message_text = [{"role":"system","content":"As an informative bot, please return the numerical digits as the output. "},{"role":"user","content":"In 2019, GDP per capita in ZAF was?         \nGenerate the numerical values only. Do not generate any text even including the input query."},{"role":"assistant","content":"I'm sorry, but I need more information to answer your question. What is \"ZAF\"?"}]

completion = openai.ChatCompletion.create(
  engine="needhelp",
  messages = message_text,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None
)