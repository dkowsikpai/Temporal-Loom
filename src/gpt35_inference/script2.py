import openai

openai.api_type = "azure"
openai.api_base = "https://met-openai.openai.azure.com/openai/deployments/needhelp/chat/completions?api-version=2023-07-01-preview" # APIM Endpoint
openai.api_version = "2023-07-01-preview"
openai.api_key = "9bb3c2b3e1aa4d2799fd35373fe8807e" #DO NOT USE ACTUAL AZURE OPENAI SERVICE KEY


response = openai.Completion.create(engine="modelname",  
                                    prompt="In 2022, GDP per capita in YEM was", temperature=1,  
                                    max_tokens=200,  top_p=0.5,  
                                    frequency_penalty=0,  
                                    presence_penalty=0,  
                                    stop=None) 
