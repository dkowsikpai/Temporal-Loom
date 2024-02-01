curl https://met-openai.openai.azure.com/openai/deployments/needhelp/chat/completions?api-version=2023-07-01-preview \
  -H "Content-Type: application/json" \
  -H "api-key: 9bb3c2b3e1aa4d2799fd35373fe8807e" \
  -d '{"messages":[{"role": "system", "content": "As an informative bot, please return the numerical digits as the output. "},{"role": "user", "content": "Does Azure OpenAI support customer managed keys?"]}'