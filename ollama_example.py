from openai import OpenAI

# Run ollama server with gpu:
#   OLLAMA_USE_CUDA=1 ollama run llama3.1:70b

client = OpenAI(
    base_url = 'http://localhost:11434/v1', # we may use GCP proxy of our lab's machines
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  # model="llama3.1:70b",
  model="deepseek-r1",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)