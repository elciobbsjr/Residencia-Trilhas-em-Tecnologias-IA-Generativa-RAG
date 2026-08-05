import os
from dotenv import load_dotenv
from openai import OpenAI

# Carrega as variáveis do .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Usa o modelo definido no .env
modelo = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

response = client.chat.completions.create(
    model=modelo,
    messages=[
        {"role": "user", "content": "Qual a capital do Brasil?"}
    ]
)

print(response.choices[0].message.content)