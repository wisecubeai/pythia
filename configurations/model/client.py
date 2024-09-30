import os
from openai import OpenAI


client = OpenAI(
    # This is the default and can be omitted
    api_key="not needed for a local LLM",
    base_url="http://localhost:8080/v1",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Who is Michael Jordan?",
        }
    ],
    model="tinyllama-1.1b-chat-v1.0.Q4_K_M",
    max_tokens=50,
    temperature=0.28,
    top_p=0.95,
    n=1,
    stream=False,
)

print(chat_completion)
