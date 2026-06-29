import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

messages = [{"role": "system", "content": "You are a concise, highly skilled Enterprise AI Agent."}
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ['quit', 'exit']:
        break
    messages.append({"role": "user", "content": user_input})

    chat_completion = client.chat.completions.create(
        messages = messages,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=500,
    )

    ai_response = chat_completion.choices[0].message.content
    print(f"Agent: {ai_response}")

    messages.append({"role": "assistant", "content": ai_response})