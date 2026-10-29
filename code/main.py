import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

SYSTEM = "You are the friendly assistant of Sweet Crumbs bakery in Lahore. Answer in 2 short lines."
# Load environment variables from .env
load_dotenv()

# Check that the API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

# Create the model once
llm = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    temperature=0.3
)

print("Sweet Crumbs Bot: Ask me anything. Type 'bye' to exit.")

while True:
    message = input("You: ").strip()

    # Rules first: free, instant and always correct
    if message.lower() == "bye":
        print("Bot: Goodbye!")
        break
    elif "refund" in message.lower():
        print("Bot: A manager will call you within 1 hour.")
    else:
        # Everything the rules cannot answer goes to the AI
        response = llm.invoke([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": message},
        ])
        print(f"Bot: {response.text}")
        print(f"     ({response.usage_metadata['output_tokens']} tokens)")