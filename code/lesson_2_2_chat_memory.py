
"""
Lesson 2.2: Sweet Crumbs Bakery Bot with Memory

Concepts:
- Lists and dictionaries
- Chat history
- Adding user and assistant messages
- Limiting conversation history
- Resetting memory
- Counting tokens
"""

import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env
load_dotenv()

# Check that the API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

from langchain.chat_models import init_chat_model

# Keep at most 6 conversation messages, excluding the system prompt.
MAX_MESSAGES = 6

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are the friendly assistant of Sweet Crumbs bakery in Lahore. "
        "Answer in 2 short lines. Be helpful and friendly."
    ),
}

# Create the model once, before the chat loop.
# Create the model once
llm = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    temperature=0.2,
)
# The system message contains the bot's permanent instructions.
messages = [SYSTEM_MESSAGE.copy()]

print("\nSweet Crumbs Bakery Bot")
print("Ask me anything about cakes and bakery products.")
print("Commands: /reset, /count, bye\n")

while True:
    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBot: Goodbye!")
        break

    # Exit the chatbot.
    if user_text.lower() == "bye":
        print("Bot: Goodbye!")
        break

    # Clear the conversation but preserve system instructions.
    elif user_text.lower() == "/reset":
        messages = [SYSTEM_MESSAGE.copy()]
        print("Bot: Memory cleared. Let's start a new conversation.\n")
        continue

    # Display the number of messages currently stored.
    elif user_text.lower() == "/count":
        print(f"Bot: {len(messages) - 1} conversation messages stored.\n")
        continue

    # Reject empty input.
    elif not user_text:
        print("Bot: Please type a question.\n")
        continue

    # Save the user's message.
    messages.append({
        "role": "user",
        "content": user_text,
    })

    # Keep the system message and the newest conversation messages.
    messages = [
        messages[0],
        *messages[1:][-MAX_MESSAGES:],
    ]

    try:
        # Send the conversation history to the model.
        response = llm.invoke(messages)
        answer = response.content

        # Save the assistant's reply for future turns.
        messages.append({
            "role": "assistant",
            "content": answer,
        })

        print(f"Bot: {answer}")

        # Token usage depends on the model/provider.
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens")

        if input_tokens is not None:
            print(
                f"    (Memory: {len(messages) - 1} conversation messages, "
                f"{input_tokens} input tokens sent)\n"
            )
        else:
            print(
                f"    (Memory: {len(messages) - 1} conversation messages; "
                "token usage unavailable)\n"
            )

    except Exception as error:
        # Remove the failed user message so it isn't left as an
        # unanswered turn in the conversation history.
        if messages and messages[-1]["role"] == "user":
            messages.pop()

        print(f"Bot: Something went wrong: {error}")
        print("Check that Ollama is running and the model is available.\n")
