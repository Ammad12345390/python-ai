
"""Task 2.1: AI Tutor — Your First AI Call."""

import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env
load_dotenv()

# Check that the API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

SYSTEM = (
    "You are a friendly AI tutor. "
    "For beginners, explain using simple words and examples. "
    "For experts, use technical details and precise terminology. "
    "Keep your answer short and clear."
)

# Create the model once
llm = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    temperature=0.2,
)


def main():
    """Ask for a topic and level, then explain it using AI."""

    topic = input("Topic: ").strip()
    level = input("Level (beginner or expert): ").strip().lower()

    # Validate the user's level
    if level not in ("beginner", "expert"):
        print("Please enter beginner or expert.")
        return

    print("AI Tutor: Let me explain!")

    start_time = time.perf_counter()

    response = llm.invoke([
        {"role": "system", "content": f"{SYSTEM} The student's level is {level}."},
        {"role": "user", "content": f"Explain this topic: {topic}"},
    ])

    elapsed = time.perf_counter() - start_time

    print(f"\nAI Tutor: {response.text}")

    # Display token usage and response time
    tokens = response.usage_metadata["total_tokens"]
    print(f"({tokens} tokens in {elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
