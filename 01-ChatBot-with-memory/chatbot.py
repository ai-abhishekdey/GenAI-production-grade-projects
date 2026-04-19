# chatbot.py

from src.database import connect, get_summary
from src.chain import build_chain, chat

# Connect to Database

client = connect()

# User Login

username = input("Enter your username: ").strip()
print(f"\nHello {username}! Type 'exit' to quit.\n")

# Chat Loop

chain  = build_chain()

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        summary = get_summary(client, username)
        print(f"\n  Messages : {summary['messages']}")
        print(f"  Tokens   : {summary['tokens']}")
        print(f"  Cost     : ${summary['cost']:.6f}")
        print("\nGoodbye! Your chat is saved.")
        break

    response, cb = chat(chain, client, username, user_input)
    print(f"AI: {response}")
    print(f"[Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.6f}]\n")
