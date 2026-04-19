import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBChatMessageHistory

load_dotenv()

# Read database names from .env 

DB_NAME = os.getenv("MONGO_DB",   "chatbot_db")
CHAT_COL= os.getenv("MONGO_CHAT_COLLECTION",  "chat_histories")
TOKEN_COL = os.getenv("MONGO_TOKEN_COLLECTION", "token_usage")

# Connect to MongoDB

def connect():
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        print("-> Connected to MongoDB\n")
        return client
    except Exception:
        print("-> Could not connect to MongoDB")
        sys.exit(1)

# Function to get chat history from MongoDB database

def get_history(username):
    return MongoDBChatMessageHistory(
        connection_string=os.getenv("MONGODB_URI"),
        session_id=username,
        database_name=DB_NAME,
        collection_name=CHAT_COL,
    )

# Function to save token usage 

def save_tokens(client, username, user_input, cb):
    client[DB_NAME][TOKEN_COL].insert_one({
        "username"          : username,
        "timestamp"         : datetime.now(timezone.utc),
        "user_message"      : user_input,
        "prompt_tokens"     : cb.prompt_tokens,
        "completion_tokens" : cb.completion_tokens,
        "total_tokens"      : cb.total_tokens,
        "cost_usd"          : cb.total_cost,
    })

# Function to get summary of token usage and cost 

def get_summary(client, username):
    records = list(client[DB_NAME][TOKEN_COL].find({"username": username}))
    return {
        "messages" : len(records),
        "tokens"   : sum(r["total_tokens"] for r in records),
        "cost"     : sum(r["cost_usd"]     for r in records),
    }
