import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():

    model = ChatOpenAI(model="gpt-4o")
    
    return model
