import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config import LLM

load_dotenv()


def get_openai_llm():

    llm = ChatOpenAI(
        model=LLM,
        temperature=0.5,
        max_tokens=512
    )

    return llm


def get_summariser_llm():

    summariser_llm = ChatOpenAI(
        model=LLM,
        temperature=0.5,
        max_tokens=2500
    )

    return summariser_llm
