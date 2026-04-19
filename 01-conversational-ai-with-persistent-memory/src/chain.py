from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.callbacks import get_openai_callback
from src.llm import get_llm
from src.database import get_history, save_tokens


# Build Chain

def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    model = get_llm()
    
    chain = prompt | model | StrOutputParser()

    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# Chat

def chat(chain, client, username, user_input):
    with get_openai_callback() as cb:
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": username}},
        )
    save_tokens(client, username, user_input, cb)
    return response, cb
