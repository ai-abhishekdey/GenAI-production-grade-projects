from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from src.llm import get_llm
from src.database import get_history, save_message, save_tokens


# Build Chain function

def build_chain():
    """
    Creates the LLM chain.

    Structure:
    - System message (instruction)
    - Chat history (multi-turn context)
    - Current user input
    """

    # prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),   # Inject past conversation here
        ("human", "{input}"),             # Current user input
    ])

    # model
    model = get_llm()
    
    # chain
    chain = prompt | model | StrOutputParser()

    return chain


# Chat Function 

def chat(chain, client, username, user_input):
    """
    Handles one full chat turn:
    1. Load previous chat history (decrypted)
    2. Send history + current input to LLM
    3. Save user + AI messages (encrypted)
    4. Track token usage
    """

    # Get previous chat history
    history = get_history(client, username, limit=20)

    #  Call LLM with history
    # get_openai_callback tracks token usage
    
    with get_openai_callback() as cb:
        response = chain.invoke({
            "input": user_input,
            "history": history
        })

    # Save messages (ENCRYPTED)
    # Save AFTER LLM call to avoid duplication in history

    save_message(client, username, "user", user_input)
    save_message(client, username, "assistant", response)

    #  Save token usage
    save_tokens(client, username, user_input, cb)

    #  Return response
    return response, cb
