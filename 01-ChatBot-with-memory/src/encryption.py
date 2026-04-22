import os
from cryptography.fernet import Fernet

def get_cipher():
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise Exception("ENCRYPTION_KEY not found in environment.")
    return Fernet(key.encode())

def encrypt(text: str) -> str:
    return get_cipher().encrypt(text.encode()).decode()

def decrypt(text: str) -> str:
    return get_cipher().decrypt(text.encode()).decode()
