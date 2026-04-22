from cryptography.fernet import Fernet

encryption_key=Fernet.generate_key().decode()

print("===============================")
print(f"ENCRYPTION_KEY={encryption_key}")
print("===============================")
