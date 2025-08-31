import secrets

# Generate the secret key
key = secrets.token_hex(16)

# Write the key to a file
with open("secret_key.txt", "w") as file:
    file.write(key)

print("Key written to 'secret_key.txt'")
