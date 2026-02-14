from dotenv import load_dotenv
import os

# Check which .env file is loaded
print(load_dotenv())  # Should return True
print(f"Current dir: {os.getcwd()}")

# Check the raw value
api_key = os.getenv("OPENAI_API_KEY")
print(f"Raw key: '{api_key}'")
print(f"Length: {len(api_key) if api_key else 'None'}")
print(f"Repr: {repr(api_key)}")