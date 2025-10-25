from dotenv import load_dotenv, find_dotenv
import os

path = find_dotenv()
print("🔍 Found .env path:", path)

success = load_dotenv(dotenv_path=path)
print("✅ load_dotenv returned:", success)

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("NEWSAPI_KEY:", os.getenv("NEWSAPI_KEY"))