

from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GROQ_API_KEY")

if key:
    print("✅ API Key loaded successfully!")
    print(f"   Key starts with: {key[:8]}...")
else:
    print("❌ API Key NOT found. Check your .env file.")