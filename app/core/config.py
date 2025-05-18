import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

class Settings:
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI-Powered FastAPI App"
    
    # API Keys - support multiple keys for fallback
    OPENAI_API_KEYS: List[str] = [
        key.strip() for key in os.getenv("OPENAI_API_KEY", "").split(",") 
        if key.strip()
    ]
    print('OPENAI_API_KEYS', OPENAI_API_KEYS)
    # Model settings
    DEFAULT_MODEL_NAME: str = "gpt-3.5-turbo"
    FALLBACK_MODEL_NAME: str = "gpt-3.5-turbo-instruct"  # Less expensive fallback
    
    # Rate limit settings
    MAX_RETRIES: int = 2
    RETRY_DELAY: float = 1.0  # seconds
    
    # Fallback response when all API options are exhausted
    FALLBACK_RESPONSE: str = "I'm sorry, but I'm currently experiencing high demand. Please try again later."
    
settings = Settings()