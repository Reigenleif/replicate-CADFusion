from openai import OpenAI
from dotenv import load_dotenv
import os

def get_client():
    load_dotenv()
    base_url = os.getenv('INFERENCE_BASE_URL', 'https://api.openai.com')

    client = OpenAI(
        base_url=base_url,
        api_key="ollama", # Assumes Ollama for Inference Server
    )
    
    return client