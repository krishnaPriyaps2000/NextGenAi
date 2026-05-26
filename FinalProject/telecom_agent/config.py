import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Configuration
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY1", "")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY2", os.getenv("GEMINI_API_KEY1", ""))
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"  # Gemini embedding model

# Tavily Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LangSmith / LangChain Tracing Configuration
# Map LANGSMITH_ environment variables from .env to standard LANGCHAIN_ variables
langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    # Clean up key (strip quotes if present)
    langsmith_api_key = langsmith_api_key.strip('"\'')
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    
    # Enable tracing
    langsmith_tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "true"
    langsmith_tracing = langsmith_tracing.strip('"\'').lower()
    if langsmith_tracing in ["true", "1", "yes"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        
    # Project name (strip quotes)
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "Telecom"
    project = project.strip('"\'')
    os.environ["LANGCHAIN_PROJECT"] = project
    
    # Endpoint (strip quotes)
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
    endpoint = endpoint.strip('"\'')
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint

# SSL — disable verification for corporate proxy environments
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["REQUESTS_CA_BUNDLE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "chroma_db")

# Ensure directories exist
for d in [DATA_DIR, SYNTHETIC_DIR, VECTOR_STORE_DIR]:
    os.makedirs(d, exist_ok=True)
