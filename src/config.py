import os
from dotenv import load_dotenv

load_dotenv()

# LLM API Keys
LLM_MODEL_NAME = "models/gemini-2.5-flash"
GOOGLE_API_KEY = "your_google_api_key"

# Neo4j Credentials
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password"
NEO4J_DATABASE= "neo4j"

# Vector DB Configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Data Paths
PDF_DIRECTORY = "Your_path"
