import os
from dotenv import load_dotenv

load_dotenv()

# LLM API Keys
LLM_MODEL_NAME = "models/gemini-2.5-flash"
GOOGLE_API_KEY = "AIzaSyAHQ5TkQVO7-DE-Hf9PWdiNPLjACL7pEu0"

# Neo4j Credentials
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Anika2005"
NEO4J_DATABASE= "neo4j"

# Vector DB Configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Data Paths
PDF_DIRECTORY = r"C:\Users\Vansh\OneDrive\Desktop\corporate_intelligence_rag\data\raw_pdfs"
