
import os
import shutil
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Import configuration parameters
from src.config import VECTOR_DB_PATH, EMBEDDING_MODEL_NAME

class VectorDBManager:
    def __init__(self):
        self.db_path = VECTOR_DB_PATH
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # Initialize the vector_db here, ensuring it loads from persistence
        # or creates a new one if the directory is empty/doesn't exist.
        # We call _get_or_create_db to handle this
        self.vector_db = self._get_or_create_db()

    def _get_or_create_db(self):
        """
        Initializes ChromaDB, loading from persistence if data exists,
        otherwise creates an empty one.
        """
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            print(f"Loading existing Chroma DB from {self.db_path}")
            # If directory exists and has content, load from it
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        else:
            print(f"Creating new or empty Chroma DB at {self.db_path}")
            # Create an empty Chroma instance for persistence.
            # Documents will be added later via add_documents.
            # Using .from_documents with [] just creates an empty collection, then it's persistent.
            # The issue was that .from_documents was being used to _add_ and _initialize_ with empty data,
            # which is not what we want. We just want to initialize for persistence.
            # A simpler way to get an empty persistent client:
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

    def add_documents(self, documents: List[Document]):
        """Adds documents to the vector database."""
        if not documents:
            print("No documents to add to vector DB.")
            return

        print(f"Adding {len(documents)} documents to vector DB...")
        # Chroma's add_documents method directly handles adding to the collection
        self.vector_db.add_documents(documents)
        # It's a good practice to explicitly persist if not using auto-persist client
        # Although Chroma generally persists when adding, explicit save can be safer
        self.vector_db.persist()
        print("Documents added and persisted to vector DB.")

    def query_vector_db(self, query: str, k: int = 5) -> List[Document]:
        """Performs a similarity search in the vector database."""
        if not hasattr(self.vector_db, 'similarity_search'):
            print("Vector DB not initialized for similarity search. Please ensure documents are added.")
            return []
        return self.vector_db.similarity_search(query, k=k)

    def reset_db(self):
        """Removes the existing vector database and re-initializes it."""
        if os.path.exists(self.db_path):
            try:
                shutil.rmtree(self.db_path)
                print(f"Removed existing Chroma DB at {self.db_path}")
            except OSError as e:
                print(f"Error removing Chroma DB directory {self.db_path}: {e}")
                print("Please ensure no other processes are accessing the directory and try again.")
                # You might want to raise the error or handle it more robustly
                return
        # After removal, re-initialize the self.vector_db property to a fresh, empty state
        self.vector_db = self._get_or_create_db()
        print("Chroma DB reset complete.")
        print("✅ Answer retrieved from: Vector DB")
