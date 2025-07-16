# Corporate Intelligence RAG System

This project implements a Retrieval-Augmented Generation (RAG) system designed for analyzing corporate documents and answering user queries using both unstructured (PDFs) and structured (graph database) data. It combines:

- **Vector database (ChromaDB)** for semantic search from document text.
- **Graph database (Neo4j)** for factual, structured queries.
- **LLM (Gemini / Google Generative AI)** for intelligent orchestration, fallback responses, and entity extraction.

The system supports ingestion of PDFs, extraction of entities and relationships, and natural language Q&A across semantic and factual layers.

---

## Features

- Entity and relationship extraction from PDF reports using LLMs  
- Knowledge graph population using Neo4j  
- Vector database powered semantic retrieval using Chroma  
- Gemini-based agent that routes questions through Graph, Vector, and LLM fallback  
- Structured entity types: `Company`, `Person`, `FinancialFigure`, `Project`, `Deadline`, `Assignment`

---

### Example Questions You Can Ask the System

These sample queries demonstrate how users can interact with the RAG system once data has been ingested:

- What is the submission deadline for the internship project?
- What projects are mentioned in the internship document?
- Which language should be used to build the system?
- Can I use LangChain in my solution?
- In how many days should I submit the project?

---


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AnikaSharma17/corporate_intelligence_rag.git
cd corporate_intelligence_rag
```

### 2. Set Up the Environment

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
# or
source venv/bin/activate  # For Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add the following content:

```env
GOOGLE_API_KEY=your_google_api_key
LLM_MODEL_NAME=gemini-pro
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=./vector_db
```

### 5. Run the System

```bash
python -m src.main
```

---

## Directory Structure

```bash
corporate_intelligence_rag/
├── data/raw_pdfs/              # Input PDF files
├── vector_db/                  # Vector DB (Chroma) persistence
├── wandb/                      # W&B logging directory
├── src/
│   ├── agent_core.py           # RAG Agent implementation
│   ├── config.py               # Configuration management
│   ├── entity_extractor.py     # LLM-based entity/relationship extraction
│   ├── graph_db_manager.py     # Neo4j integration and ingestion
│   ├── vector_db_manager.py    # Chroma integration
│   ├── pdf_processor.py        # PDF text extraction and chunking
│   └── main.py                 # Orchestrates ingestion + Q&A system
```

---

## Challenges Faced

- **Entity-Relationship Mapping**: Mapping text to structured entities required refining prompt design and schema validation using Pydantic. Many unsupported or vague relationships had to be skipped or rephrased.
- **Redundancy in Vector Responses**: Initially, vector search returned repeated text chunks. This was resolved using a de-duplication step before returning results.
- **Graph Schema Limitations**: Some relationship types (e.g., “uses”, “offers API for”) didn't fit into the initial graph model. These were either handled manually or flagged for extension.
- **Dependency Conflicts**: LangChain updates and deprecations (e.g., Chroma/HuggingFace modules) required pinning versions and updating imports accordingly.

---

## Assumptions Made

- Input PDFs are business/corporate documents with sufficient structure for extraction.
- Company names, project titles, and relationships can be reliably detected with Gemini prompts.
- Graph schema can be expanded if needed, but unrecognized relationships will be skipped.
- The system is not intended for real-time or large-scale ingestion but works well for curated corpora.
