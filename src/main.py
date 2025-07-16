import os
import wandb
from src.pdf_processor import PDFProcessor
from src.entity_extractor import EntityExtractor
from src.vector_db_manager import VectorDBManager
from src.graph_db_manager import GraphDBManager
from src.agent_core import RAGAgent

# Import configuration parameters
from src.config import (
    PDF_DIRECTORY,
    CHUNK_SIZE,       # Although used in PDFProcessor, good to keep in config
    CHUNK_OVERLAP,    # Although used in PDFProcessor, good to keep in config
    LLM_MODEL_NAME,   # Used for WandB logging
    NEO4J_URI,        # Used for connection in setup_databases
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    EMBEDDING_MODEL_NAME
)

def setup_databases():
    """Initializes and connects to databases."""
    print("Setting up Vector Database Manager...")
    vector_db_manager = VectorDBManager()
    print("Setting up Graph Database Manager...")
    graph_db_manager = GraphDBManager(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    return vector_db_manager, graph_db_manager

def ingest_documents_pipeline(vector_db_manager: VectorDBManager, graph_db_manager: GraphDBManager):
    """
    Runs the full ingestion pipeline: PDF -> Extract -> Store in Vector DB & Graph DB.
    """
    print(f"\n--- Starting Data Ingestion from {PDF_DIRECTORY} ---")

    # 1. Clear existing data (for fresh runs or testing)
    print("Clearing existing Vector DB...")
    vector_db_manager.reset_db()
    print("Clearing existing Graph DB...")
    graph_db_manager.clear_graph_db()
    # Create schema constraints for better performance and data integrity
    graph_db_manager.create_schema_constraints()

    # 2. Process PDFs and get text chunks
    pdf_processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = pdf_processor.load_and_split_pdfs(PDF_DIRECTORY)
    if not documents:
        print(f"No PDF documents found or processed in {PDF_DIRECTORY}. Exiting ingestion.")
        # Log to WandB even if no documents found
        wandb.log({
            "ingestion_stats/num_pdfs_processed": len([f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]),
            "ingestion_stats/num_text_chunks": 0,
            "ingestion_stats/total_chars_processed": 0
        })
        return

    # Log document/chunk stats to WandB
    wandb.log({
        "ingestion_stats/num_pdfs_processed": len([f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]),
        "ingestion_stats/num_text_chunks": len(documents),
        "ingestion_stats/total_chars_processed": sum(len(d.page_content) for d in documents)
    })

    # 3. Add chunks to Vector Database
    vector_db_manager.add_documents(documents)

    # 4. Extract entities and relationships from chunks and ingest into Graph DB
    entity_extractor = EntityExtractor(llm_model_name=LLM_MODEL_NAME) # Pass LLM model name
    print("\nStarting entity and relationship extraction for graph database...")

    total_companies = 0
    total_persons = 0
    total_financial_figures = 0
    total_relationships = 0

    for i, doc in enumerate(documents):
        print(f"Extracting from chunk {i+1}/{len(documents)} (Source: {doc.metadata.get('filename')})...")
        extracted_data = entity_extractor.extract_entities_and_relationships(doc.page_content)

        # --- IMPORTANT: Manual adjustment for linking FinancialFigure to Company ---
        # This is a critical step because the LLM might extract a FinancialFigure
        # and a Company, but not directly link them with a 'company_name' property
        # on the FinancialFigure Pydantic model. This loop attempts to infer
        # the company based on relationships, for ingestion into Neo4j.
        # In a production system, you'd strive for direct extraction or more
        # robust post-processing.
        for rel in extracted_data.relationships:
            if rel.relationship_type in ["REPORTS_REVENUE", "REPORTS_NET_INCOME", "HAS_FINANCIAL_FIGURE"]:
                # Find the FinancialFigure object that matches the target of this relationship
                for ff_idx, ff in enumerate(extracted_data.financial_figures):
                    # This check is a heuristic;
                    # improved logic might be needed based on actual LLM output patterns
                    if str(ff.value) == rel.target_entity_name and str(ff.year) == rel.context: # Heuristic: Context often contains year
                        ff.company_name = rel.source_entity_name
                        # Update the specific financial figure object
                        extracted_data.financial_figures[ff_idx] = ff
                        break
        # --- End of manual adjustment ---


        graph_db_manager.ingest_extracted_data(extracted_data)
        print(f"Processed chunk {i+1}. Extracted: Companies={len(extracted_data.companies)}, Persons={len(extracted_data.persons)}, FinancialFigures={len(extracted_data.financial_figures)}, Relationships={len(extracted_data.relationships)}")

        total_companies += len(extracted_data.companies)
        total_persons += len(extracted_data.persons)
        total_financial_figures += len(extracted_data.financial_figures)
        total_relationships += len(extracted_data.relationships)

    # Log total extraction stats to WandB
    wandb.log({
        "extraction_stats/total_companies_extracted": total_companies,
        "extraction_stats/total_persons_extracted": total_persons,
        "extraction_stats/total_financial_figures_extracted": total_financial_figures,
        "extraction_stats/total_relationships_extracted": total_relationships
    })

    print("\n--- Data Ingestion Complete ---")
    graph_db_manager.close() # Close graph DB connection after ingestion

def run_qa_system():
    """Runs the interactive Q&A system."""
    print("\n--- Starting Q&A System ---")
    rag_agent = RAGAgent()

    print("\nReady! Ask your corporate intelligence questions (type 'exit' to quit).")
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'exit':
            break
        try:
            # Log the user query to WandB
            if wandb.run:
                wandb.log({"qa_session/user_query": user_query})

            answer = rag_agent.query(user_query)
            print(f"\nAI Answer: {answer}")

            # Log the AI's answer to WandB
            if wandb.run:
                wandb.log({"qa_session/ai_answer": answer})

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your LLM API keys are correct and databases are running.")
            if wandb.run:
                wandb.log({"qa_session/error": str(e)}) # Log errors to WandB

    print("\n--- Q&A System Ended ---")


if __name__ == "__main__":
    # Initialize WandB run at the very beginning
    # The 'config' dictionary allows you to log hyperparameters and settings
    wandb.init(
        project="corporate-intelligence-rag", # Name of your project in WandB
        config={
            "llm_model": LLM_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "pdf_directory": PDF_DIRECTORY,
            "neo4j_uri": NEO4J_URI,
            "neo4j_database": NEO4J_DATABASE,
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
    )

    # Ensure the data directory exists
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    # You might want to copy some dummy PDFs into data/raw_pdfs for initial testing.

    vector_db_manager, graph_db_manager = setup_databases()

    # Phase 1: Ingest documents
    ingest_documents_pipeline(vector_db_manager, graph_db_manager)

    # Phase 2: Run Q&A system
    run_qa_system()

    # End the WandB run
    wandb.finish()