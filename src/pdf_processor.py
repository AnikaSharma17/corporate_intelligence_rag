import os
import pypdfium2 as pdfium
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        try:
            doc = pdfium.PdfDocument(pdf_path)
            text_content = ""
            for i in range(len(doc)):
                page = doc[i]
                text_page = page.get_textpage()
                text_content += text_page.get_text_range() + "\n"
            doc.close()
            return text_content
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def load_and_split_pdfs(self, directory: str) -> list[Document]:
        """Loads PDFs from a directory and splits them into LangChain Documents."""
        documents = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(directory, filename)
                print(f"Processing {pdf_path}...")
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    # Add metadata to the document chunks
                    metadata = {"source": pdf_path, "filename": filename}
                    chunks = self.text_splitter.create_documents([text], metadatas=[metadata])
                    documents.extend(chunks)
        print(f"Extracted and split {len(documents)} chunks from PDFs.")
        return documents

# Example usage (for testing)
if __name__ == "__main__":
    # Create a dummy PDF for testing
    with open("data/raw_pdfs/test_doc.pdf", "w") as f:
        f.write("This is a test document. It contains some information about a company's revenue in 2023. John Doe is the CEO.")

    processor = PDFProcessor()
    docs = processor.load_and_split_pdfs("data/raw_pdfs")
    for doc in docs[:2]: # Print first two chunks
        print(f"--- Chunk from {doc.metadata.get('filename')} ---")
        print(doc.page_content)