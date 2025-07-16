from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import LLM_MODEL_NAME, GOOGLE_API_KEY
import json

# --- Entity Schemas ---
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    title: str = Field(description="Title or role of the person within the company")

class Company(BaseModel):
    name: str = Field(description="Name of the company")
    ticker: str = Field(description="Stock ticker symbol (if applicable), 'N/A' if not found")
    industry: str = Field(description="Industry of the company, 'N/A' if not specified")

class FinancialFigure(BaseModel):
    metric: str = Field(description="Financial metric (e.g., 'Revenue', 'Net Income', 'Assets')")
    value: Union[float, str] = Field(description="Numeric value of the financial figure")
    unit: str = Field(description="Unit of the value (e.g., 'million USD', 'billion EUR', '%')")
    year: Union[int, str] = Field(description="Fiscal year or period the figure relates to")

class Project(BaseModel):
    name: str = Field(description="Name of the project")
    description: str = Field(description="Brief description of the project")

class Deadline(BaseModel):
    date: str = Field(description="Deadline date (e.g., 'July 20th, 2025')")
    context: str = Field(description="Text snippet from which the deadline was extracted")

class Relationship(BaseModel):
    source_entity_name: str = Field(description="Name of the source entity")
    source_entity_type: str = Field(description="Type of the source entity")
    relationship_type: str = Field(description="Type of relationship")
    target_entity_name: str = Field(description="Name of the target entity")
    target_entity_type: str = Field(description="Type of the target entity")
    context: str = Field(description="Text context")

class ExtractionResult(BaseModel):
    persons: List[Person] = Field(default_factory=list)
    companies: List[Company] = Field(default_factory=list)
    financial_figures: List[FinancialFigure] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    deadlines: List[Deadline] = Field(default_factory=list)


# --- Main Extraction Class ---
class EntityExtractor:
    def __init__(self, llm_model_name: str = LLM_MODEL_NAME):
        if "gemini" in llm_model_name.lower() and GOOGLE_API_KEY:
            self.llm = ChatGoogleGenerativeAI(model=llm_model_name, google_api_key=GOOGLE_API_KEY)
        else:
            raise ValueError("Unsupported LLM model or missing API key")

        self.parser = JsonOutputParser(pydantic_object=ExtractionResult)

        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """
            You are an expert at extracting structured information from business text.
            Extract key entities such as:
              - People (names, roles)
              - Companies (name, ticker, industry)
              - Financial figures (metric, value, unit, year)
              - Projects (project name, description)
              - Deadlines (date and associated context)
              - Relationships between entities

            Output a JSON matching this schema:
            {format_instructions}

            Use 'N/A' if specific details like ticker or industry are unavailable.
            If a deadline is given like "submit by July 20th", extract the date and quote that line as context.
            """),
            ("human", "Extract information from the following text:\n{text}")
        ])

        self.extraction_chain = self.extraction_prompt | self.llm | self.parser

    def extract_entities_and_relationships(self, text: str) -> ExtractionResult:
        try:
            format_instructions = self.parser.get_format_instructions()
            result = self.extraction_chain.invoke({
                "text": text,
                "format_instructions": format_instructions
            })
            return ExtractionResult(**result)
        except Exception as e:
            print(f"Extraction error: {e}")
            return ExtractionResult()


# --- Testing Example ---
if __name__ == "__main__":
    extractor = EntityExtractor()
    test_text = """
    You must submit your Nebula9.ai internship project by Saturday, July 20th.
    The project titled "AI for Social Impact" involves building a real-time dashboard.
    Apple Inc. (AAPL) had a revenue of $383.29 billion in 2023.
    Tim Cook is the CEO of Apple.
    Google (GOOGL) had a net income of $73.8 billion in 2023.
    """

    extracted = extractor.extract_entities_and_relationships(test_text)

    print("\n--- Companies ---")
    for c in extracted.companies:
        print(c)

    print("\n--- People ---")
    for p in extracted.persons:
        print(p)

    print("\n--- Financial Figures ---")
    for f in extracted.financial_figures:
        print(f)

    print("\n--- Projects ---")
    for proj in extracted.projects:
        print(proj)

    print("\n--- Deadlines ---")
    for d in extracted.deadlines:
        print(d)

    print("\n--- Relationships ---")
    for r in extracted.relationships:
        print(r)

    print("\n--- Full JSON ---")
    print(json.dumps(extracted.model_dump(), indent=2))