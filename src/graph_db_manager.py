from neo4j import GraphDatabase
from typing import List, Dict, Any
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
from src.entity_extractor import (
    ExtractionResult, Person, Company, FinancialFigure, Relationship
)

class GraphDBManager:
    def __init__(self, uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self._test_connection()

    def _test_connection(self):
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print("✅ Successfully connected to Neo4j.")
        except Exception as e:
            print(f"❌ Could not connect to Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def create_schema_constraints(self):
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE CONSTRAINT company_name_unique IF NOT EXISTS 
                FOR (c:Company) REQUIRE c.name IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT person_name_unique IF NOT EXISTS 
                FOR (p:Person) REQUIRE p.name IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT financial_figure_unique IF NOT EXISTS 
                FOR (f:FinancialFigure) REQUIRE (f.metric, f.year, f.company_name) IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT project_title_unique IF NOT EXISTS 
                FOR (p:Project) REQUIRE p.title IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT deadline_date_unique IF NOT EXISTS 
                FOR (d:Deadline) REQUIRE d.date IS UNIQUE
            """)
            print("✅ Schema constraints created.")

    def ingest_extracted_data(self, data: ExtractionResult):
        """Ingests extracted entities and relationships into Neo4j."""
        with self.driver.session(database=self.database) as session:
            tx = session.begin_transaction()
            print(f"📦 Ingesting {len(data.companies)} companies, {len(data.persons)} persons, {len(data.financial_figures)} financial figures, {len(data.relationships)} relationships...")

            # Companies
            for company in data.companies:
                tx.run("""
                    MERGE (c:Company {name: $name})
                    SET c.ticker = $ticker, c.industry = $industry
                """, name=company.name, ticker=company.ticker, industry=company.industry)

            # Persons
            for person in data.persons:
                tx.run("""
                    MERGE (p:Person {name: $name})
                    SET p.title = $title
                """, name=person.name, title=person.title)

            # Financial Figures
            for ff in data.financial_figures:
                tx.run("""
                    MERGE (f:FinancialFigure {
                        metric: $metric, year: $year, company_name: $company_name
                    })
                    SET f.value = $value, f.unit = $unit
                """, metric=ff.metric, year=ff.year, value=ff.value, unit=ff.unit,
                     company_name=getattr(ff, "company_name", "N/A"))

            # Projects
            for proj in getattr(data, "projects", []):
                tx.run("""
                       MERGE (p:Project {name: $name})
                    """, name=proj.name)


            # Deadlines
            for dl in getattr(data, "deadlines", []):
                tx.run("""
                    MERGE (d:Deadline {date: $date})
                    SET d.context = $context
                """, date=dl.date, context=dl.context)

            # Relationships
            for rel in data.relationships:
                try:
                    if rel.source_entity_type == "Company" and rel.target_entity_type == "Person" and rel.relationship_type == "HAS_CEO":
                        tx.run(f"""
                            MATCH (c:Company {{name: $source_name}})
                            MATCH (p:Person {{name: $target_name}})
                            MERGE (c)-[:{rel.relationship_type}]->(p)
                        """, source_name=rel.source_entity_name, target_name=rel.target_entity_name)

                    elif rel.source_entity_type == "Company" and rel.target_entity_type == "FinancialFigure" and rel.relationship_type == "REPORTS_REVENUE":
                        tx.run(f"""
                            MATCH (c:Company {{name: $source_name}})
                            MATCH (f:FinancialFigure {{
                                metric: 'Revenue', value: $value, unit: $unit, year: $year
                            }})
                            MERGE (c)-[:{rel.relationship_type}]->(f)
                        """, source_name=rel.source_entity_name, value=rel.target_entity_name,
                             unit=rel.target_entity_type, year=rel.context)

                    elif rel.source_entity_type == "Company" and rel.target_entity_type == "Company" and rel.relationship_type == "IS_COMPETITOR_OF":
                        tx.run(f"""
                            MATCH (c1:Company {{name: $source_name}})
                            MATCH (c2:Company {{name: $target_name}})
                            MERGE (c1)-[:{rel.relationship_type}]->(c2)
                        """, source_name=rel.source_entity_name, target_name=rel.target_entity_name)
                    else:
                        print(f"⚠️ Skipping unsupported relationship: {rel}")
                except Exception as rel_err:
                    print(f"❌ Error ingesting relationship {rel}: {rel_err}")

            tx.commit()
            print("✅ Data ingestion to Neo4j complete.")

    def query_graph_db(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Executes a Cypher query against the graph database."""
        print(f"📤 Executing Cypher query: {cypher_query}")
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

    def clear_graph_db(self):
        """Deletes all nodes and relationships from the database."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🧨 Cleared all data from Neo4j.")

# Example usage
if __name__ == "__main__":
    db_manager = GraphDBManager()
    db_manager.clear_graph_db()
    db_manager.create_schema_constraints()

    from src.entity_extractor import EntityExtractor
    extractor = EntityExtractor()
    test_text = """
    Apple Inc. (AAPL) reported revenue of $383.29 billion in fiscal year 2023.
    Tim Cook is the current CEO of Apple.
    Microsoft is a major competitor of Apple.
    The final project deadline is July 20th, 2025.
    Project title: HayDay — an agriculture chatbot.
    """
    extracted_data = extractor.extract_entities_and_relationships(test_text)

    # Link company names to Financial Figures manually (if needed)
    for ff in extracted_data.financial_figures:
        if ff.metric.lower() in ["revenue", "net income"]:
            ff.company_name = "Apple Inc."

    db_manager.ingest_extracted_data(extracted_data)

    result = db_manager.query_graph_db("MATCH (c:Company)-[:HAS_CEO]->(p:Person) RETURN c.name, p.name")
    print("\n🧑‍💼 CEOs:\n", result)

    result2 = db_manager.query_graph_db("MATCH (p:Project) RETURN p.title, p.description")
    print("\n📚 Projects:\n", result2)

    db_manager.close()
