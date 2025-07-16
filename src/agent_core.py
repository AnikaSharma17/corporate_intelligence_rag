from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph

import wandb
from src.config import (
    LLM_MODEL_NAME,
    GOOGLE_API_KEY,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE
)
from src.vector_db_manager import VectorDBManager


class RAGAgent:
    def __init__(self):
        # Initialize Gemini LLM
        if "gemini" in LLM_MODEL_NAME.lower() and GOOGLE_API_KEY:
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
        else:
            raise ValueError("Unsupported LLM or missing API key")

        # Init vector and graph systems
        self.vector_db_manager = VectorDBManager()
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        self.tools = self._initialize_tools()
        self.agent_executor = self._initialize_agent()

    def _initialize_tools(self):
        @tool
        def query_vector_database(query: str) -> str:
            """Search the vector database for semantically relevant chunks from documents."""
            print(f"\n🔍 [VectorDB] Query: '{query}'")
            docs = self.vector_db_manager.query_vector_db(query)
            chunks = list(dict.fromkeys([doc.page_content.strip() for doc in docs]))  # Remove duplicates
            result = "\n\n".join(chunks)
            if wandb.run:
                wandb.log({
                    "tool_calls/vector_db_query": query,
                    "tool_calls/vector_db_result": result
                })
            return result or "[Vector DB] Sorry, that info isn’t in the PDF."

        @tool
        def query_graph_database(cypher_query: str) -> str:
            """Run a Cypher query on the Neo4j graph database to extract structured facts."""
            print(f"\n🧠 [GraphDB] Cypher: '{cypher_query}'")
            try:
                results = self.graph.query(cypher_query)
                result_str = str(results)
                if wandb.run:
                    wandb.log({
                        "tool_calls/graph_db_cypher": cypher_query,
                        "tool_calls/graph_db_result": result_str
                    })
                return result_str or "[Graph DB] No matching facts found."
            except Exception as e:
                error = f"[Graph DB Error] {e}"
                print(error)
                if wandb.run:
                    wandb.log({"tool_calls/graph_db_cypher_error": error})
                return error

        return [query_vector_database, query_graph_database]

    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a corporate intelligence analyst. Choose the correct tool for each question:
1. Use `query_graph_database` for factual, structured data like relationships, CEOs, deadlines.
2. Use `query_vector_database` for summaries or semantic understanding from documents.
3. If both tools fail, generate a helpful answer using Gemini LLM directly.

Avoid repeating the same chunk multiple times. Always try the graph first for fact-based queries.

If no relevant answer is found from either tool, say: "Sorry, that info isn’t available right now."
"""),
            HumanMessage(content="Here are your tools: {tools}"),
            HumanMessage(content="Graph schema: {graph_schema}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def query(self, user_query: str) -> str:
        if wandb.run:
            wandb.log({"agent_trace/user_query": user_query})

        try:
            response = self.agent_executor.invoke({
                "input": user_query,
                "tools": self.tools,
                "graph_schema": self.graph.schema
            })
            answer = response.get("output", "No answer generated.")
            print("\n📤 [ANSWER SOURCE] → Graph DB / Vector DB / LLM fallback")
            if wandb.run:
                wandb.log({"agent_trace/final_answer": answer})
            return answer

        except Exception as e:
            error = f"[Agent Error] {e}"
            print(error)
            if wandb.run:
                wandb.log({"agent_trace/error": error})
            return "Sorry, I couldn’t answer that right now."



# Example usage block (for direct testing of agent_core.py, though main.py is preferred)
# if __name__ == "__main__":
#     # This block requires your .env to be set up correctly and
#     # your databases to be populated for meaningful results.
#     # It's primarily for debugging the agent itself.
#     # For full workflow testing, use main.py.

#     # Dummy setup for testing if databases aren't fully populated:
#     # You'd need to manually populate vector_db and neo4j with test data
#     # to get useful responses here.

#     print("Initializing RAG Agent. Ensure Neo4j and Chroma DB are set up and populated.")
#     try:
#         rag_agent = RAGAgent()

#         queries = [
#             "What is the general business strategy of Apple?", # Semantic
#             "Who is the CEO of Apple Inc.?", # Factual (Graph)
#             "What was Google's net income in 2023?", # Factual (Graph)
#             "What are the main products of Microsoft?", # Semantic
#             "Tell me about the relationship between Apple and Microsoft.", # Hybrid (Graph for relation, Vector for context)
#             "List all companies and their CEOs that you know.", # Factual (Graph)
#             "Summarize the recent financial performance of Apple." # Hybrid
#         ]

#         for q in queries:
#             print(f"\n--- User Query: {q} ---")
#             answer = rag_agent.query(q)
#             print(f"--- Agent Answer: {answer} ---")
#             print("-" * 50)

#     except Exception as e:
#         print(f"Failed to initialize or run agent: {e}")
#         print("Please ensure your .env is correctly configured and databases are accessible.")