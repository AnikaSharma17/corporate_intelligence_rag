from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "Anika2005")  # ← Only if this is your DB password

try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        result = session.run("RETURN 1")
        print("✅ Connection successful:", result.single())
except Exception as e:
    print("❌ Connection failed:", e)

import google.generativeai as genai

genai.configure(api_key="AIzaSyAHQ5TkQVO7-DE-Hf9PWdiNPLjACL7pEu0")
for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
