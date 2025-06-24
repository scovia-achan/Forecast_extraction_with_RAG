from chroma_manager import ChromaDBManager, DocumentQueryEngine
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from prompt_template import *
from dotenv import load_dotenv

def setup_query_engine():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize models (same config as processing)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    
    # Load existing vector store
    chroma_manager = ChromaDBManager("./chroma_db")
    vectorstore = chroma_manager.load_existing_vectorstore(embeddings)

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=prompt_template
    )
    
    # Create query engine
    query_engine = DocumentQueryEngine(vectorstore, llm, prompt)
    return query_engine

def main():
    query_engine = setup_query_engine()
    
    # Now you can query multiple times
    questions = [
        
        "In the text, what predictions have been made about Africa's digital transformation on Africa or connectivity and internet penetration in the region?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = query_engine.ask(
            topic="internet penetration or digital Transformation",
            place="Africa",
            year="2010 to 2025"
        )
        print(f"A: {answer}")

if __name__ == "__main__":
    main()