from chroma_manager import ChromaDBManager, DocumentQueryEngine
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from prompt_template import *
from info_extractors import *
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

     # NEW: Add projection extraction
    print("\n" + "="*60)
    print("🎯 EXTRACTING STRUCTURED PROJECTIONS")
    print("="*60)
    
    # Create projection extractor using the same query engine
    extractor = ProjectionExtractor(query_engine)
    
    # Try different extraction methods
    print("🔍 Method 1: Comprehensive search...")
    projections1 = extractor.extract_projections()
    
    print(f"\n🔍 Method 2: Document-by-document analysis...")
    projections2 = extractor.extract_by_document()
    
    # Combine and deduplicate
    all_projections = projections1 + projections2
    final_projections = extractor._deduplicate(all_projections)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Method 1 found: {len(projections1)} projections")
    print(f"Method 2 found: {len(projections2)} projections") 
    print(f"Combined unique: {len(final_projections)} projections")
    
    # Save combined results
    extractor.save_to_csv(final_projections, "comprehensive_projections.csv")
    
    # Display a few examples
    if final_projections:
        print(f"\n📋 Sample projections found:")
        for i, proj in enumerate(final_projections[:5]):  # Show top 5
            print(f"\n{i+1}. {proj.get('projection', 'N/A')}")
            print(f"   📍 Entity: {proj.get('entity', 'N/A')}")
            print(f"   📅 Year Predicted: {proj.get('year_predicted', 'N/A')}")
            print(f"   🎯 Target Year: {proj.get('target_year', 'N/A')}")
            print(f"   📄 Source: {proj.get('source_file', 'N/A')}")
    else:
        print("No projections found. Check if documents contain prediction language.")


if __name__ == "__main__":
    main()