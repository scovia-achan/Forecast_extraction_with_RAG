from pdf_processing import *
from chroma_manager import *

def process_and_store_documents():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    PDF_FOLDER = "./data/pdfs"
    
    # Your existing configuration
    chunking_config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
    model_config = ModelConfig(
        model_name="gemini-1.5-flash",
        temperature=0.1,
        embedding_model="models/text-embedding-004"
    )
    
    # Process documents
    pipeline = DocumentProcessingPipeline(
        pdf_folder_path=PDF_FOLDER,
        google_api_key=api_key,
        chunking_config=chunking_config,
        model_config=model_config
    )
    
    results = pipeline.run_pipeline()
    print(f"Processing complete! Stored {results['processed_chunks_count']} chunks")
    print(f"Connection test: {'Passed' if results['connection_status'] else 'Failed'}")
    print(f"Chroma vectorstore loaded: {results['chroma_vectorstore'] is not None}")
    return results

if __name__ == "__main__":
    process_and_store_documents()