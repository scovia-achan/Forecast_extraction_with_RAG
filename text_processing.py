import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import PyPDF2
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from text_processing import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for text chunking parameters"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]

@dataclass
class ModelConfig:
    """Configuration for Gemini model parameters"""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    embedding_model: str = "models/text-embedding-004"

class PDFDocumentLoader:
    """Handles loading and extracting text from PDF documents"""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    def load_documents(self) -> List[Document]:
        """Load all PDF documents from the specified folder"""
        documents = []
        pdf_files = list(self.folder_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.folder_path}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                doc_content = self._extract_text_from_pdf(pdf_file)
                if doc_content.strip():  # Only add non-empty documents
                    document = Document(
                        page_content=doc_content,
                        metadata={
                            "source": str(pdf_file),
                            "filename": pdf_file.name,
                            "file_size": pdf_file.stat().st_size
                        }
                    )
                    documents.append(document)
                    logger.info(f"Successfully loaded: {pdf_file.name}")
                else:
                    logger.warning(f"No text content found in: {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a single PDF file"""
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Add page separator for better chunking
                        text_content.append(f"\n--- Page {page_num + 1} ---\n")
                        text_content.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1} from {pdf_path.name}: {str(e)}")
                    continue
        
        return "\n".join(text_content)

class DocumentPreprocessor:
    """Handles preprocessing and chunking of documents"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess and chunk documents"""
        if not documents:
            logger.warning("No documents provided for preprocessing")
            return []
        
        logger.info(f"Preprocessing {len(documents)} documents")
        
        # Clean documents
        cleaned_documents = self._clean_documents(documents)
        
        # Split documents into chunks
        chunked_documents = self.text_splitter.split_documents(cleaned_documents)
        
        # Add chunk metadata
        processed_chunks = self._add_chunk_metadata(chunked_documents)
        
        logger.info(f"Created {len(processed_chunks)} chunks from {len(documents)} documents")
        return processed_chunks
    
    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean document content"""
        cleaned_docs = []
        
        for doc in documents:
            # Basic text cleaning
            cleaned_content = doc.page_content
            
            # Remove excessive whitespace
            cleaned_content = " ".join(cleaned_content.split())
            
            # Remove common PDF artifacts
            cleaned_content = self._remove_pdf_artifacts(cleaned_content)
            
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata.copy()
            )
            cleaned_docs.append(cleaned_doc)
        
        return cleaned_docs
    
    def _remove_pdf_artifacts(self, text: str) -> str:
        """Remove common PDF extraction artifacts"""
        # Remove page headers/footers patterns
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be artifacts
            if len(line) < 3:
                continue
            # Skip lines that are just numbers (page numbers)
            if line.isdigit():
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _add_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """Add additional metadata to chunks"""
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "word_count": len(chunk.page_content.split())
            })
        
        return chunks

class GeminiModelManager:
    """Manages initialization and configuration of Google Gemini models"""
    
    def __init__(self, api_key: str, config: ModelConfig = None):
        self.api_key = api_key
        self.config = config or ModelConfig()
        self._initialize_genai()
        self.llm = None
        self.embeddings = None
    
    def _initialize_genai(self):
        """Initialize the Google GenerativeAI client"""
        try:
            genai.configure(api_key=self.api_key)
            logger.info("Google GenerativeAI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Google GenerativeAI: {str(e)}")
            raise
    
    def initialize_llm(self) -> GoogleGenerativeAI:
        """Initialize the Gemini LLM model"""
        try:
            self.llm = GoogleGenerativeAI(
                model=self.config.model_name,
                google_api_key=self.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            logger.info(f"Gemini LLM initialized: {self.config.model_name}")
            return self.llm
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
            raise
    
    def initialize_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Initialize the Google embeddings model"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model,
                google_api_key=self.api_key
            )
            logger.info(f"Google embeddings initialized: {self.config.embedding_model}")
            return self.embeddings
        except Exception as e:
            logger.error(f"Failed to initialize Google embeddings: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test the connection to Google's services"""
        try:
            # Test LLM
            if self.llm:
                response = self.llm.invoke("Hello, this is a test.")
                logger.info("LLM connection test successful")
            
            # Test embeddings
            if self.embeddings:
                test_embedding = self.embeddings.embed_query("test query")
                logger.info(f"Embeddings connection test successful (dimension: {len(test_embedding)})")
            
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

class DocumentProcessingPipeline:
    """Main pipeline class that orchestrates the entire document processing workflow"""
    
    def __init__(self, 
                 pdf_folder_path: str,
                 google_api_key: str,
                 chunking_config: ChunkingConfig = None,
                 model_config: ModelConfig = None,
                 chroma_dir: str = "./chroma_db"):
        
        self.pdf_folder_path = pdf_folder_path
        self.google_api_key = google_api_key
        
        # Initialize components
        self.loader = PDFDocumentLoader(pdf_folder_path)
        self.preprocessor = DocumentPreprocessor(chunking_config)
        self.model_manager = GeminiModelManager(google_api_key, model_config)
        self.chromadb_manager = ChromaDBManager(chroma_dir)
        
        # Will store processed data
        self.raw_documents: List[Document] = []
        self.processed_chunks: List[Document] = []
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete document processing pipeline"""
        logger.info("Starting document processing pipeline")
        
        try:
            # Step 1: Load documents
            self.raw_documents = self.loader.load_documents()
            
            # Step 2: Preprocess and chunk documents
            self.processed_chunks = self.preprocessor.preprocess_documents(self.raw_documents)
            
            # Step 3: Initialize models
            llm = self.model_manager.initialize_llm()
            embeddings = self.model_manager.initialize_embeddings()
            
            # Step 4: Test connections
            connection_status = self.model_manager.test_connection()
            
            # Step 5: Store chunks in Chroma DB
            vectorstore = self.chromadb_manager.create_or_load_vectorstore(
                chunks=self.processed_chunks,
                embedding_model=embeddings
            )

            # Prepare results
            results = {
                "raw_documents_count": len(self.raw_documents),
                "processed_chunks_count": len(self.processed_chunks),
                "llm_model": llm,
                "embeddings_model": embeddings,
                "connection_status": connection_status,
                "chroma_vectorstore": vectorstore,
                "documents": self.raw_documents,
                "chunks": self.processed_chunks
            }
            
            logger.info("Document processing and vector storage completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Configuration
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    print("API Key:", api_key) 

    PDF_FOLDER = "./data/pdfs"
    GOOGLE_API_KEY = api_key
    
    # Custom configurations (optional)
    chunking_config = ChunkingConfig(
        chunk_size=1200,
        chunk_overlap=200
    )
    
    model_config = ModelConfig(
        model_name="gemini-1.5-flash",
        temperature=0.1,
        embedding_model="models/text-embedding-004"
    )
    
    # Initialize and run pipeline
    pipeline = DocumentProcessingPipeline(
        pdf_folder_path=PDF_FOLDER,
        google_api_key=GOOGLE_API_KEY,
        chunking_config=chunking_config,
        model_config=model_config
    )
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Access the results
    print(f"Processed {results['raw_documents_count']} documents into {results['processed_chunks_count']} chunks")
    print(f"Models initialized: LLM={results['llm_model'] is not None}, Embeddings={results['embeddings_model'] is not None}")
    print(f"Rmebedding model: {results['embeddings_model'].model}")
    print(f"Connection test: {'Passed' if results['connection_status'] else 'Failed'}")