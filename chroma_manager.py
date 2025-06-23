from langchain.vectorstores import Chroma
import logging
from typing import List, Optional, Dict
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Handles adding processed chunks to Chroma vector store"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.db = None

    def create_or_load_vectorstore(
        self,
        chunks: List[Document],
        embedding_model: GoogleGenerativeAIEmbeddings,
        collection_name: str = "document_chunks",
    ) -> Chroma:
        """
        Create or load a Chroma vector store with the provided documents and embedding model.
        """
        try:
            logger.info(
                f"Creating/loading Chroma vector store at: {self.persist_directory}"
            )
            self.db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=self.persist_directory,
                collection_name=collection_name,
            )
            self.db.persist()
            logger.info("Chroma vector store initialized and persisted successfully.")
            return self.db
        except Exception as e:
            logger.error(f"Failed to initialize Chroma vector store: {str(e)}")
            raise

    def get_db(self) -> Chroma:
        """Access the vector store if already initialized"""
        if not self.db:
            raise ValueError("Chroma DB not initialized yet.")
        return self.db
