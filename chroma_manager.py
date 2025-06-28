from langchain.vectorstores import Chroma
import logging
from typing import List, Optional, Dict
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


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
        embedding_model: OpenAIEmbeddings,
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

    def load_existing_vectorstore(
        self, 
        embedding_model: OpenAIEmbeddings,
        collection_name: str = "document_chunks"
    ) -> Chroma:
        """Load an existing Chroma vector store without processing new documents"""
        try:
            logger.info(f"Loading existing Chroma vector store from: {self.persist_directory}")
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )
            logger.info("Existing Chroma vector store loaded successfully.")
            return self.db
        except Exception as e:
            logger.error(f"Failed to load existing Chroma vector store: {str(e)}")
            raise
    
class DocumentQueryEngine:
    """Handles querying the vector store using LangChain prompts and LLM"""

    def __init__(self,
                 vectorstore: Chroma,
                 llm: ChatOpenAI,
                 prompt: PromptTemplate
                 ):
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
            "k": 30,           # Get more chunks
            "lambda_mult": 0.5  # More diversity
    }
        )
        self.llm = llm
        # self.prompt_template = prompt_template
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",  # use basic "stuff" chain; can replace with "map_reduce", etc.
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, topic: str, place: str, year: str) -> str:
        try:
            formatted_question = f"What were the main trends in {topic} in {place} in {year}?"
            docs = self.retriever.invoke(formatted_question)
            logger.info("Retrieved Docs:\n" + "\n\n".join([d.page_content[:300] for d in docs]))
            print("---------------------------------------------------------------------------------------------------------")
            print()
            return self.qa_chain.invoke(formatted_question)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return "An error occurred during querying."



