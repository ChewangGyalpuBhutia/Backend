from typing import Optional, Dict, List
import torch
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from logging import getLogger, INFO

class LangChain:
    def __init__(self, debug: bool = False):
        """
        Initialize LangChain with local models.
        
        Args:
            debug (bool): Enable debug logging if True
        """
        self.logger = getLogger(__name__)
        if debug:
            self.logger.setLevel(INFO)
            
        try:
            # Initialize embedding model using SentenceTransformers
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            # Initialize the T5 model and tokenizer for question answering
            model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to appropriate device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            
            # Create question-answering pipeline with proper error handling
            self.qa_pipeline = self._create_qa_pipeline()
            
            # Initialize LLM with explicit configurations
            self.llm = HuggingFacePipeline(
                pipeline=self.qa_pipeline,
                model_kwargs={"temperature": 0.7}
            )
            
            # Initialize text splitter with safety checks
            self.text_splitter = self._initialize_text_splitter()
            
            # Store document vectors in memory with type checking
            self._document_stores: Dict[int, FAISS] = {}
            
            self.logger.info("LangChain initialization successful")
            
        except Exception as e:
            self.logger.error(f"Error initializing LangChain: {str(e)}")
            raise

    def _create_qa_pipeline(self):
        """Create and configure the question-answering pipeline with error handling."""
        try:
            return pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                device=0 if self.device == "cuda" else -1,
                framework="pt",
                model_kwargs={"cache_dir": None}
            )
        except Exception as e:
            self.logger.error(f"Failed to create QA pipeline: {str(e)}")
            raise

    def _initialize_text_splitter(self):
        """Initialize the text splitter with validation."""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Validate splitter with a test string
            test_split = splitter.split_text("Test document content.")
            if not isinstance(test_split, list):
                raise ValueError("Text splitter initialization failed validation")
                
            return splitter
            
        except Exception as e:
            self.logger.error(f"Failed to initialize text splitter: {str(e)}")
            raise

    def process_document(self, content: str, doc_id: int) -> bool:
        """
        Process a document and store its vectors.
        
        Args:
            content (str): Document content to process
            doc_id (int): Unique identifier for the document
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        if not content or not isinstance(content, str):
            self.logger.error("Invalid document content provided")
            return False
            
        try:
            # Split the document into chunks with validation
            texts = self.text_splitter.split_text(content)
            if not texts:
                self.logger.warning("Document produced no chunks after splitting")
                return False
                
            # Create vector store with progress logging
            self.logger.info(f"Creating vector store for document {doc_id}")
            vectorstore = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"source": f"chunk_{i}", "doc_id": doc_id} for i in range(len(texts))]
            )
            
            # Store the vectorstore with atomic operation
            self._document_stores[doc_id] = vectorstore
            self.logger.info(f"Successfully processed document {doc_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing document {doc_id}: {str(e)}")
            return False

    def ask_question(self, content: str, question: str) -> str:
        """
        Answer a question based on the document content.
        
        Args:
            content (str): Document content to search
            question (str): Question to answer
            
        Returns:
            str: Generated answer or error message
        """
        if not question or not isinstance(question, str):
            return "Please provide a valid question."
            
        try:
            # Process the document if needed
            doc_id = hash(content)
            if doc_id not in self._document_stores:
                self.logger.info(f"Processing new document with id {doc_id}")
                success = self.process_document(content, doc_id)
                if not success:
                    return "Sorry, I couldn't process the document properly."
            
            # Get relevant context with validation
            relevant_chunks = self.get_relevant_context(content, question)
            if not relevant_chunks:
                return "I couldn't find relevant information in the document."
            
            # Prepare prompt with context length validation
            context = " ".join(relevant_chunks)
            if len(context) > 1000:  # Arbitrary limit for safety
                context = context[:1000] + "..."
                
            prompt = self._create_qa_prompt(context, question)
            
            # Generate answer with error handling
            response = self.qa_pipeline(
                prompt,
                max_length=512,
                min_length=50,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3
            )
            
            answer = response[0]['generated_text'].strip()
            
            # Validate and clean answer
            if not answer or len(answer) < 10:  # Arbitrary minimum length
                return "I couldn't generate a relevant answer."
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while trying to answer your question."

    def _create_qa_prompt(self, context: str, question: str) -> str:
        """Create a formatted prompt for the QA model."""
        return f"""
        Context: {context}
        
        Question: {question}
        
        Answer the question based on the context provided. If the answer cannot be found in the context, say "I cannot answer this based on the given context."
        
        Answer:
        """

    def get_relevant_context(self, content: str, question: str, num_chunks: int = 2) -> List[str]:
        """
        Get the most relevant context chunks for a question.
        
        Args:
            content (str): Document content to search
            question (str): Question to find context for
            num_chunks (int): Number of chunks to retrieve
            
        Returns:
            List[str]: List of relevant context chunks
        """
        try:
            doc_id = hash(content)
            if doc_id not in self._document_stores:
                self.logger.info(f"Processing new document for context retrieval: {doc_id}")
                success = self.process_document(content, doc_id)
                if not success:
                    return []
            
            # Get relevant documents with validation
            docs = self._document_stores[doc_id].similarity_search(
                question,
                k=min(num_chunks, 5)  # Limit maximum chunks for safety
            )
            
            return [doc.page_content for doc in docs]
            
        except Exception as e:
            self.logger.error(f"Error getting context: {str(e)}")
            return []

    def clear_document(self, doc_id: int) -> bool:
        """
        Clear a document's vectors from memory.
        
        Args:
            doc_id (int): ID of document to clear
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if doc_id in self._document_stores:
                del self._document_stores[doc_id]
                self.logger.info(f"Successfully cleared document {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing document {doc_id}: {str(e)}")
            return False