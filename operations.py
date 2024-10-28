from typing import Optional, Dict, List
import torch
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class Operations:
    def __init__(self):
        """Initialize LangChain with local models."""
        try:
            # Initialize embedding model using SentenceTransformers
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, faster model
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            # Initialize the T5 model and tokenizer for question answering
            model_name = "google/flan-t5-small"  # Using smaller model for faster loading
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            # Create question-answering pipeline
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize LLM
            self.llm = HuggingFacePipeline(
                pipeline=self.qa_pipeline
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Store document vectors in memory
            self._document_stores: Dict[int, FAISS] = {}
            
        except Exception as e:
            print(f"Error initializing LangChain: {str(e)}")
            raise

    def process_document(self, content: str, doc_id: int) -> bool:
        """Process a document and store its vectors."""
        try:
            # Split the document into chunks
            texts = self.text_splitter.split_text(content)
            
            # Create vector store
            vectorstore = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"source": f"chunk_{i}"} for i in range(len(texts))]
            )
            
            # Store the vectorstore in memory
            self._document_stores[doc_id] = vectorstore
            
            return True
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return False

    def ask_question(self, content: str, question: str) -> str:
        """Answer a question based on the document content."""
        try:
            # Process the document if needed
            doc_id = hash(content)
            if doc_id not in self._document_stores:
                success = self.process_document(content, doc_id)
                if not success:
                    return "Sorry, I couldn't process the document properly."
            
            # Get relevant context
            relevant_chunks = self.get_relevant_context(content, question)
            if not relevant_chunks:
                return "I couldn't find relevant information in the document."
            
            # Prepare prompt with context
            context = " ".join(relevant_chunks)
            prompt = f"""
            Context: {context}
            
            Question: {question}
            
            Answer the question based on the context provided. If the answer cannot be found in the context, say "I cannot answer this based on the given context."
            
            Answer:
            """
            
            # Generate answer using the pipeline
            response = self.qa_pipeline(
                prompt,
                max_length=512,
                min_length=50,
                num_beams=4,
                do_sample=False
            )
            
            answer = response[0]['generated_text'].strip()
            
            # Clean up the answer
            if not answer:
                return "I couldn't generate a relevant answer."
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while trying to answer your question."

    def get_relevant_context(self, content: str, question: str, num_chunks: int = 2) -> List[str]:
        """Get the most relevant context chunks for a question."""
        try:
            doc_id = hash(content)
            if doc_id not in self._document_stores:
                success = self.process_document(content, doc_id)
                if not success:
                    return []
            
            # Get relevant documents
            docs = self._document_stores[doc_id].similarity_search(
                question,
                k=num_chunks
            )
            
            return [doc.page_content for doc in docs]
            
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return []

    def clear_document(self, doc_id: int) -> bool:
        """Clear a document's vectors from memory."""
        try:
            if doc_id in self._document_stores:
                del self._document_stores[doc_id]
            return True
        except Exception:
            return False