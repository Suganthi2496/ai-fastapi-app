from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from app.core.config import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import tempfile
import os
import time
import logging
import asyncio
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self._setup_llm()
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="""
            You are a helpful AI assistant. Answer the user's question based on the context and chat history.
            
            Chat History: {chat_history}
            User Question: {question}
            
            Your answer:
            """
        )
        
        self.qa_chain = self._create_qa_chain()
        
        # Track current API key index
        self.current_key_index = 0
        self.current_model = settings.DEFAULT_MODEL_NAME
        self.using_fallback = False
        
        # Store settings for reference
        self.settings = settings
    
    def _setup_llm(self):
        """Set up the language model with the current settings"""
        if not settings.OPENAI_API_KEYS:
            logger.error("No API keys configured. Please set OPENAI_API_KEY in your environment.")
            raise ValueError("OpenAI API key is missing")
        
        api_key = settings.OPENAI_API_KEYS[self.current_key_index] if hasattr(self, 'current_key_index') else settings.OPENAI_API_KEYS[0]
        model = self.current_model if hasattr(self, 'current_model') else settings.DEFAULT_MODEL_NAME
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name=model,
            temperature=0.7
        )
    
    def _create_qa_chain(self):
        """Create QA chain with the current LLM"""
        return LLMChain(
            llm=self.llm,
            prompt=self.qa_prompt,
            memory=self.memory,
            verbose=True
        )
    
    def _try_next_api_key(self):
        """Try the next available API key"""
        # If we've already tried all keys
        if self.current_key_index >= len(settings.OPENAI_API_KEYS) - 1:
            # If we're using the primary model, switch to fallback model and reset key index
            if self.current_model == settings.DEFAULT_MODEL_NAME:
                self.current_model = settings.FALLBACK_MODEL_NAME
                self.current_key_index = 0
                self.using_fallback = True
                logger.info(f"Switching to fallback model: {self.current_model}")
                self._setup_llm()
                self.qa_chain = self._create_qa_chain()
                return True
            else:
                # We've tried all keys with all models
                logger.error("All API keys and models have been exhausted")
                return False
        else:
            # Try the next key
            self.current_key_index += 1
            logger.info(f"Switching to API key {self.current_key_index}")
            self._setup_llm()
            self.qa_chain = self._create_qa_chain()
            return True
    
    async def _execute_with_fallback(self, operation_func, *args, **kwargs):
        """Execute an operation with fallback mechanisms for API quota issues"""
        retries = 0
        while retries <= settings.MAX_RETRIES:
            try:
                return await operation_func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"API Error: {error_str}")
                
                # Handle quota and rate limit errors
                if ("quota" in error_str or 
                    "rate limit" in error_str or 
                    "capacity" in error_str or 
                    "429" in error_str or
                    "insufficient_quota" in error_str):
                    
                    if self._try_next_api_key():
                        retries += 1
                        await asyncio.sleep(settings.RETRY_DELAY)
                        continue
                    else:
                        # All fallbacks exhausted
                        return settings.FALLBACK_RESPONSE
                else:
                    # For other errors, re-raise
                    raise
        
        return settings.FALLBACK_RESPONSE
    
    async def get_response(self, question: str) -> str:
        """Get AI response for the given question."""
        async def _get_response():
            response = await self.qa_chain.arun(question=question)
            return response.strip()
        
        return await self._execute_with_fallback(_get_response)
    
    async def generate_content(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate content based on the provided prompt."""
        async def _generate_content():
            content_prompt = PromptTemplate(
                input_variables=["prompt"],
                template="""Create content based on the following instructions:
                
                Instructions: {prompt}
                
                Generated content:
                """
            )
            
            content_chain = LLMChain(
                llm=self.llm,
                prompt=content_prompt,
                verbose=True
            )
            
            response = await content_chain.arun(prompt=prompt)
            return response.strip()
        
        return await self._execute_with_fallback(_generate_content)
    
    async def process_document(self, file_path: str, question: str) -> str:
        """Process a document and answer questions about it."""
        async def _process_document():
            # Load and split the document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEYS[self.current_key_index])
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Create retrieval QA chain
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            # Get response
            response = await qa_chain.ainvoke({"query": question})
            return response["result"]
            
        return await self._execute_with_fallback(_process_document)