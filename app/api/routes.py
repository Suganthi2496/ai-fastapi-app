from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, status
from pydantic import BaseModel
from typing import Optional, List
from app.service.ai_service import AIService
from app.core.config import settings
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    fallback_used: bool = False

class ContentRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 500

class ContentResponse(BaseModel):
    content: str
    fallback_used: bool = False

class APIErrorResponse(BaseModel):
    detail: str
    error_type: str
    suggestion: Optional[str] = None

# Dependency to get AI service
def get_ai_service():
    return AIService()

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Ask a question to the AI assistant"""
    try:
        answer = await ai_service.get_response(request.question)
        # Check if fallback response was used
        fallback_used = answer == settings.FALLBACK_RESPONSE
        return QuestionResponse(answer=answer, fallback_used=fallback_used)
    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"Error in ask_question: {error_str}")
        
        if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=APIErrorResponse(
                    detail="API quota exceeded",
                    error_type="insufficient_quota",
                    suggestion="Please try again later or contact support to upgrade your plan."
                ).dict()
            )
        elif "api key" in error_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=APIErrorResponse(
                    detail="Invalid API key",
                    error_type="authentication_error",
                    suggestion="Check your API key configuration."
                ).dict()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Error processing request: {str(e)}"
            )

@router.post("/generate", response_model=ContentResponse)
async def generate_content(
    request: ContentRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Generate content based on a prompt"""
    try:
        content = await ai_service.generate_content(request.prompt, request.max_tokens)
        # Check if fallback response was used
        fallback_used = content == settings.FALLBACK_RESPONSE
        return ContentResponse(content=content, fallback_used=fallback_used)
    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"Error in generate_content: {error_str}")
        
        if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=APIErrorResponse(
                    detail="API quota exceeded",
                    error_type="insufficient_quota",
                    suggestion="Please try again later or contact support to upgrade your plan."
                ).dict()
            )
        elif "api key" in error_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=APIErrorResponse(
                    detail="Invalid API key",
                    error_type="authentication_error",
                    suggestion="Check your API key configuration."
                ).dict()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating content: {str(e)}"
            )
    
@router.post("/document-qa")
async def document_qa(
    file: UploadFile = File(...),
    question: str = Form(...),
    ai_service: AIService = Depends(get_ai_service)
):
    """Upload a document and ask questions about it"""
    try:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        # Process the document
        answer = await ai_service.process_document(temp_file_path, question)
        
        # Clean up
        os.unlink(temp_file_path)
        
        # Check if fallback response was used
        fallback_used = answer == settings.FALLBACK_RESPONSE
        
        return {"answer": answer, "fallback_used": fallback_used}
    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"Error in document_qa: {error_str}")
        
        if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=APIErrorResponse(
                    detail="API quota exceeded",
                    error_type="insufficient_quota",
                    suggestion="Please try again later or contact support to upgrade your plan."
                ).dict()
            )
        elif "api key" in error_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=APIErrorResponse(
                    detail="Invalid API key",
                    error_type="authentication_error",
                    suggestion="Check your API key configuration."
                ).dict()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(e)}"
            )