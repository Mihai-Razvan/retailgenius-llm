"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message/query")
    use_rag: bool = Field(True, description="Whether to use RAG for context retrieval")
    max_tokens: int = Field(512, description="Maximum tokens in response")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Model response")
    context_used: bool = Field(..., description="Whether RAG context was used")
    retrieved_docs: Optional[List[Dict]] = Field(None, description="Retrieved documents if RAG was used")
    model_name: str = Field(..., description="Model identifier used")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    rag_available: bool = Field(..., description="Whether RAG is available")

