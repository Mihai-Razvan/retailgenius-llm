"""
FastAPI application for RetailGenius LLM customer support.

This API serves the fine-tuned LLM with RAG capabilities for
e-commerce customer support queries.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.models import ChatRequest, ChatResponse, HealthResponse
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStore
from rag.catalog_loader import CatalogLoader

app = FastAPI(
    title="RetailGenius LLM API",
    description="Fine-tuned LLM for e-commerce customer support with RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and RAG components
model = None
tokenizer = None
rag_retriever: Optional[RAGRetriever] = None
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_path = None


def load_model(base_model_name: str, adapter_path: Optional[str] = None):
    """
    Load base model and optionally fine-tuned adapter.
    
    Args:
        base_model_name: Hugging Face model identifier
        adapter_path: Optional path to fine-tuned adapter weights
    """
    global model, tokenizer, model_name
    
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization for memory efficiency
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load adapter if provided
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading fine-tuned adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model_name = f"{base_model_name} (fine-tuned)"
    else:
        print("No adapter found, using base model")
        model_name = base_model_name
    
    model.eval()
    print("Model loaded successfully")


def load_rag():
    """Initialize RAG components (vector store and retriever)."""
    global rag_retriever
    
    try:
        # Initialize catalog loader
        catalog_loader = CatalogLoader(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "retailgenius"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
        
        # Initialize vector store
        index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index/index.faiss")
        vector_store = VectorStore(index_path=index_path)
        
        # Load or build index
        if Path(index_path).exists():
            print(f"Loading FAISS index from {index_path}")
            vector_store.load_index()
        else:
            print("Building new FAISS index...")
            documents = catalog_loader.get_product_documents()
            metadata = catalog_loader.get_product_metadata()
            vector_store.build_index(documents, metadata)
        
        # Initialize retriever
        rag_retriever = RAGRetriever(vector_store, catalog_loader)
        print("RAG components loaded successfully")
    
    except Exception as e:
        print(f"Warning: Failed to load RAG components: {e}")
        print("API will work without RAG functionality")
        rag_retriever = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and RAG on startup."""
    base_model = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    adapter = os.getenv("ADAPTER_PATH", "./training/adapters")
    
    load_model(base_model, adapter)
    load_rag()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        rag_available=rag_retriever is not None
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for customer support queries.
    
    Uses fine-tuned model with optional RAG context retrieval.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Retrieve context if RAG is enabled and available
        context = ""
        retrieved_docs = None
        context_used = False
        
        if request.use_rag and rag_retriever is not None:
            try:
                retrieved = rag_retriever.retrieve(request.message, k=3)
                if retrieved:
                    context = rag_retriever.get_context(request.message, k=3)
                    retrieved_docs = [
                        {
                            "document": r["document"][:500],  # Truncate for response
                            "score": r["score"],
                            "metadata": r["metadata"]
                        }
                        for r in retrieved
                    ]
                    context_used = True
            except Exception as e:
                print(f"RAG retrieval error: {e}")
                # Continue without RAG context
        
        # Format prompt with context
        if context:
            prompt = f"""<s>[INST] You are a helpful customer support assistant for an e-commerce store.

{context}

Customer Question: {request.message}

Please provide a helpful and accurate answer based on the product information above. [/INST]"""
        else:
            prompt = f"<s>[INST] {request.message} [/INST]"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (after [/INST])
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return ChatResponse(
            response=response,
            context_used=context_used,
            retrieved_docs=retrieved_docs if context_used else None,
            model_name=model_name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RetailGenius LLM API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RetailGenius LLM API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

