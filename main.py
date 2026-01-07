"""
Main script demonstrating the complete RetailGenius LLM workflow.

This script ties together:
1. Dataset generation
2. Model fine-tuning
3. RAG pipeline setup
4. API server startup

Usage:
    python main.py --mode [generate|train|rag|api|all]
"""

import argparse
import sys
from pathlib import Path
import subprocess
import os


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ No GPU detected. Training will be very slow.")
            return False
    except ImportError:
        print("⚠ PyTorch not installed. Cannot check GPU.")
        return False


def generate_dataset():
    """Generate synthetic e-commerce QA dataset."""
    print("\n" + "="*60)
    print("STEP 1: Generating Dataset")
    print("="*60)
    
    from data.generate_dataset import main as generate_main
    generate_main()
    
    dataset_path = Path("data/dataset.jsonl")
    if dataset_path.exists():
        print(f"✓ Dataset generated: {dataset_path}")
        return True
    else:
        print("✗ Dataset generation failed")
        return False


def train_model():
    """Fine-tune the model using QLoRA."""
    print("\n" + "="*60)
    print("STEP 2: Fine-tuning Model")
    print("="*60)
    
    if not check_gpu():
        response = input("Continue without GPU? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return False
    
    dataset_path = Path("data/dataset.jsonl")
    if not dataset_path.exists():
        print("✗ Dataset not found. Run with --mode generate first.")
        return False
    
    config_path = Path("training/config.yaml")
    if not config_path.exists():
        print("✗ Training config not found.")
        return False
    
    print("Starting fine-tuning...")
    print("Note: This may take several hours on GPU, or much longer on CPU.")
    
    try:
        from training.train import train
        train(
            config_path=str(config_path),
            dataset_path=str(dataset_path)
        )
        
        adapter_path = Path("training/adapters")
        if adapter_path.exists() and any(adapter_path.iterdir()):
            print(f"✓ Model fine-tuning complete. Adapter saved to: {adapter_path}")
            return True
        else:
            print("⚠ Training completed but adapter not found. Check logs for errors.")
            return False
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False


def setup_rag():
    """Set up RAG pipeline with vector store."""
    print("\n" + "="*60)
    print("STEP 3: Setting up RAG Pipeline")
    print("="*60)
    
    try:
        from rag.catalog_loader import CatalogLoader
        from rag.vector_store import VectorStore
        from rag.retriever import RAGRetriever
        
        # Initialize catalog loader
        print("Connecting to database...")
        loader = CatalogLoader(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "retailgenius"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
        
        # Create sample catalog if needed
        print("Loading product catalog...")
        try:
            loader.create_sample_catalog()
        except Exception as e:
            print(f"Note: Could not create sample catalog (may already exist): {e}")
        
        # Get documents
        documents = loader.get_product_documents()
        metadata = loader.get_product_metadata()
        print(f"✓ Loaded {len(documents)} product documents")
        
        # Build vector store
        print("Building FAISS vector index...")
        index_path = Path("faiss_index/index.faiss")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        vector_store = VectorStore(index_path=str(index_path))
        vector_store.build_index(documents, metadata)
        
        print(f"✓ Vector store built and saved to: {index_path}")
        
        # Test retrieval
        print("\nTesting RAG retrieval...")
        retriever = RAGRetriever(vector_store, loader)
        test_query = "What laptops are available?"
        results = retriever.retrieve(test_query, k=2)
        print(f"✓ Test query '{test_query}' retrieved {len(results)} documents")
        
        return True
    
    except Exception as e:
        print(f"✗ RAG setup failed: {e}")
        print("Make sure PostgreSQL is running and accessible.")
        return False


def start_api():
    """Start the FastAPI server."""
    print("\n" + "="*60)
    print("STEP 4: Starting API Server")
    print("="*60)
    
    print("Starting FastAPI server on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        import uvicorn
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    except KeyboardInterrupt:
        print("\n✓ Server stopped.")
        return True
    except Exception as e:
        print(f"✗ API server failed: {e}")
        return False


def run_all():
    """Run the complete workflow."""
    print("\n" + "="*60)
    print("RetailGenius LLM - Complete Workflow")
    print("="*60)
    
    steps = [
        ("Dataset Generation", generate_dataset),
        ("Model Training", train_model),
        ("RAG Setup", setup_rag),
    ]
    
    for step_name, step_func in steps:
        success = step_func()
        if not success:
            print(f"\n✗ Workflow stopped at: {step_name}")
            print("Fix the issue and rerun, or continue manually.")
            return
    
    print("\n" + "="*60)
    print("✓ All setup steps completed!")
    print("="*60)
    print("\nYou can now start the API server with:")
    print("  python main.py --mode api")
    print("\nOr use Docker:")
    print("  docker-compose -f docker/docker-compose.yml up")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RetailGenius LLM - Complete Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode generate    # Generate dataset only
  python main.py --mode train        # Fine-tune model only
  python main.py --mode rag          # Set up RAG pipeline only
  python main.py --mode api          # Start API server only
  python main.py --mode all          # Run all setup steps
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "train", "rag", "api", "all"],
        default="all",
        help="Workflow mode to execute"
    )
    
    args = parser.parse_args()
    
    mode_map = {
        "generate": generate_dataset,
        "train": train_model,
        "rag": setup_rag,
        "api": start_api,
        "all": run_all,
    }
    
    func = mode_map[args.mode]
    func()


if __name__ == "__main__":
    main()

