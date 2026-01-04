"""
Build and manage FAISS vector database for RAG retrieval.

This module handles embedding generation and FAISS index creation
for efficient similarity search over product catalog documents.
"""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """FAISS vector store for product catalog documents."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None
    ):
        """
        Initialize vector store with embedding model.
        
        Args:
            model_name: Hugging Face model identifier for embeddings
            index_path: Optional path to save/load FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.embedding_model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.documents: List[str] = []
        self.metadata: List[dict] = []
    
    def build_index(
        self,
        documents: List[str],
        metadata: Optional[List[dict]] = None
    ) -> None:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of text documents to index
            metadata: Optional metadata for each document
        """
        print(f"Generating embeddings for {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (using inner product for normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents = documents
        self.metadata = metadata if metadata else [{}] * len(documents)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
        
        # Save if path provided
        if self.index_path:
            self.save_index()
    
    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Save FAISS index and associated data to disk.
        
        Args:
            index_path: Optional override for save path
        """
        save_path = index_path or self.index_path
        if not save_path:
            raise ValueError("No index path provided for saving")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path))
        
        # Save documents and metadata
        import pickle
        data_path = save_path.with_suffix('.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'model_name': self.model_name
            }, f)
        
        print(f"Saved index to {save_path}")
        print(f"Saved documents/metadata to {data_path}")
    
    def load_index(self, index_path: Optional[str] = None) -> None:
        """
        Load FAISS index and associated data from disk.
        
        Args:
            index_path: Optional override for load path
        """
        load_path = index_path or self.index_path
        if not load_path:
            raise ValueError("No index path provided for loading")
        
        load_path = Path(load_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path))
        
        # Load documents and metadata
        import pickle
        data_path = load_path.with_suffix('.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            # Verify model matches
            if data.get('model_name') != self.model_name:
                print(f"Warning: Saved model ({data.get('model_name')}) differs from current ({self.model_name})")
        
        print(f"Loaded index from {load_path}")
        print(f"Index contains {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[dict]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
        
        Returns:
            List of dictionaries with 'document', 'score', and 'metadata' keys
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                })
        
        return results


if __name__ == "__main__":
    # Example usage
    from rag.catalog_loader import CatalogLoader
    
    # Load documents from database
    loader = CatalogLoader()
    documents = loader.get_product_documents()
    metadata = loader.get_product_metadata()
    
    # Build vector store
    vector_store = VectorStore(
        index_path="./faiss_index/index.faiss"
    )
    vector_store.build_index(documents, metadata)
    
    # Test search
    query = "What laptops do you have in stock?"
    results = vector_store.search(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Product: {result['metadata'].get('product_name', 'N/A')}")
        print(f"   Document: {result['document'][:200]}...")

