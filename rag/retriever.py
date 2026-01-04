"""
Top-k retrieval function for RAG pipeline.

This module provides a high-level interface for retrieving relevant
documents from the vector store given a user query.
"""

from typing import List, Dict, Optional
from .vector_store import VectorStore
from .catalog_loader import CatalogLoader


class RAGRetriever:
    """Retriever for RAG pipeline combining vector store and catalog."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        catalog_loader: Optional[CatalogLoader] = None
    ):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Initialized VectorStore instance
            catalog_loader: Optional CatalogLoader for additional metadata
        """
        self.vector_store = vector_store
        self.catalog_loader = catalog_loader
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a query.
        
        Args:
            query: User query text
            k: Number of documents to retrieve
            min_score: Minimum similarity score threshold
        
        Returns:
            List of retrieved documents with scores and metadata
        """
        results = self.vector_store.search(query, k=k)
        
        # Filter by minimum score
        filtered_results = [
            r for r in results
            if r['score'] >= min_score
        ]
        
        return filtered_results
    
    def get_context(self, query: str, k: int = 3) -> str:
        """
        Get formatted context string from retrieved documents.
        
        This is useful for injecting context into LLM prompts.
        
        Args:
            query: User query
            k: Number of documents to include in context
        
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k=k)
        
        if not results:
            return "No relevant product information found."
        
        context_parts = ["Relevant Product Information:"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n[{i}] {result['document']}")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from rag.catalog_loader import CatalogLoader
    from rag.vector_store import VectorStore
    
    # Initialize components
    catalog_loader = CatalogLoader()
    vector_store = VectorStore(index_path="./faiss_index/index.faiss")
    
    # Load or build index
    index_path = Path("./faiss_index/index.faiss")
    if index_path.exists():
        print("Loading existing index...")
        vector_store.load_index()
    else:
        print("Building new index...")
        documents = catalog_loader.get_product_documents()
        metadata = catalog_loader.get_product_metadata()
        vector_store.build_index(documents, metadata)
    
    # Initialize retriever
    retriever = RAGRetriever(vector_store, catalog_loader)
    
    # Test queries
    test_queries = [
        "What laptops are available?",
        "Tell me about return policies",
        "How much does shipping cost?",
        "What electronics do you have in stock?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = retriever.retrieve(query, k=3)
        print(f"\nRetrieved {len(results)} documents:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            if result['metadata']:
                print(f"   Product: {result['metadata'].get('product_name', 'N/A')}")
            print(f"   Preview: {result['document'][:150]}...")
        
        # Get formatted context
        context = retriever.get_context(query, k=2)
        print(f"\nFormatted Context:\n{context[:300]}...")

