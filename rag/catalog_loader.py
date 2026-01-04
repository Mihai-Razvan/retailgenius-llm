"""
Load product catalog from PostgreSQL database.

This module handles database connections and extracts product information
for building the RAG vector store.
"""

import os
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool


class CatalogLoader:
    """Load product catalog from PostgreSQL database."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "retailgenius",
        user: str = "postgres",
        password: str = "postgres",
        connection_pool: Optional[SimpleConnectionPool] = None
    ):
        """
        Initialize catalog loader with database connection parameters.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            connection_pool: Optional pre-configured connection pool
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection_pool = connection_pool
    
    def _get_connection(self):
        """Get database connection from pool or create new one."""
        if self.connection_pool:
            return self.connection_pool.getconn()
        else:
            return psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
    
    def _return_connection(self, conn):
        """Return connection to pool if using pool."""
        if self.connection_pool:
            self.connection_pool.putconn(conn)
        else:
            conn.close()
    
    def create_sample_catalog(self):
        """
        Create sample product catalog table and insert sample data.
        
        This is a helper function for demonstration. In production,
        the catalog would already exist in the database.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create products table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS products (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        category VARCHAR(100),
                        price DECIMAL(10, 2),
                        specifications TEXT,
                        shipping_info TEXT,
                        return_policy TEXT,
                        in_stock BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert sample products
                sample_products = [
                    {
                        "name": "Dell XPS 15 Laptop",
                        "description": "Premium 15-inch laptop with Intel i7 processor, 16GB RAM, and 512GB SSD. Perfect for professionals and content creators.",
                        "category": "Electronics",
                        "price": 1299.99,
                        "specifications": "Intel i7-13700H, 16GB DDR5 RAM, 512GB NVMe SSD, NVIDIA RTX 4050, 15.6-inch 4K OLED display",
                        "shipping_info": "Free shipping on orders over $50. Standard shipping: 5-7 business days. Express: 2-3 business days available.",
                        "return_policy": "30-day return policy. Must be in original packaging with all accessories. Unopened items eligible for full refund.",
                        "in_stock": True
                    },
                    {
                        "name": "iPhone 15 Pro",
                        "description": "Latest iPhone with Titanium design, A17 Pro chip, and advanced camera system. Available in multiple storage options.",
                        "category": "Electronics",
                        "price": 999.00,
                        "specifications": "6.1-inch Super Retina XDR display, A17 Pro chip, 48MP main camera, 128GB/256GB/512GB/1TB storage options",
                        "shipping_info": "Free express shipping. Orders ship within 24 hours. Available for in-store pickup.",
                        "return_policy": "14-day return policy for unopened devices. Opened devices subject to restocking fee. Must include all original packaging.",
                        "in_stock": True
                    },
                    {
                        "name": "Sony WH-1000XM5 Headphones",
                        "description": "Premium noise-cancelling wireless headphones with 30-hour battery life and industry-leading sound quality.",
                        "category": "Electronics",
                        "price": 399.99,
                        "specifications": "Active noise cancellation, 30-hour battery, quick charge (3 min = 3 hours), Bluetooth 5.2, LDAC support",
                        "shipping_info": "Standard shipping: 5-7 business days. Free shipping on orders over $50.",
                        "return_policy": "30-day return policy. Items must be in new condition with original packaging. Electronics must be unopened or have protective film intact.",
                        "in_stock": True
                    },
                    {
                        "name": "Apple Watch Series 9",
                        "description": "Latest Apple Watch with advanced health features, GPS, and always-on display. Available in multiple sizes and colors.",
                        "category": "Electronics",
                        "price": 429.00,
                        "specifications": "45mm or 41mm case, GPS, Always-On Retina display, S9 SiP chip, 18-hour battery life, water resistant 50m",
                        "shipping_info": "Free shipping. Express shipping available. Ships within 1-2 business days.",
                        "return_policy": "14-day return policy. Must be in original packaging. Personalized items cannot be returned.",
                        "in_stock": True
                    },
                    {
                        "name": "LG UltraGear 27-inch Monitor",
                        "description": "4K gaming monitor with 144Hz refresh rate, IPS panel, and HDR support. Ideal for gaming and professional work.",
                        "category": "Electronics",
                        "price": 599.99,
                        "specifications": "27-inch 4K UHD, 144Hz refresh rate, IPS panel, HDR10, 1ms response time, G-Sync compatible, USB-C connectivity",
                        "shipping_info": "Free shipping on orders over $50. Standard shipping: 5-7 business days. Large item - may require signature.",
                        "return_policy": "30-day return policy. Must be in original packaging. No restocking fee for unopened items.",
                        "in_stock": True
                    },
                ]
                
                # Clear existing data (for demo purposes)
                cur.execute("DELETE FROM products")
                
                # Insert sample products
                for product in sample_products:
                    cur.execute("""
                        INSERT INTO products (name, description, category, price, specifications, shipping_info, return_policy, in_stock)
                        VALUES (%(name)s, %(description)s, %(category)s, %(price)s, %(specifications)s, %(shipping_info)s, %(return_policy)s, %(in_stock)s)
                    """, product)
                
                conn.commit()
                print(f"Created sample catalog with {len(sample_products)} products")
        
        finally:
            self._return_connection(conn)
    
    def load_products(self) -> List[Dict]:
        """
        Load all products from the database.
        
        Returns:
            List of product dictionaries with all fields
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, description, category, price, specifications,
                           shipping_info, return_policy, in_stock
                    FROM products
                    ORDER BY id
                """)
                products = cur.fetchall()
                return [dict(product) for product in products]
        finally:
            self._return_connection(conn)
    
    def get_product_documents(self) -> List[str]:
        """
        Convert products to text documents for vectorization.
        
        Each product is converted to a comprehensive text document
        containing all relevant information for RAG retrieval.
        
        Returns:
            List of text documents, one per product
        """
        products = self.load_products()
        documents = []
        
        for product in products:
            # Create comprehensive document with all product information
            doc_parts = [
                f"Product: {product['name']}",
                f"Category: {product['category']}",
                f"Price: ${product['price']:.2f}",
                f"Description: {product['description']}",
            ]
            
            if product.get('specifications'):
                doc_parts.append(f"Specifications: {product['specifications']}")
            
            if product.get('shipping_info'):
                doc_parts.append(f"Shipping Information: {product['shipping_info']}")
            
            if product.get('return_policy'):
                doc_parts.append(f"Return Policy: {product['return_policy']}")
            
            availability = "In Stock" if product.get('in_stock') else "Out of Stock"
            doc_parts.append(f"Availability: {availability}")
            
            documents.append("\n".join(doc_parts))
        
        return documents
    
    def get_product_metadata(self) -> List[Dict]:
        """
        Get product metadata for each document.
        
        Returns:
            List of metadata dictionaries corresponding to documents
        """
        products = self.load_products()
        metadata = []
        
        for product in products:
            metadata.append({
                "product_id": product['id'],
                "product_name": product['name'],
                "category": product['category'],
                "price": float(product['price']) if product['price'] else None,
            })
        
        return metadata


if __name__ == "__main__":
    # Example usage
    loader = CatalogLoader(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "retailgenius"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )
    
    # Create sample catalog
    loader.create_sample_catalog()
    
    # Load documents
    documents = loader.get_product_documents()
    print(f"Loaded {len(documents)} product documents")
    print("\nFirst document:")
    print(documents[0][:500] + "..." if len(documents[0]) > 500 else documents[0])

