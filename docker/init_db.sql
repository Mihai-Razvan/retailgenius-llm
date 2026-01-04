-- Initialize database schema for product catalog
-- This script runs automatically when PostgreSQL container starts

-- Create products table
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
);

-- Insert sample products
INSERT INTO products (name, description, category, price, specifications, shipping_info, return_policy, in_stock) VALUES
('Dell XPS 15 Laptop', 
 'Premium 15-inch laptop with Intel i7 processor, 16GB RAM, and 512GB SSD. Perfect for professionals and content creators.',
 'Electronics',
 1299.99,
 'Intel i7-13700H, 16GB DDR5 RAM, 512GB NVMe SSD, NVIDIA RTX 4050, 15.6-inch 4K OLED display',
 'Free shipping on orders over $50. Standard shipping: 5-7 business days. Express: 2-3 business days available.',
 '30-day return policy. Must be in original packaging with all accessories. Unopened items eligible for full refund.',
 TRUE),

('iPhone 15 Pro',
 'Latest iPhone with Titanium design, A17 Pro chip, and advanced camera system. Available in multiple storage options.',
 'Electronics',
 999.00,
 '6.1-inch Super Retina XDR display, A17 Pro chip, 48MP main camera, 128GB/256GB/512GB/1TB storage options',
 'Free express shipping. Orders ship within 24 hours. Available for in-store pickup.',
 '14-day return policy for unopened devices. Opened devices subject to restocking fee. Must include all original packaging.',
 TRUE),

('Sony WH-1000XM5 Headphones',
 'Premium noise-cancelling wireless headphones with 30-hour battery life and industry-leading sound quality.',
 'Electronics',
 399.99,
 'Active noise cancellation, 30-hour battery, quick charge (3 min = 3 hours), Bluetooth 5.2, LDAC support',
 'Standard shipping: 5-7 business days. Free shipping on orders over $50.',
 '30-day return policy. Items must be in new condition with original packaging. Electronics must be unopened or have protective film intact.',
 TRUE),

('Apple Watch Series 9',
 'Latest Apple Watch with advanced health features, GPS, and always-on display. Available in multiple sizes and colors.',
 'Electronics',
 429.00,
 '45mm or 41mm case, GPS, Always-On Retina display, S9 SiP chip, 18-hour battery life, water resistant 50m',
 'Free shipping. Express shipping available. Ships within 1-2 business days.',
 '14-day return policy. Must be in original packaging. Personalized items cannot be returned.',
 TRUE),

('LG UltraGear 27-inch Monitor',
 '4K gaming monitor with 144Hz refresh rate, IPS panel, and HDR support. Ideal for gaming and professional work.',
 'Electronics',
 599.99,
 '27-inch 4K UHD, 144Hz refresh rate, IPS panel, HDR10, 1ms response time, G-Sync compatible, USB-C connectivity',
 'Free shipping on orders over $50. Standard shipping: 5-7 business days. Large item - may require signature.',
 '30-day return policy. Must be in original packaging. No restocking fee for unopened items.',
 TRUE)
ON CONFLICT DO NOTHING;

