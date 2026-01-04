"""
Generate synthetic e-commerce QA dataset for fine-tuning.

This script creates a JSONL dataset with 5k-10k question-answer pairs
covering product inquiries, shipping, returns, and order status.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from faker import Faker

fake = Faker()


def generate_product_qa() -> List[Dict[str, str]]:
    """Generate product-related Q&A pairs."""
    products = [
        ("laptop", "Dell XPS 15, 16GB RAM, 512GB SSD, Intel i7 processor"),
        ("smartphone", "iPhone 15 Pro, 256GB, Titanium finish, Pro camera system"),
        ("headphones", "Sony WH-1000XM5, noise-cancelling, 30-hour battery"),
        ("smartwatch", "Apple Watch Series 9, GPS, 45mm, aluminum case"),
        ("tablet", "iPad Air, 10.9-inch, 256GB, M2 chip"),
        ("camera", "Canon EOS R6, 20MP, 4K video, full-frame sensor"),
        ("monitor", "LG UltraGear 27-inch, 4K, 144Hz, IPS panel"),
        ("keyboard", "Mechanical keyboard, RGB backlight, Cherry MX switches"),
    ]
    
    qa_pairs = []
    
    for product_name, description in products:
        # Product specifications
        qa_pairs.append({
            "question": f"What are the specifications of the {product_name}?",
            "answer": f"The {product_name} features: {description}. It's designed for professional use and comes with a 1-year warranty.",
            "category": "product_info"
        })
        
        # Availability
        qa_pairs.append({
            "question": f"Is the {product_name} in stock?",
            "answer": f"Yes, the {product_name} is currently in stock and available for immediate shipping. We have {random.randint(5, 50)} units available.",
            "category": "product_info"
        })
        
        # Price
        price = random.randint(200, 2000)
        qa_pairs.append({
            "question": f"How much does the {product_name} cost?",
            "answer": f"The {product_name} is priced at ${price}. We also offer financing options with 0% APR for qualified buyers.",
            "category": "product_info"
        })
        
        # Compatibility
        qa_pairs.append({
            "question": f"Will the {product_name} work with my existing setup?",
            "answer": f"The {product_name} is compatible with most modern systems. It supports USB-C, Bluetooth 5.0, and works with Windows, macOS, and Linux. Please check the product page for detailed compatibility information.",
            "category": "product_info"
        })
    
    return qa_pairs


def generate_shipping_qa() -> List[Dict[str, str]]:
    """Generate shipping-related Q&A pairs."""
    qa_pairs = []
    
    shipping_options = [
        ("standard", "5-7 business days", 5.99),
        ("express", "2-3 business days", 12.99),
        ("overnight", "next business day", 24.99),
        ("free", "7-10 business days", 0.00),
    ]
    
    for option_name, delivery_time, cost in shipping_options:
        qa_pairs.append({
            "question": f"How long does {option_name} shipping take?",
            "answer": f"{option_name.capitalize()} shipping typically takes {delivery_time}. Orders placed before 2 PM EST on weekdays are processed the same day.",
            "category": "shipping"
        })
        
        qa_pairs.append({
            "question": f"How much does {option_name} shipping cost?",
            "answer": f"{option_name.capitalize()} shipping costs ${cost:.2f}. Free shipping is available on orders over $50.",
            "category": "shipping"
        })
    
    # International shipping
    qa_pairs.extend([
        {
            "question": "Do you ship internationally?",
            "answer": "Yes, we ship to over 50 countries worldwide. International shipping typically takes 10-21 business days and costs vary by destination. Customs fees may apply.",
            "category": "shipping"
        },
        {
            "question": "Can I track my order?",
            "answer": "Yes, once your order ships, you'll receive a tracking number via email. You can track your package in real-time on our website or through the carrier's website.",
            "category": "shipping"
        },
        {
            "question": "What happens if my package is lost?",
            "answer": "If your package is lost in transit, please contact our support team within 30 days of the expected delivery date. We'll investigate and either reship your order or provide a full refund.",
            "category": "shipping"
        },
        {
            "question": "Can I change my shipping address after ordering?",
            "answer": "You can change your shipping address within 2 hours of placing your order by contacting customer service. After that, changes may not be possible if the order has already been processed.",
            "category": "shipping"
        },
    ])
    
    return qa_pairs


def generate_returns_qa() -> List[Dict[str, str]]:
    """Generate returns and refunds Q&A pairs."""
    qa_pairs = [
        {
            "question": "What is your return policy?",
            "answer": "We offer a 30-day return policy for unused items in original packaging. Items must be in new condition with all accessories included. Electronics must be unopened or have protective film intact.",
            "category": "returns"
        },
        {
            "question": "How do I return an item?",
            "answer": "To return an item, log into your account, go to 'My Orders', select the item you want to return, and click 'Return Item'. You'll receive a prepaid return label via email. Print it, attach it to your package, and drop it off at any carrier location.",
            "category": "returns"
        },
        {
            "question": "How long do refunds take?",
            "answer": "Once we receive your returned item, we'll process your refund within 5-7 business days. The refund will appear in your original payment method within 2-3 additional business days, depending on your bank.",
            "category": "returns"
        },
        {
            "question": "Can I return a used item?",
            "answer": "Items must be in new, unused condition to be eligible for return. If an item shows signs of use, wear, or damage, it may not qualify for a full refund. We may offer a partial refund or store credit instead.",
            "category": "returns"
        },
        {
            "question": "Do you offer exchanges?",
            "answer": "Yes, we offer exchanges for different sizes, colors, or models. Start a return request and select 'Exchange' instead of 'Refund'. If the new item costs more, you'll pay the difference. If it costs less, you'll receive a refund for the difference.",
            "category": "returns"
        },
        {
            "question": "What if I received a damaged item?",
            "answer": "If you received a damaged item, please contact us immediately with photos of the damage. We'll send a replacement right away at no cost to you, and you can keep or return the damaged item using a prepaid label we'll provide.",
            "category": "returns"
        },
        {
            "question": "Are there items that cannot be returned?",
            "answer": "Yes, certain items cannot be returned for hygiene or safety reasons: opened software, personalized items, gift cards, perishable goods, and intimate apparel. Please check the product page for specific return restrictions.",
            "category": "returns"
        },
    ]
    
    return qa_pairs


def generate_order_status_qa() -> List[Dict[str, str]]:
    """Generate order status Q&A pairs."""
    qa_pairs = [
        {
            "question": "How can I check my order status?",
            "answer": "You can check your order status by logging into your account and visiting 'My Orders', or by using the order confirmation number sent to your email. You'll see real-time updates including order confirmation, processing, shipping, and delivery.",
            "category": "order_status"
        },
        {
            "question": "My order hasn't shipped yet, why?",
            "answer": "Orders typically ship within 1-2 business days. If your order hasn't shipped after 3 business days, it may be due to inventory issues, payment verification, or high order volume. Please contact support with your order number for assistance.",
            "category": "order_status"
        },
        {
            "question": "Can I cancel my order?",
            "answer": "You can cancel your order within 1 hour of placing it through your account. After that, if the order hasn't shipped, contact customer service immediately. Once an order ships, it cannot be cancelled but can be returned after delivery.",
            "category": "order_status"
        },
        {
            "question": "I didn't receive my order confirmation email",
            "answer": "Please check your spam/junk folder first. If you still don't see it, verify the email address used during checkout. You can also view your order in your account under 'My Orders'. Contact support if you need assistance.",
            "category": "order_status"
        },
        {
            "question": "Can I modify my order after placing it?",
            "answer": "You can modify your order within 1 hour of placing it if it hasn't been processed. After that, you'll need to contact customer service. Changes may not be possible if the order is already being prepared for shipment.",
            "category": "order_status"
        },
    ]
    
    return qa_pairs


def generate_synthetic_qa(num_samples: int = 8000) -> List[Dict[str, str]]:
    """
    Generate synthetic e-commerce Q&A dataset.
    
    Args:
        num_samples: Target number of samples (will generate closest possible)
    
    Returns:
        List of Q&A dictionaries with question, answer, and category
    """
    all_qa = []
    
    # Generate base Q&A pairs
    all_qa.extend(generate_product_qa())
    all_qa.extend(generate_shipping_qa())
    all_qa.extend(generate_returns_qa())
    all_qa.extend(generate_order_status_qa())
    
    # Expand dataset with variations
    variations = []
    for qa in all_qa:
        # Add original
        variations.append(qa)
        
        # Add rephrased variations
        if "how much" in qa["question"].lower():
            variations.append({
                "question": qa["question"].replace("How much", "What is the price"),
                "answer": qa["answer"],
                "category": qa["category"]
            })
        
        if "when" in qa["question"].lower() or "how long" in qa["question"].lower():
            variations.append({
                "question": qa["question"].replace("How long", "When will").replace("when", "how long"),
                "answer": qa["answer"],
                "category": qa["category"]
            })
    
    # Add more synthetic variations using faker
    base_qa = variations.copy()
    while len(variations) < num_samples:
        base = random.choice(base_qa)
        
        # Create variations with different wording
        new_qa = {
            "question": base["question"],
            "answer": base["answer"],
            "category": base["category"]
        }
        
        # Add some randomization
        if random.random() < 0.3:
            # Add customer name variation
            customer_name = fake.first_name()
            new_qa["question"] = f"Hi, {new_qa['question'].lower()}"
        
        variations.append(new_qa)
        
        if len(variations) >= num_samples:
            break
    
    # Trim to exact number if needed
    return variations[:num_samples]


def save_dataset(qa_pairs: List[Dict[str, str]], output_path: Path) -> None:
    """
    Save Q&A pairs to JSONL file.
    
    Args:
        qa_pairs: List of Q&A dictionaries
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            # Format for instruction-following models
            formatted = {
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"],
                "category": qa["category"]
            }
            f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(qa_pairs)} Q&A pairs and saved to {output_path}")


def main():
    """Main function to generate dataset."""
    output_path = Path(__file__).parent / "dataset.jsonl"
    
    print("Generating synthetic e-commerce QA dataset...")
    qa_pairs = generate_synthetic_qa(num_samples=8000)
    
    print(f"Generated {len(qa_pairs)} Q&A pairs")
    print(f"Categories: {set(qa['category'] for qa in qa_pairs)}")
    
    save_dataset(qa_pairs, output_path)
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()

