# RetailGenius LLM

A fine-tuned open-source Large Language Model (LLM) for e-commerce customer support, featuring Retrieval-Augmented Generation (RAG) capabilities and production-ready API deployment.

## 🎯 Project Overview

RetailGenius LLM is an end-to-end machine learning project that demonstrates:

- **LLM Fine-Tuning**: Efficient fine-tuning of Mistral-7B using QLoRA (Quantized Low-Rank Adaptation) for memory-efficient training
- **RAG Pipeline**: Retrieval-Augmented Generation system using FAISS vector search over product catalogs
- **Production API**: FastAPI-based REST API with Docker containerization
- **MLOps Practices**: Complete workflow from data preparation to model serving

This project showcases expertise in modern LLM development, efficient fine-tuning techniques, and production ML system design.

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Generation│
│  (Synthetic QA) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │
│  (QLoRA/PEFT)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  RAG Pipeline  │◄────│  PostgreSQL  │
│  (FAISS)       │     │  Catalog     │
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│  (/chat)        │
└─────────────────┘
```

## 📁 Project Structure

```
retailgenius-llm/
├── data/
│   ├── __init__.py
│   └── generate_dataset.py      # Synthetic e-commerce QA dataset generator
├── training/
│   ├── __init__.py
│   ├── train.py                  # QLoRA fine-tuning script
│   └── config.yaml               # Training hyperparameters
├── rag/
│   ├── __init__.py
│   ├── catalog_loader.py         # PostgreSQL product catalog loader
│   ├── vector_store.py           # FAISS vector database builder
│   └── retriever.py              # Top-k retrieval function
├── api/
│   ├── __init__.py
│   ├── app.py                    # FastAPI application
│   └── models.py                 # Pydantic request/response models
├── docker/
│   ├── Dockerfile                # API containerization
│   ├── docker-compose.yml        # Full stack orchestration
│   └── init_db.sql               # Database initialization
├── main.py                       # Complete workflow script
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ (or use Docker)
- CUDA-capable GPU (recommended for training, 16GB+ VRAM)
- Docker and Docker Compose (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/retailgenius-llm.git
   cd retailgenius-llm
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL** (if not using Docker)
   ```bash
   # Create database
   createdb retailgenius
   
   # Or use Docker
   docker run -d \
     --name retailgenius-db \
     -e POSTGRES_DB=retailgenius \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=postgres \
     -p 5432:5432 \
     postgres:15-alpine
   ```

### Usage

#### Option 1: Complete Workflow (Recommended)

Run all setup steps in sequence:

```bash
python main.py --mode all
```

This will:
1. Generate synthetic e-commerce QA dataset (8k samples)
2. Fine-tune Mistral-7B model (requires GPU)
3. Set up RAG pipeline with FAISS vector store
4. Provide instructions for starting the API

#### Option 2: Individual Steps

**Generate Dataset**
```bash
python main.py --mode generate
# Output: data/dataset.jsonl
```

**Fine-tune Model**
```bash
python main.py --mode train
# Output: training/adapters/
# Note: Requires GPU, may take several hours
```

**Set up RAG Pipeline**
```bash
# Ensure PostgreSQL is running
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=retailgenius
export DB_USER=postgres
export DB_PASSWORD=postgres

python main.py --mode rag
# Output: faiss_index/index.faiss
```

**Start API Server**
```bash
python main.py --mode api
# Server runs on http://localhost:8000
```

#### Option 3: Docker Deployment

Deploy the complete stack with Docker Compose:

```bash
cd docker
docker-compose up -d
```

The API will be available at `http://localhost:8000`.

## 📖 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "rag_available": true
}
```

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What laptops do you have in stock?",
    "use_rag": true,
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

Response:
```json
{
  "response": "We have the Dell XPS 15 Laptop in stock...",
  "context_used": true,
  "retrieved_docs": [...],
  "model_name": "mistralai/Mistral-7B-Instruct-v0.2 (fine-tuned)"
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI with interactive API testing.

## 🔧 Configuration

### Training Configuration

Edit `training/config.yaml` to adjust:
- Model name and base model
- Training hyperparameters (epochs, batch size, learning rate)
- QLoRA parameters (rank, alpha, dropout)
- Hardware settings (FP16, 4-bit quantization)

### API Configuration

Set environment variables:
```bash
export BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export ADAPTER_PATH="./training/adapters"
export DB_HOST="localhost"
export DB_PORT=5432
export DB_NAME="retailgenius"
export DB_USER="postgres"
export DB_PASSWORD="postgres"
export FAISS_INDEX_PATH="./faiss_index/index.faiss"
```

## 🧪 Technical Details

### Model Fine-Tuning

- **Base Model**: Mistral-7B-Instruct-v0.2
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Efficiency**: 4-bit quantization + LoRA adapters (reduces memory by ~75%)
- **Dataset**: 8,000 synthetic e-commerce Q&A pairs
- **Training Time**: ~3-4 hours on A100 GPU (varies by hardware)

### RAG Pipeline

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Retrieval**: Cosine similarity with top-k selection
- **Database**: PostgreSQL for product catalog storage

### API Features

- **Framework**: FastAPI with async support
- **Model Loading**: 4-bit quantized inference for memory efficiency
- **RAG Integration**: Automatic context retrieval and injection
- **Error Handling**: Comprehensive error handling and logging
- **Health Checks**: Built-in health monitoring

## 📊 Performance Considerations

- **Training**: Requires GPU with 16GB+ VRAM for efficient training
- **Inference**: Can run on CPU (slower) or GPU (faster)
- **Memory**: Model uses ~4-6GB RAM with 4-bit quantization
- **Latency**: ~1-3 seconds per request (depends on hardware)

## 🎓 CV-Friendly Description

**RetailGenius LLM** - End-to-End LLM Fine-Tuning and Production Deployment

Developed a production-ready customer support LLM system featuring:

- **Efficient Fine-Tuning**: Implemented QLoRA (Quantized Low-Rank Adaptation) to fine-tune Mistral-7B on 8k synthetic e-commerce Q&A pairs, reducing memory requirements by 75% while maintaining model quality
- **RAG Architecture**: Built a Retrieval-Augmented Generation pipeline using FAISS vector search and PostgreSQL, enabling real-time product catalog retrieval for context-aware responses
- **Production API**: Designed and deployed a FastAPI-based REST API with Docker containerization, supporting async inference and automatic context injection
- **MLOps Pipeline**: Created end-to-end workflow from synthetic data generation to model serving, including configuration management, logging, and health monitoring
- **Technologies**: PyTorch, Hugging Face Transformers, PEFT, FAISS, FastAPI, PostgreSQL, Docker

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the base model
- [Hugging Face](https://huggingface.co/) for transformers and PEFT libraries
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a portfolio project demonstrating ML engineering capabilities. For production use, additional considerations such as model evaluation, monitoring, and security should be implemented.

