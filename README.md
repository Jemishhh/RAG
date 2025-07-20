# 🚀 Advanced RAG PDF Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) system for intelligent PDF document analysis with hybrid search, semantic chunking, and professional response formatting.

## ✨ Features

### 🔍 Advanced Retrieval System

- **Hybrid Search**: Combines dense vector search with sparse BM25 retrieval
- **Query Expansion**: Intelligent query enhancement using LLM
- **Semantic Reranking**: Advanced document reranking with TF-IDF similarity
- **Diversity Boost**: Prevents similar documents from clustering together
- **Fallback Retrieval**: Local search when vector store is unavailable

### 📄 Smart Document Processing

- **Semantic Chunking**: Intelligent text segmentation based on document structure
- **Boundary Detection**: Identifies natural document boundaries (sections, chapters, etc.)
- **Context-Aware Splitting**: Preserves semantic coherence in chunks
- **Hierarchical Processing**: Multi-level document understanding

### 🎯 Professional Response System

- **Structured Output**: Professional formatting with executive summaries
- **Source Citations**: Automatic reference to document sections and pages
- **Financial Analysis**: Specialized handling of financial data and metrics
- **Business Intelligence**: Strategic insights and trend analysis
- **Visual Formatting**: Rich text formatting with emojis and styling

### ⚡ Performance Optimizations

- **Intelligent Caching**: LRU cache for query expansions and responses
- **Real-time Statistics**: Live performance metrics and cache hit rates
- **Configurable Parameters**: Fine-tuned retrieval and processing settings
- **Error Handling**: Robust error management and graceful degradation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Document       │───▶│  Vector Store   │
│   & Processing  │    │  Chunking       │    │  (Qdrant)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query          │───▶│  Hybrid         │
│   Interface     │    │  Expansion      │    │  Retrieval      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Professional   │◀───│  LLM Response   │◀───│  Document       │
│  Response       │    │  Generation     │    │  Reranking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
# Google AI API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Qdrant Vector Database Configuration (Optional)
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Flask Configuration
FLASK_DEBUG=False
FLASK_ENV=production
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

### 4. Access the Interface

Open your browser and navigate to: `http://localhost:5000/ui`

## 📋 Configuration

### Query Expansion Configuration

```python
QueryExpansionConfig(
    enable_llm_expansion=True,      # Enable LLM-based query expansion
    max_expanded_terms=5,           # Maximum terms to add
    cache_expansions=True,          # Cache expansion results
    expansion_threshold=8,          # Minimum query length for expansion
    enable_synonym_expansion=True,  # Include synonyms
    enable_context_aware_expansion=True  # Context-aware expansion
)
```

### Retrieval Configuration

```python
RetrievalConfig(
    enable_hybrid_search=True,      # Enable hybrid dense+sparse search
    dense_weight=0.7,               # Weight for dense vector search
    sparse_weight=0.3,              # Weight for sparse BM25 search
    top_k_dense=8,                  # Top k for dense retrieval
    top_k_sparse=5,                 # Top k for sparse retrieval
    final_top_k=6,                  # Final number of documents
    enable_reranking=True,          # Enable semantic reranking
    rerank_threshold=0.25,          # Reranking threshold
    enable_diversity_boost=True,    # Prevent similar document clustering
    diversity_threshold=0.3         # Diversity threshold
)
```

### Chunking Configuration

```python
ChunkingConfig(
    chunk_size=1000,                # Target chunk size
    chunk_overlap=150,              # Overlap between chunks
    enable_semantic_chunking=True,  # Enable semantic boundary detection
    enable_boundary_detection=True, # Detect document boundaries
    min_chunk_size=80,              # Minimum chunk size
    max_chunk_size=1800,            # Maximum chunk size
    enable_hierarchical_chunking=True  # Hierarchical processing
)
```

## 🎯 Usage Examples

### Financial Analysis

```
User: "What are the key financial highlights for Q3 2023?"

Assistant:
**Executive Summary**
The company reported strong Q3 2023 performance with **revenue growth of 15.2%** and **EBITDA margin expansion** to 18.5%.

**Detailed Analysis**
• **Revenue Performance**: Total revenue reached **$2.5M** (+15.2% YoY)
• **Profitability Metrics**:
  - Gross margin: **45.2%** (vs 42.1% Q3 2022)
  - Operating margin: **12.8%** (vs 10.5% Q3 2022)
  - Net profit margin: **8.5%** (vs 6.8% Q3 2022)
• **Cash Flow**: Operating cash flow of **$1.2M**, up 22% YoY
• **Balance Sheet**: Total assets **$10.5M**, current ratio **2.1x**

**Key Takeaways**
- Strong revenue growth driven by new product launches
- Improved profitability through operational efficiency
- Healthy cash generation supporting growth initiatives

**Source References**
[Section: Financial Performance, Page: 15-18]
[Section: Management Discussion, Page: 25-28]
```

### Risk Assessment

```
User: "What are the main risks mentioned in the document?"

Assistant:
**Executive Summary**
The document identifies **5 primary risk categories** with **market volatility** and **regulatory changes** as the highest priority concerns.

**Detailed Analysis**
• **Market Risks**:
  - **Market volatility**: 15-20% potential impact on revenue
  - **Competitive pressure**: New entrants in key segments
  - **Economic downturn**: Sensitivity to GDP growth rates

• **Operational Risks**:
  - **Supply chain disruption**: Single-source dependencies
  - **Technology obsolescence**: Rapid industry changes
  - **Key personnel loss**: Critical skill dependencies

• **Regulatory Risks**:
  - **Compliance changes**: New industry regulations
  - **Tax policy changes**: Potential impact on profitability
  - **Environmental regulations**: Sustainability requirements

**Risk Mitigation Strategies**
- Diversified supply chain with multiple vendors
- Continuous technology investment and R&D
- Comprehensive insurance coverage
- Regular compliance audits and training

**Source References**
[Section: Risk Factors, Page: 45-52]
[Section: Risk Management, Page: 53-58]
```

## 🔧 API Endpoints

### Web Interface

- `GET /ui` - Interactive web interface

### Core API

- `GET /` - API information and endpoints
- `POST /upload` - Upload PDF document
- `POST /chat` - Send chat message
- `POST /clear` - Clear document collection
- `GET /stats` - Get system statistics

### Example API Usage

```python
import requests

# Upload PDF
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/upload', files=files)
    print(response.json())

# Send chat message
data = {
    'messages': [
        {'role': 'user', 'content': 'What are the key financial metrics?'}
    ]
}
response = requests.post('http://localhost:5000/chat', json=data)
print(response.json()['response'])

# Get statistics
stats = requests.get('http://localhost:5000/stats').json()
print(f"Documents: {stats['documents_count']}")
print(f"Cache hit rate: {stats['cache_stats']['hit_rate']}")
```

## 📊 Performance Monitoring

The system provides real-time performance metrics:

- **Document Count**: Number of processed documents
- **Cache Hit Rate**: Query expansion cache efficiency
- **Vector Store Status**: Cloud vs local storage
- **Retrieval Performance**: Search accuracy and speed

## 🛠️ Advanced Features

### Custom Configurations

```python
from pdf_chatbot import EnhancedPDFChatbot, QueryExpansionConfig, RetrievalConfig

# Custom configuration
custom_query_config = QueryExpansionConfig(
    enable_llm_expansion=True,
    max_expanded_terms=8,
    expansion_threshold=5
)

custom_retrieval_config = RetrievalConfig(
    enable_hybrid_search=True,
    dense_weight=0.8,
    sparse_weight=0.2,
    final_top_k=8
)

# Initialize with custom config
chatbot = EnhancedPDFChatbot(
    google_api_key="your_key",
    query_config=custom_query_config,
    retrieval_config=custom_retrieval_config
)
```

### Batch Processing

```python
# Process multiple documents
documents = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
for doc in documents:
    with open(doc, 'rb') as f:
        result = chatbot.upload_pdf_from_bytes(f.read(), doc)
        print(f"Processed {doc}: {result}")
```

## 🔒 Security & Best Practices

### API Key Management

- Store API keys in environment variables
- Never commit API keys to version control
- Use separate keys for development and production

### File Upload Security

- Maximum file size: 16MB
- PDF file validation
- Temporary file cleanup

### Error Handling

- Graceful degradation when services are unavailable
- Comprehensive error logging
- User-friendly error messages

## 🚀 Deployment

### Local Development

```bash
python app.py
```

### Production Deployment

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t rag-chatbot .
docker run -p 5000:5000 rag-chatbot
```

### Environment Variables for Production

```bash
export GOOGLE_API_KEY="your_production_key"
export QDRANT_URL="your_qdrant_url"
export QDRANT_API_KEY="your_qdrant_key"
export FLASK_ENV="production"
export FLASK_DEBUG="False"
```

## 📈 Performance Optimization

### Cache Optimization

- Query expansion caching reduces API calls
- Configurable cache size and eviction policies
- Cache hit rate monitoring

### Retrieval Optimization

- Hybrid search balances accuracy and speed
- Configurable weights for dense vs sparse retrieval
- Diversity boosting prevents redundant results

### Memory Management

- Efficient document chunking reduces memory usage
- LRU cache prevents memory leaks
- Automatic cleanup of temporary files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🔄 Version History

### v2.0 (Current)

- Enhanced system prompts with comprehensive instructions
- Advanced hybrid retrieval with diversity boosting
- Professional response formatting
- Real-time performance monitoring
- Improved error handling and logging

### v1.0

- Basic RAG functionality
- Simple vector search
- Basic PDF processing
- Web interface

---

**Built with ❤️ using LangChain, Google AI, and modern web technologies**
