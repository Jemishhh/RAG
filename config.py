"""
Configuration file for Enhanced PDF Chatbot
Set your API keys and configuration options here
"""

import os
from typing import Optional
from pdf_chatbot import QueryExpansionConfig, RetrievalConfig, ChunkingConfig, ResponseConfig

# API Configuration - Use environment variables
GOOGLE_API_KEY: Optional[str] = "AIzaSyBKSfNAqFAAeThlPgE7aQ9OaYhzrUqotKM"
QDRANT_URL: Optional[str] = os.environ.get("QDRANT_URL")
QDRANT_API_KEY: Optional[str] = os.environ.get("QDRANT_API_KEY")

# Application Configuration
COLLECTION_NAME: str = "enhanced_pdf_documents"
MAX_FILE_SIZE: int = 16 * 1024 * 1024  # 16MB
DEBUG_MODE: bool = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

# Enhanced Default Configurations
DEFAULT_QUERY_CONFIG = QueryExpansionConfig(
    enable_llm_expansion=False,  # Disable LLM expansion for now
    max_expanded_terms=3,
    cache_expansions=True,
    expansion_threshold=15,  # Only expand longer queries
    enable_synonym_expansion=True,
    enable_context_aware_expansion=False  # Disable context-aware expansion
)

DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig(
    enable_hybrid_search=True,
    dense_weight=0.7,
    sparse_weight=0.3,
    top_k_dense=8,
    top_k_sparse=5,
    final_top_k=6,
    enable_reranking=True,
    rerank_threshold=0.25,
    rerank_min_query_length=10,
    enable_diversity_boost=True,
    diversity_threshold=0.3
)

DEFAULT_CHUNKING_CONFIG = ChunkingConfig(
    chunk_size=1000,
    chunk_overlap=150,
    enable_semantic_chunking=True,
    enable_boundary_detection=True,
    min_chunk_size=80,
    max_chunk_size=1800,
    enable_hierarchical_chunking=True
)

DEFAULT_RESPONSE_CONFIG = ResponseConfig(
    enable_structured_output=True,
    enable_source_citations=True,
    enable_confidence_scoring=True,
    max_response_length=2000,
    enable_follow_up_suggestions=True,
    enable_visual_formatting=True
)

# Performance Configuration
CACHE_SIZE: int = 2000
MAX_TOKENS: int = 8000
TEMPERATURE: float = 0.1

# Validation
def validate_config() -> bool:
    """Validate that required configuration is present"""
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set. Set it in environment variables.")
        print("   Create a .env file with: GOOGLE_API_KEY=your_actual_api_key")
        print("   Or set it as an environment variable: export GOOGLE_API_KEY=your_key")
        return False
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Info: QDRANT_URL and QDRANT_API_KEY not set. Using local fallback only.")
        print("   For better performance, set up Qdrant cloud or local instance.")
    
    return True

def get_chatbot_config():
    """Get configuration for chatbot initialization"""
    return {
        "google_api_key": GOOGLE_API_KEY,
        "qdrant_url": QDRANT_URL,
        "qdrant_api_key": QDRANT_API_KEY,
        "collection_name": COLLECTION_NAME,
        "query_config": DEFAULT_QUERY_CONFIG,
        "retrieval_config": DEFAULT_RETRIEVAL_CONFIG,
        "chunking_config": DEFAULT_CHUNKING_CONFIG,
        "response_config": DEFAULT_RESPONSE_CONFIG
    }

def get_performance_config():
    """Get performance-related configuration"""
    return {
        "cache_size": CACHE_SIZE,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "max_file_size": MAX_FILE_SIZE,
        "debug_mode": DEBUG_MODE
    } 