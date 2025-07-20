import os
from enhanced_pdf_chatbot import (
    EnhancedQueryConfig, 
    EnhancedRetrievalConfig, 
    EnhancedChunkingConfig, 
    EnhancedResponseConfig
)

def get_enhanced_chatbot_config():
    """Get enhanced chatbot configuration with optimized settings"""
    return {
        'google_api_key': 'AIzaSyBKSfNAqFAAeThlPgE7aQ9OaYhzrUqotKM',
        'qdrant_url': os.environ.get('QDRANT_URL'),
        'qdrant_api_key': os.environ.get('QDRANT_API_KEY'),
        'collection_name': 'enhanced_pdf_documents',
        'query_config': EnhancedQueryConfig(
            enable_llm_expansion=False,  # Disable LLM expansion for now
            max_expanded_terms=3,
            cache_expansions=True,
            expansion_threshold=50,  # Increase threshold to avoid expanding long queries
            enable_synonym_expansion=True,
            enable_context_aware_expansion=False,  # Disable context-aware expansion
            enable_spelling_correction=False,  # Disable spelling correction to avoid query modification
            enable_query_classification=True
        ),
        'retrieval_config': EnhancedRetrievalConfig(
            enable_hybrid_search=False,  # Disable hybrid search to use simple retrieval
            dense_weight=0.65,
            sparse_weight=0.35,
            top_k_dense=12,
            top_k_sparse=10,
            final_top_k=10,
            enable_reranking=False,  # Disable reranking for simplicity
            rerank_threshold=0.2,
            rerank_min_query_length=8,
            enable_diversity_boost=False,  # Disable diversity boost
            diversity_threshold=0.25,
            enable_source_filtering=True,
            enable_relevance_scoring=True
        ),
        'chunking_config': EnhancedChunkingConfig(
            chunk_size=700,
            chunk_overlap=250,
            enable_semantic_chunking=True,
            enable_boundary_detection=True,
            min_chunk_size=120,
            max_chunk_size=1200,
            enable_hierarchical_chunking=True,
            enable_table_preservation=True,
            enable_image_captioning=True
        ),
        'response_config': EnhancedResponseConfig(
            enable_structured_output=True,
            enable_source_citations=True,
            enable_confidence_scoring=True,
            max_response_length=3000,
            enable_follow_up_suggestions=True,
            enable_visual_formatting=True,
            enable_detailed_analysis=True,
            enable_key_insights=True
        )
    }

def validate_enhanced_config():
    """Validate enhanced configuration"""
    config = get_enhanced_chatbot_config()
    
    if not config['google_api_key']:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    if not config['qdrant_url'] or not config['qdrant_api_key']:
        print("Warning: QDRANT_URL and QDRANT_API_KEY not provided. Using local fallback only.")
    
    return True

def get_performance_config():
    """Get performance optimization settings"""
    return {
        'max_file_size': 16 * 1024 * 1024,  # 16MB
        'max_concurrent_uploads': 3,
        'cache_ttl': 3600,  # 1 hour
        'rate_limit_per_minute': 60,
        'timeout_seconds': 30
    } 