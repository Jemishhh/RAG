
import os
import tempfile
import json
import hashlib
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

@dataclass
class QueryExpansionConfig:
    enable_llm_expansion: bool = True
    max_expanded_terms: int = 5
    cache_expansions: bool = True
    expansion_threshold: int = 8
    enable_synonym_expansion: bool = True
    enable_context_aware_expansion: bool = True

@dataclass
class RetrievalConfig:
    enable_hybrid_search: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    top_k_dense: int = 8
    top_k_sparse: int = 5
    final_top_k: int = 6
    enable_reranking: bool = True
    rerank_threshold: float = 0.25
    rerank_min_query_length: int = 10
    enable_diversity_boost: bool = True
    diversity_threshold: float = 0.3

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    enable_semantic_chunking: bool = True
    enable_boundary_detection: bool = True
    min_chunk_size: int = 80
    max_chunk_size: int = 1800
    enable_hierarchical_chunking: bool = True

@dataclass
class ResponseConfig:
    enable_structured_output: bool = True
    enable_source_citations: bool = True
    enable_confidence_scoring: bool = True
    max_response_length: int = 2000
    enable_follow_up_suggestions: bool = True
    enable_visual_formatting: bool = True

class LLMCache:
    def __init__(self, max_size: int = 2000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
   
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
        self.miss_count += 1
        return None
   
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
       
        self.cache[key] = value
        self.access_times[key] = time.time()
   
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self):
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{hit_rate:.1f}%"
        }

class ContextAwareChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
       
        self.section_patterns = [
            r'\n\s*(?:SECTION|Section|SEC\.)\s*\d+',
            r'\n\s*(?:CHAPTER|Chapter|CHAP\.)\s*\d+',
            r'\n\s*(?:ARTICLE|Article|ART\.)\s*\d+',
            r'\n\s*\d+\.\s*[A-Z]',
            r'\n\s*[A-Z]\.\s*[A-Z]',
            r'\n\s*(?:WHEREAS|THEREFORE|NOW)',
            r'\n\s*(?:Executive Summary|Introduction|Conclusion|Background)',
            r'\n\s*(?:Financial|Revenue|Profit|Loss|Balance)',
            r'\n\s*(?:Risk Factors|Management Discussion|Business Overview)',
            r'\n\s*(?:Notes to Financial Statements|Cash Flow|Income Statement)',
        ]
       
        self.boundary_indicators = [
            'Executive Summary', 'Introduction', 'Background', 'Methodology',
            'Results', 'Discussion', 'Conclusion', 'References', 'Appendix',
            'Financial Statements', 'Balance Sheet', 'Income Statement',
            'Cash Flow', 'Notes to Financial Statements', 'Risk Factors',
            'Management Discussion', 'Business Overview', 'Market Analysis',
            'Competitive Analysis', 'Strategic Initiatives', 'Operational Review',
            'Corporate Governance', 'Sustainability Report', 'Outlook'
        ]

    def detect_semantic_boundaries(self, text: str) -> List[int]:
        boundaries = []
       
        for pattern in self.section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                boundaries.append(match.start())
       
        for indicator in self.boundary_indicators:
            pattern = r'\n\s*' + re.escape(indicator) + r'(?:\s|:|\n)'
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                boundaries.append(match.start())
       
        # Enhanced paragraph boundary detection
        paragraph_breaks = re.finditer(r'\n\s*\n', text)
        for match in paragraph_breaks:
            pos = match.start()
            before = text[max(0, pos-150):pos].strip()
            after = text[pos:pos+150].strip()
           
            if before and after:
                # Check for topic shift
                before_words = set(before.split()[:8])
                after_words = set(after.split()[:8])
                overlap = len(before_words & after_words)
                
                if overlap < 3:  # Low overlap indicates topic shift
                    boundaries.append(pos)
       
        return sorted(list(set(boundaries)))
   
    def chunk_with_context(self, documents: List[Document]) -> List[Document]:
        if not self.config.enable_semantic_chunking:
            return self.base_splitter.split_documents(documents)
       
        enhanced_chunks = []
       
        for doc in documents:
            content = doc.page_content
           
            if self.config.enable_boundary_detection:
                boundaries = self.detect_semantic_boundaries(content)
               
                if boundaries:
                    chunks = []
                    start = 0
                   
                    for boundary in boundaries:
                        if boundary > start:
                            chunk_text = content[start:boundary].strip()
                            if len(chunk_text) >= self.config.min_chunk_size:
                                chunks.append(chunk_text)
                        start = boundary
                   
                    if start < len(content):
                        final_chunk = content[start:].strip()
                        if len(final_chunk) >= self.config.min_chunk_size:
                            chunks.append(final_chunk)
                   
                    processed_chunks = []
                    for chunk in chunks:
                        if len(chunk) > self.config.max_chunk_size:
                            sub_chunks = self.base_splitter.split_text(chunk)
                            processed_chunks.extend(sub_chunks)
                        else:
                            processed_chunks.append(chunk)
                   
                    for i, chunk in enumerate(processed_chunks):
                        enhanced_chunks.append(Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                'chunk_index': i,
                                'total_chunks': len(processed_chunks),
                                'chunk_type': 'semantic',
                                'chunk_length': len(chunk)
                            }
                        ))
                else:
                    # Fallback to base splitter if no boundaries detected
                    enhanced_chunks.extend(self.base_splitter.split_documents([doc]))
            else:
                enhanced_chunks.extend(self.base_splitter.split_documents([doc]))
       
        return enhanced_chunks

class AdvancedReranker:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
        self.doc_vectors = None
        self.documents = []
        self.keyword_weights = {
            'financial': 1.5, 'revenue': 1.4, 'profit': 1.4, 'growth': 1.3,
            'market': 1.3, 'strategy': 1.3, 'risk': 1.2, 'performance': 1.2,
            'quarterly': 1.2, 'annual': 1.2, 'forecast': 1.3, 'outlook': 1.3
        }
   
    def fit(self, documents: List[str]):
        if not documents:
            return
        self.documents = documents
        self.doc_vectors = self.tfidf.fit_transform(documents)
   
    def calculate_similarity(self, query: str, document: str) -> float:
        try:
            query_vector = self.tfidf.transform([query])
            doc_vector = self.tfidf.transform([document])
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            
            # Apply keyword weighting
            query_lower = query.lower()
            doc_lower = document.lower()
            
            keyword_boost = 1.0
            for keyword, weight in self.keyword_weights.items():
                if keyword in query_lower and keyword in doc_lower:
                    keyword_boost *= weight
            
            return similarity * keyword_boost
        except:
            return 0.0
   
    def rerank_documents(self, query: str, documents: List[Document], scores: List[float]) -> List[Tuple[Document, float]]:
        if self.doc_vectors is None:
            return list(zip(documents, scores))
       
        doc_texts = [doc.page_content for doc in documents]
        reranked_scores = []
       
        for doc_text in doc_texts:
            similarity = self.calculate_similarity(query, doc_text)
            reranked_scores.append(similarity)
       
        # Combine original scores with semantic similarity
        combined_scores = [0.6 * orig + 0.4 * rerank for orig, rerank in zip(scores, reranked_scores)]
        
        # Sort by combined scores
        sorted_pairs = sorted(zip(documents, combined_scores), key=lambda x: x[1], reverse=True)
        return sorted_pairs

class FallbackRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._setup_fallback()
   
    def _setup_fallback(self):
        if not self.documents:
            return
        self.doc_texts = [doc.page_content.lower() for doc in self.documents]
   
    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.documents:
            return []
       
        query_lower = query.lower()
        query_words = set(query_lower.split())
       
        doc_scores = []
        for i, doc_text in enumerate(self.doc_texts):
            doc_words = set(doc_text.split())
            intersection = query_words & doc_words
            score = len(intersection) / len(query_words) if query_words else 0
            doc_scores.append((i, score))
       
        # Sort by score and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in doc_scores[:k] if score > 0]
       
        return [self.documents[idx] for idx in top_indices]

class HybridRetriever:
    def __init__(self, vector_store, documents: List[Document], config: RetrievalConfig):
        self.vector_store = vector_store
        self.documents = documents
        self.config = config
        self._setup_bm25()
        self._setup_reranker()
   
    def _setup_bm25(self):
        if not self.documents:
            return
        doc_texts = [doc.page_content for doc in self.documents]
        tokenized_docs = [doc.split() for doc in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
   
    def _setup_reranker(self):
        if self.config.enable_reranking:
            doc_texts = [doc.page_content for doc in self.documents]
            self.reranker = AdvancedReranker()
            self.reranker.fit(doc_texts)
   
    def dense_retrieval(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            return []
       
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Dense retrieval error: {e}")
            return []
   
    def sparse_retrieval(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not hasattr(self, 'bm25') or not self.bm25:
            return []
       
        try:
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top k documents
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            
            for idx in top_indices:
                if scores[idx] > -1:  # Allow slightly negative scores
                    results.append((self.documents[idx], float(scores[idx])))
            
            return results
        except Exception as e:
            print(f"Sparse retrieval error: {e}")
            return []
   
    def hybrid_search(self, query: str) -> List[Document]:
        if len(query) < self.config.rerank_min_query_length:
            # For short queries, use simple retrieval
            if self.vector_store:
                return self.vector_store.similarity_search(query, k=self.config.final_top_k)
            else:
                fallback = FallbackRetriever(self.documents)
                return fallback.search(query, k=self.config.final_top_k)
       
        # Hybrid search
        dense_results = self.dense_retrieval(query, self.config.top_k_dense)
        sparse_results = self.sparse_retrieval(query, self.config.top_k_sparse)
       
        # Combine results
        combined_results = {}
       
        # Add dense results
        for doc, score in dense_results:
            doc_id = id(doc)
            combined_results[doc_id] = {
                'doc': doc,
                'dense_score': score,
                'sparse_score': 0.0,
                'combined_score': score * self.config.dense_weight
            }
       
        # Add sparse results
        for doc, score in sparse_results:
            doc_id = id(doc)
            if doc_id in combined_results:
                combined_results[doc_id]['sparse_score'] = score
                combined_results[doc_id]['combined_score'] += score * self.config.sparse_weight
            else:
                combined_results[doc_id] = {
                    'doc': doc,
                    'dense_score': 0.0,
                    'sparse_score': score,
                    'combined_score': score * self.config.sparse_weight
                }
       
        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
       
        # Apply diversity boost if enabled
        if self.config.enable_diversity_boost:
            diverse_results = self._apply_diversity_boost(sorted_results)
        else:
            diverse_results = sorted_results
       
        # Rerank if enabled
        if self.config.enable_reranking and hasattr(self, 'reranker'):
            docs = [item['doc'] for item in diverse_results[:self.config.final_top_k]]
            scores = [item['combined_score'] for item in diverse_results[:self.config.final_top_k]]
            reranked = self.reranker.rerank_documents(query, docs, scores)
            return [doc for doc, _ in reranked]
        else:
            return [item['doc'] for item in diverse_results[:self.config.final_top_k]]
    
    def _apply_diversity_boost(self, results: List[Dict]) -> List[Dict]:
        """Apply diversity boost to avoid similar documents clustering together"""
        if len(results) <= 1:
            return results
        
        diverse_results = [results[0]]
        remaining = results[1:]
        
        while remaining and len(diverse_results) < self.config.final_top_k:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                # Calculate diversity score based on content similarity
                diversity_score = 0
                for selected in diverse_results:
                    similarity = self._calculate_content_similarity(
                        candidate['doc'].page_content,
                        selected['doc'].page_content
                    )
                    diversity_score += (1 - similarity)
                
                diversity_score /= len(diverse_results)
                
                # Combine with original score
                final_score = (0.7 * candidate['combined_score'] + 
                              0.3 * diversity_score)
                
                if final_score > best_diversity_score:
                    best_diversity_score = final_score
                    best_candidate = candidate
            
            if best_candidate:
                diverse_results.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return diverse_results
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union)
        except:
            return 0.0

class EnhancedPDFChatbot:
    def __init__(self,
                 google_api_key: str = None,
                 qdrant_url: str = None,
                 qdrant_api_key: str = None,
                 collection_name: str = "advanced_pdf_documents",
                 query_config: QueryExpansionConfig = None,
                 retrieval_config: RetrievalConfig = None,
                 chunking_config: ChunkingConfig = None,
                 response_config: ResponseConfig = None):
       
        # Use environment variables for API keys
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        elif not os.environ.get("GOOGLE_API_KEY"):
            print("Warning: No Google API key provided. Set GOOGLE_API_KEY environment variable.")
       
        # Use environment variables for Qdrant
        if not qdrant_url:
            qdrant_url = os.environ.get("QDRANT_URL")
        if not qdrant_api_key:
            qdrant_api_key = os.environ.get("QDRANT_API_KEY")
       
        self.collection_name = collection_name
        self.query_config = query_config or QueryExpansionConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.chunking_config = chunking_config or ChunkingConfig()
        self.response_config = response_config or ResponseConfig()
       
        self.context_chunker = ContextAwareChunker(self.chunking_config)
        self.documents = []
        self.hybrid_retriever = None
        self.llm_cache = LLMCache()
       
        # Initialize Qdrant client
        if qdrant_url and qdrant_api_key:
            try:
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                print(f"Connected to Qdrant")
            except Exception as e:
                print(f"Failed to connect to Qdrant: {e}")
                self.qdrant_client = None
        else:
            print("No Qdrant credentials provided. Using local fallback only.")
            self.qdrant_client = None
       
        # Initialize Google AI models
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
            print("Initialized Google AI models")
        except Exception as e:
            print(f"Failed to initialize Google AI models: {e}")
            raise
       
        self.vector_store = None
       
        # Enhanced system prompt with comprehensive instructions
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert business document analyst and financial advisor with deep expertise in analyzing PDF documents, financial reports, business plans, and corporate documents. Your role is to provide precise, well-researched, and professionally formatted answers based solely on the provided document context.

                **CORE MISSION**
                Transform complex business documents into clear, actionable insights that enable informed decision-making. You are a trusted advisor who provides accurate, comprehensive, and professionally formatted responses.

                📋 **RESPONSE STRUCTURE & FORMATTING**

                **Executive Summary** (2-3 sentences)
                • Brief overview of key findings or answer to the question
                • Highlight the most important numbers, trends, or insights
                • Use bold formatting for emphasis: **Key Metric: $1.2M**

                **Detailed Analysis** (main content)
                • Use bullet points (•) for lists and key findings
                • Include specific numbers, percentages, and dates with proper formatting
                • Reference document sections: [Section X], [Page Y], [Paragraph Z]
                • Use numbered lists (1., 2., 3.) for sequential information
                • Highlight important trends, comparisons, or relationships
                • Use tables for comparative data when appropriate

                **Key Takeaways** (if applicable)
                • Summarize the 3-5 most important points
                • Note any limitations or missing information
                • Provide actionable insights

                **Source References**
                • List all document sections referenced
                • Include page numbers when available
                • Format: [Section: Financial Performance, Page: 15-18]

                💼 **FINANCIAL ANALYSIS GUIDELINES**

                **Revenue & Sales Analysis:**
                • Report exact figures with proper formatting: **Revenue: $2.5M (+15% YoY)**
                • Include growth rates, year-over-year comparisons
                • Highlight seasonal patterns or trends
                • Compare to industry benchmarks when available

                **Profitability Metrics:**
                • Gross margin: **Gross Margin: 45.2%**
                • Operating margin: **Operating Margin: 12.8%**
                • Net profit margin: **Net Profit Margin: 8.5%**
                • EBITDA: **EBITDA: $850K**
                • Include trend analysis and comparisons

                **Cash Flow Analysis:**
                • Distinguish between operating, investing, and financing cash flows
                • Highlight cash generation vs. cash consumption
                • Note working capital changes
                • Format: **Operating Cash Flow: $1.2M, Investing: -$500K**

                **Balance Sheet Review:**
                • Report assets, liabilities, and equity with specific figures
                • Calculate key ratios: Current Ratio, Debt-to-Equity, ROE
                • Highlight significant changes or trends
                • Format: **Total Assets: $10.5M, Current Ratio: 2.1x**

                **Financial Ratios & Metrics:**
                • Calculate and explain key financial ratios
                • Include industry comparisons when available
                • Highlight trends and implications
                • Format: **ROE: 15.2%, Industry Avg: 12.1%**

                🏢 **BUSINESS STRATEGY ANALYSIS**

                **Market Position & Competitive Analysis:**
                • Market share and competitive landscape
                • Competitive advantages and differentiators
                • Market trends and opportunities
                • Format: **Market Share: 12.5%, #3 in segment**

                **Growth Initiatives & Strategy:**
                • Expansion plans and new market entry
                • Product development and innovation
                • Strategic partnerships and acquisitions
                • Format: **New Market Entry: Asia-Pacific, Target: 2024 Q3**

                **Risk Assessment:**
                • Specific risks with mitigation strategies
                • Regulatory and compliance risks
                • Market and operational risks
                • Format: **Risk Level: Medium, Mitigation: Diversified supply chain**

                **Operational Performance:**
                • Efficiency metrics and productivity indicators
                • Capacity utilization and operational trends
                • Quality metrics and customer satisfaction
                • Format: **Capacity Utilization: 85%, Efficiency: +12% YoY**

                **DOCUMENT-SPECIFIC ANALYSIS**

                **Annual Reports:**
                • Financial performance highlights
                • Strategic initiatives and achievements
                • Risk factors and outlook
                • Management discussion and analysis

                **Business Plans:**
                • Market opportunity and size
                • Competitive advantage and positioning
                • Financial projections and assumptions
                • Go-to-market strategy

                **Legal Documents:**
                • Key terms and obligations
                • Important dates and deadlines
                • Risk factors and compliance requirements
                • Financial implications

                **Technical Reports:**
                • Methodology and approach
                • Key findings and conclusions
                • Implications and recommendations
                • Technical specifications and requirements

                **Regulatory Filings:**
                • Compliance status and requirements
                • Material disclosures and information
                • Regulatory risks and obligations
                • Financial impact of regulations

                🎨 **FORMATTING REQUIREMENTS**

                **Text Formatting:**
                • Use **bold** for emphasis on key terms, numbers, and metrics
                • Use *italics* for document references and citations
                • Use bullet points (•) for lists and key findings
                • Use numbered lists (1., 2., 3.) for sequential information
                • Use section headers (##) for organizing complex responses

                **Data Presentation:**
                • Format numbers consistently: **$1,250,000** or **1.25M**
                • Use percentages: **15.2%** or **+15.2% YoY**
                • Include units: **Revenue: $2.5M**, **Growth: 12.5%**
                • Use tables for comparative data when appropriate

                **Visual Elements:**
                • Use section headers for organizing complex responses
                • Use horizontal lines (---) to separate major sections
                • Use indentation for sub-points and details
                • Maintain consistent spacing and formatting

                **QUALITY STANDARDS**

                **Completeness:**
                • Address all parts of the question thoroughly
                • Provide comprehensive analysis when possible
                • Include relevant context and background information

                **Clarity:**
                • Use clear, professional language suitable for business audiences
                • Avoid jargon unless necessary and explain technical terms
                • Structure information logically and coherently

                **Relevance:**
                • Focus on information directly related to the query
                • Prioritize the most important and actionable insights
                • Avoid tangential information unless specifically requested

                **Accuracy:**
                • Double-check all numbers, dates, and facts against the context
                • Use exact terminology as it appears in the document
                • Verify calculations and percentages

                **Objectivity:**
                • Present information neutrally without personal opinions
                • Provide balanced analysis of pros and cons
                • Acknowledge limitations and uncertainties

                **WHEN INFORMATION IS MISSING**

                **Clear Communication:**
                • State explicitly: "This specific information is not available in the provided documents"
                • Suggest related information that is available
                • Note what additional documents might contain the missing information

                **Alternative Approaches:**
                • Provide estimates based on available data (clearly marked as estimates)
                • Suggest proxy metrics or alternative measures
                • Recommend additional research or documentation needed

                **Professional Handling:**
                • Don't make assumptions or invent information
                • Be transparent about limitations
                • Offer to help refine the question or suggest alternative approaches

                **QUERY UNDERSTANDING & RESPONSE STRATEGY**

                **Question Types:**
                • **Financial Questions**: Focus on numbers, trends, ratios, and comparisons
                • **Strategic Questions**: Emphasize plans, initiatives, market position, and competitive analysis
                • **Operational Questions**: Highlight processes, efficiency, performance metrics, and operational trends
                • **Risk Questions**: Identify specific risks, mitigation strategies, and impact assessment
                • **Comparative Questions**: Provide side-by-side analysis with clear comparisons and contrasts

                **Response Depth:**
                • **High-level questions**: Provide executive summary with key highlights
                • **Detailed questions**: Include comprehensive analysis with specific data points
                • **Technical questions**: Use precise terminology and include methodology when relevant
                • **Trend questions**: Focus on patterns, changes over time, and future implications

                **Context Awareness:**
                • Consider the document type and industry context
                • Adapt language and focus based on the audience (executive vs. technical)
                • Prioritize information based on business relevance and impact

                **PERFORMANCE METRICS & REPORTING**

                **Key Performance Indicators (KPIs):**
                • Revenue growth and profitability metrics
                • Operational efficiency and productivity measures
                • Market performance and competitive positioning
                • Customer satisfaction and retention metrics

                **Trend Analysis:**
                • Year-over-year comparisons and growth rates
                • Seasonal patterns and cyclical trends
                • Long-term performance trajectories
                • Benchmark comparisons and industry analysis

                **Forecasting & Projections:**
                • Future outlook and projections when available
                • Assumptions and methodology behind forecasts
                • Risk factors affecting projections
                • Sensitivity analysis and scenario planning

                CONTEXT FROM DOCUMENT:
                {context}

                EXPANDED QUERY CONTEXT:
                {expanded_query}

                Remember: You are a trusted business advisor providing professional, accurate, and immediately actionable insights for business decision-making. Your responses should be comprehensive, well-structured, and professionally formatted to enable informed decision-making.
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
   
    def upload_pdf_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            chunks = self.context_chunker.chunk_with_context(docs)
            self.documents = chunks

            if not self.vector_store and self.qdrant_client:
                self._setup_vector_store()

            if self.vector_store:
                self.vector_store.add_documents(chunks)

            return f"{filename} uploaded and indexed with {len(chunks)} chunks."
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _setup_vector_store(self):
        if not self.qdrant_client:
            print("No Qdrant client available, using fallback only")
            return
       
        try:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' exists")
            except UnexpectedResponse as e:
                if "doesn't exist" in str(e):
                    print(f"📁 Creating collection: {self.collection_name}")
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=768,
                            distance=Distance.COSINE
                        )
                    )
                    print(f"Collection created")
                else:
                    raise e
       
        except Exception as e:
            print(f"Collection setup error: {e}")
            return
       
        try:
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            print(f"Vector store initialized")
        except Exception as e:
            print(f"Vector store error: {e}")
            self.vector_store = None

    def _expand_query(self, query: str) -> Dict[str, Any]:
        if not self.query_config.enable_llm_expansion or len(query) <= self.query_config.expansion_threshold:
            return {
                "original_query": query,
                "expanded_terms": [],
                "expanded_query": query,
                "expansion_used": False
            }
       
        cache_key = hashlib.md5(query.encode()).hexdigest()
       
        if self.query_config.cache_expansions:
            cached_result = self.llm_cache.get(cache_key)
            if cached_result:
                expanded_terms = cached_result.split(',')
                return {
                    "original_query": query,
                    "expanded_terms": expanded_terms,
                    "expanded_query": f"{query} {' '.join(expanded_terms)}",
                    "expansion_used": True
                }
       
        try:
            expansion_prompt = f"""
            Analyze this business query and suggest {self.query_config.max_expanded_terms} related terms that might appear in business documents:
            Query: "{query}"

            Consider:
            1. Financial terminology (revenue, profit, EBITDA, cash flow, etc.)
            2. Business processes and operations
            3. Industry-specific language and jargon
            4. Regulatory and compliance terms
            5. Strategic and management terms
            6. Market and competitive terms

            Return only the terms, comma-separated:
            """

            llm_response = self.llm.invoke(expansion_prompt)
            expanded_terms = [term.strip() for term in llm_response.split(',') if term.strip()]
            expanded_terms = expanded_terms[:self.query_config.max_expanded_terms]
           
            if self.query_config.cache_expansions:
                self.llm_cache.set(cache_key, ','.join(expanded_terms))
           
            return {
                "original_query": query,
                "expanded_terms": expanded_terms,
                "expanded_query": f"{query} {' '.join(expanded_terms)}",
                "expansion_used": True
            }
        except Exception as e:
            print(f"LLM expansion error: {e}")
            return {
                "original_query": query,
                "expanded_terms": [],
                "expanded_query": query,
                "expansion_used": False
            }

    def query(self, messages: List[BaseMessage]) -> str:
        if not self.documents:
            raise ValueError("No documents found. Please upload and process a PDF first.")

        if not self.hybrid_retriever:
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=self.documents,
                config=self.retrieval_config
            )

        latest_human_msg = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        if not latest_human_msg:
            raise ValueError("No user message found")

        expanded_info = self._expand_query(latest_human_msg)
        expanded_query = expanded_info["expanded_query"]

        retrieved_docs = self.hybrid_retriever.hybrid_search(expanded_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        chain = self.prompt_template | self.llm | StrOutputParser()

        response = chain.invoke({
            "messages": messages,
            "context": context,
            "expanded_query": expanded_query
        })

        return response

    def clear_collection(self):
        if self.qdrant_client:
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                self.documents = []
                self.hybrid_retriever = None
                self.llm_cache.clear()
                return f"Collection {self.collection_name} cleared successfully"
            except Exception as e:
                return f"Error clearing collection: {e}"
        return "No Qdrant client available"

    def get_stats(self):
        cache_stats = self.llm_cache.get_stats()
        return {
            "documents_count": len(self.documents),
            "collection_name": self.collection_name,
            "cache_stats": cache_stats,
            "vector_store_available": self.vector_store is not None,
            "qdrant_available": self.qdrant_client is not None,
            "retrieval_config": {
                "hybrid_search": self.retrieval_config.enable_hybrid_search,
                "dense_weight": self.retrieval_config.dense_weight,
                "sparse_weight": self.retrieval_config.sparse_weight,
                "reranking": self.retrieval_config.enable_reranking
            },
            "chunking_config": {
                "chunk_size": self.chunking_config.chunk_size,
                "semantic_chunking": self.chunking_config.enable_semantic_chunking,
                "boundary_detection": self.chunking_config.enable_boundary_detection
            }
        }

