import os
import json
import hashlib
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

class FreeQueryExpansionCache:
    """Free version using local storage and simple similarity"""
    
    def __init__(self, db_path: str = "query_cache.db"):
        self.db_path = db_path
        self.init_database()
        self.load_synonym_database()
        
        # Load free sentence transformer for embeddings
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Simple rule-based expansion patterns
        self.expansion_patterns = {
            'financial': ['revenue', 'sales', 'profit', 'income', 'earnings', 'cost', 'expense'],
            'temporal': ['quarterly', 'annual', 'monthly', 'yearly', 'q1', 'q2', 'q3', 'q4'],
            'business': ['employee', 'customer', 'client', 'project', 'product', 'service'],
            'metrics': ['performance', 'growth', 'efficiency', 'quality', 'roi', 'kpi']
        }
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_expansions (
                    query_hash TEXT PRIMARY KEY,
                    original_query TEXT,
                    expanded_terms TEXT,
                    created_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
    
    def load_synonym_database(self):
        """Enhanced synonym database"""
        self.synonyms = {
            # Financial terms
            "revenue": ["sales", "income", "earnings", "turnover", "net sales", "gross sales", "receipts"],
            "profit": ["net income", "earnings", "bottom line", "net profit", "operating profit", "margin"],
            "cost": ["expense", "expenditure", "spending", "outlay", "charges", "fees"],
            "budget": ["allocation", "funds", "financial plan", "spending plan", "allowance"],
            "loss": ["deficit", "shortfall", "negative", "red ink", "losses"],
            
            # Business operations
            "employee": ["staff", "worker", "personnel", "team member", "workforce", "human resources"],
            "customer": ["client", "consumer", "buyer", "patron", "user", "purchaser"],
            "product": ["item", "goods", "merchandise", "offering", "solution", "commodity"],
            "service": ["offering", "solution", "support", "assistance", "help"],
            "project": ["initiative", "program", "task", "assignment", "undertaking", "venture"],
            "company": ["organization", "business", "firm", "corporation", "enterprise"],
            
            # Time periods
            "quarterly": ["q1", "q2", "q3", "q4", "quarter", "three months", "3 months"],
            "annual": ["yearly", "year", "12 months", "fiscal year", "per year"],
            "monthly": ["month", "30 days", "per month", "each month"],
            "daily": ["day", "per day", "each day", "24 hours"],
            
            # Performance metrics
            "performance": ["results", "outcome", "achievement", "success", "metrics", "kpi"],
            "growth": ["increase", "expansion", "development", "progress", "improvement", "rise"],
            "efficiency": ["productivity", "optimization", "effectiveness", "streamlining"],
            "quality": ["standard", "excellence", "reliability", "consistency", "grade"],
            
            # Common abbreviations
            "roi": ["return on investment", "return", "profitability", "investment return"],
            "kpi": ["key performance indicator", "metric", "measure", "indicator"],
            "ceo": ["chief executive officer", "president", "leader", "head"],
            "cfo": ["chief financial officer", "finance head", "financial officer"],
            "hr": ["human resources", "personnel", "people operations", "staff management"]
        }
    
    def get_query_hash(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """Expand using synonym database"""
        expanded_terms = [query]
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Direct synonym matching
        for word in query_words:
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
        
        # Pattern-based expansion
        for pattern_type, terms in self.expansion_patterns.items():
            for term in terms:
                if term in query_lower:
                    expanded_terms.extend([t for t in terms if t != term])
                    break
        
        # Remove duplicates
        return list(dict.fromkeys(expanded_terms))
    
    def expand_with_similarity(self, query: str) -> List[str]:
        """Use sentence similarity for expansion"""
        try:
            # Get all synonym values for similarity matching
            all_terms = []
            for synonyms in self.synonyms.values():
                all_terms.extend(synonyms)
            
            # Add query to compare
            all_terms.append(query)
            
            # Get embeddings
            embeddings = self.sentence_model.encode(all_terms)
            query_embedding = embeddings[-1].reshape(1, -1)
            term_embeddings = embeddings[:-1]
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, term_embeddings)[0]
            
            # Get top similar terms (threshold 0.5)
            similar_terms = [all_terms[i] for i, sim in enumerate(similarities) if sim > 0.5]
            
            # Combine with original query
            expanded_terms = [query] + similar_terms
            return list(dict.fromkeys(expanded_terms))
            
        except Exception as e:
            print(f"Similarity expansion failed: {e}")
            return [query]
    
    def get_cached_expansion(self, query: str) -> Optional[List[str]]:
        """Get cached expansion"""
        query_hash = self.get_query_hash(query)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT expanded_terms, created_at FROM query_expansions 
                WHERE query_hash = ?
            ''', (query_hash,))
            
            result = cursor.fetchone()
            if result:
                expanded_terms, created_at = result
                created_date = datetime.fromisoformat(created_at)
                
                # Check if cache is valid (30 days)
                if datetime.now() - created_date < timedelta(days=30):
                    cursor.execute('''
                        UPDATE query_expansions 
                        SET access_count = access_count + 1 
                        WHERE query_hash = ?
                    ''', (query_hash,))
                    conn.commit()
                    return json.loads(expanded_terms)
        
        return None
    
    def cache_expansion(self, query: str, expanded_terms: List[str]):
        """Cache expansion results"""
        query_hash = self.get_query_hash(query)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO query_expansions 
                (query_hash, original_query, expanded_terms, created_at, access_count)
                VALUES (?, ?, ?, ?, 1)
            ''', (query_hash, query, json.dumps(expanded_terms), datetime.now().isoformat()))
            conn.commit()

class FreePDFChatbot:
    """Free PDF chatbot using Hugging Face models"""
    
    def __init__(self):
        print("Initializing free PDF chatbot...")
        
        # Load free models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize QA pipeline with a free model
        print("Loading QA model...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad"
        )
        
        self.documents = []
        self.document_embeddings = None
        self.vectorstore = None
        self.query_cache = FreeQueryExpansionCache()
        self.expansion_stats = {"synonym_hits": 0, "similarity_hits": 0, "cache_hits": 0}
    
    def load_pdf(self, pdf_path: str) -> bool:
        """Load PDF using PyPDF2 (free)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Split into chunks
                chunks = self.split_text(text)
                self.documents = chunks
                
                # Create embeddings
                print("Creating embeddings...")
                self.document_embeddings = self.sentence_model.encode(chunks)
                
                # Create FAISS index
                dimension = self.document_embeddings.shape[1]
                self.vectorstore = faiss.IndexFlatL2(dimension)
                self.vectorstore.add(self.document_embeddings.astype('float32'))
                
                print(f"Loaded PDF with {len(chunks)} chunks")
                return True
                
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False
    
    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple text splitting"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk]
    
    def expand_query_hybrid(self, query: str) -> List[str]:
        """Free hybrid expansion"""
        # Try cache first
        cached = self.query_cache.get_cached_expansion(query)
        if cached:
            self.expansion_stats["cache_hits"] += 1
            return cached
        
        # Try synonyms
        synonym_expansion = self.query_cache.expand_with_synonyms(query)
        if len(synonym_expansion) > 1:
            self.expansion_stats["synonym_hits"] += 1
            self.query_cache.cache_expansion(query, synonym_expansion)
            return synonym_expansion
        
        # Try similarity-based expansion
        similarity_expansion = self.query_cache.expand_with_similarity(query)
        if len(similarity_expansion) > 1:
            self.expansion_stats["similarity_hits"] += 1
            self.query_cache.cache_expansion(query, similarity_expansion)
            return similarity_expansion
        
        return [query]
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents"""
        if not self.vectorstore:
            return []
        
        # Expand query
        expanded_terms = self.expand_query_hybrid(query)
        expanded_query = " ".join(expanded_terms)
        
        # Get query embedding
        query_embedding = self.sentence_model.encode([expanded_query])
        
        # Search
        distances, indices = self.vectorstore.search(query_embedding.astype('float32'), k)
        
        # Return relevant documents
        relevant_docs = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return relevant_docs
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer question using free models"""
        if not self.documents:
            return {
                "answer": "No PDF loaded. Please upload a PDF first.",
                "expanded_terms": [],
                "expansion_method": "none"
            }
        
        try:
            # Track expansion method
            original_stats = self.expansion_stats.copy()
            
            # Get relevant documents
            relevant_docs = self.retrieve_relevant_docs(question)
            
            if not relevant_docs:
                return {
                    "answer": "No relevant information found in the document.",
                    "expanded_terms": [],
                    "expansion_method": "none"
                }
            
            # Combine documents for context
            context = " ".join(relevant_docs)
            
            # Truncate context if too long (model limitation)
            if len(context) > 4000:
                context = context[:4000]
            
            # Get answer using QA pipeline
            result = self.qa_pipeline(question=question, context=context)
            
            # Determine expansion method
            expansion_method = "none"
            if self.expansion_stats["cache_hits"] > original_stats["cache_hits"]:
                expansion_method = "cache"
            elif self.expansion_stats["synonym_hits"] > original_stats["synonym_hits"]:
                expansion_method = "synonym_database"
            elif self.expansion_stats["similarity_hits"] > original_stats["similarity_hits"]:
                expansion_method = "similarity"
            
            # Get expanded terms
            expanded_terms = self.expand_query_hybrid(question)
            
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "expanded_terms": expanded_terms,
                "expansion_method": expansion_method,
                "stats": self.expansion_stats.copy()
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "expanded_terms": [],
                "expansion_method": "error"
            }
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total_queries = sum(self.expansion_stats.values())
        
        return {
            "session_stats": self.expansion_stats,
            "cost_savings": {
                "synonym_db_usage": f"{(self.expansion_stats['synonym_hits'] / max(total_queries, 1)) * 100:.1f}%",
                "similarity_usage": f"{(self.expansion_stats['similarity_hits'] / max(total_queries, 1)) * 100:.1f}%",
                "cache_usage": f"{(self.expansion_stats['cache_hits'] / max(total_queries, 1)) * 100:.1f}%"
            },
            "total_queries": total_queries
        }