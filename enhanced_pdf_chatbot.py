import os
import tempfile
import json
import hashlib
import time
import re
import io
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

# Enhanced document processing imports
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import pandas as pd
import tabula
import camelot
import spacy
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import textstat

# Web crawling imports
try:
    from web_crawler import WebContentProcessor
    WEB_CRAWLING_AVAILABLE = True
except ImportError:
    WEB_CRAWLING_AVAILABLE = False
    print("Warning: Web crawling dependencies not available. Install with: pip install requests beautifulsoup4 selenium webdriver-manager")

@dataclass
class EnhancedQueryConfig:
    enable_llm_expansion: bool = True
    max_expanded_terms: int = 5
    cache_expansions: bool = True
    expansion_threshold: int = 8
    enable_synonym_expansion: bool = True
    enable_context_aware_expansion: bool = True
    enable_spelling_correction: bool = True
    enable_query_classification: bool = True

@dataclass
class EnhancedRetrievalConfig:
    enable_hybrid_search: bool = True
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    top_k_dense: int = 10
    top_k_sparse: int = 8
    final_top_k: int = 8
    enable_reranking: bool = True
    rerank_threshold: float = 0.25
    rerank_min_query_length: int = 10
    enable_diversity_boost: bool = True
    diversity_threshold: float = 0.3
    enable_source_filtering: bool = True
    enable_relevance_scoring: bool = True

@dataclass
class EnhancedChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 200
    enable_semantic_chunking: bool = True
    enable_boundary_detection: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    enable_hierarchical_chunking: bool = True
    enable_table_preservation: bool = True
    enable_image_captioning: bool = True

@dataclass
class EnhancedResponseConfig:
    enable_structured_output: bool = True
    enable_source_citations: bool = True
    enable_confidence_scoring: bool = True
    max_response_length: int = 2500
    enable_follow_up_suggestions: bool = True
    enable_visual_formatting: bool = True
    enable_detailed_analysis: bool = True
    enable_key_insights: bool = True

class DocumentProcessor:
    """Enhanced document processor for handling PDFs with tables, images, and complex layouts"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('punkt')
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF using multiple methods"""
        tables = []
        
        try:
            # Method 1: Using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Ensure table has data
                            tables.append({
                                'page': page_num + 1,
                                'table_num': table_num + 1,
                                'method': 'pdfplumber',
                                'data': table,
                                'text': self._table_to_text(table)
                            })
        except Exception as e:
            print(f"pdfplumber extraction error: {e}")
        
        try:
            # Method 2: Using tabula
            tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            for page_num, page_tables in enumerate(tabula_tables):
                for table_num, table in enumerate(page_tables):
                    if not table.empty:
                        tables.append({
                            'page': page_num + 1,
                            'table_num': table_num + 1,
                            'method': 'tabula',
                            'data': table.to_dict('records'),
                            'text': self._dataframe_to_text(table)
                        })
        except Exception as e:
            print(f"tabula extraction error: {e}")
        
        return tables
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract images and perform OCR"""
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Perform OCR on the image
                            ocr_text = self._perform_ocr_on_image(img_data)
                            
                            images.append({
                                'page': page_num + 1,
                                'image_num': img_index + 1,
                                'ocr_text': ocr_text,
                                'description': self._generate_image_description(ocr_text)
                            })
                        
                        pix = None
                    except Exception as e:
                        print(f"Image processing error: {e}")
                        continue
            
            doc.close()
        except Exception as e:
            print(f"Image extraction error: {e}")
        
        return images
    
    def _perform_ocr_on_image(self, img_data: bytes) -> str:
        """Perform OCR on image data"""
        try:
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Preprocess image for better OCR
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to get better text recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def _generate_image_description(self, ocr_text: str) -> str:
        """Generate description of image based on OCR text"""
        if not ocr_text:
            return "Image with no readable text"
        
        # Analyze OCR text to determine image type
        text_lower = ocr_text.lower()
        
        if any(word in text_lower for word in ['chart', 'graph', 'plot', 'figure']):
            return f"Chart/Graph: {ocr_text[:100]}..."
        elif any(word in text_lower for word in ['table', 'data', 'summary']):
            return f"Table/Data: {ocr_text[:100]}..."
        elif any(word in text_lower for word in ['logo', 'brand', 'company']):
            return f"Logo/Brand: {ocr_text[:100]}..."
        else:
            return f"Image with text: {ocr_text[:100]}..."
    
    def _table_to_text(self, table: List[List]) -> str:
        """Convert table data to readable text"""
        if not table:
            return ""
        
        text_parts = []
        for row in table:
            row_text = " | ".join([str(cell) if cell else "" for cell in row])
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text"""
        return df.to_string(index=False)

class SpellingCorrector:
    """Enhanced spelling correction using NLTK and custom rules"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.common_business_terms = {
            'ebitda', 'roi', 'kpi', 'cfo', 'ceo', 'cto', 'cmo', 'hr', 'qa',
            'api', 'ui', 'ux', 'saas', 'paas', 'iaas', 'b2b', 'b2c', 'c2c',
            'ipo', 'm&a', 'pe', 'pb', 'ev', 'dcf', 'wacc', 'irr', 'npv'
        }
    
    def correct_spelling(self, text: str) -> str:
        """Correct spelling in text while preserving business terms"""
        words = word_tokenize(text)
        corrected_words = []
        
        for word in words:
            # Skip if it's a business term or proper noun
            if (word.lower() in self.common_business_terms or 
                word[0].isupper() or 
                any(char.isdigit() for char in word)):
                corrected_words.append(word)
                continue
            
            # Check if word exists in WordNet
            if wordnet.synsets(word):
                corrected_words.append(word)
            else:
                # Find similar words
                suggestions = self._get_suggestions(word)
                if suggestions:
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _get_suggestions(self, word: str) -> List[str]:
        """Get spelling suggestions for a word"""
        suggestions = []
        
        # Check for common misspellings
        common_misspellings = {
            'revenue': ['revenu', 'revnue'],
            'profit': ['profitt', 'proffit'],
            'financial': ['finacial', 'finanacial'],
            'business': ['bussiness', 'buisness'],
            'management': ['managment', 'mangement'],
            'strategy': ['stratagy', 'strategie'],
            'performance': ['performence', 'perfomance'],
            'analysis': ['analisis', 'anaylsis'],
            'quarter': ['quater', 'quaterly'],
            'annual': ['anual', 'annul']
        }
        
        word_lower = word.lower()
        if word_lower in common_misspellings:
            return common_misspellings[word_lower]
        
        # Use WordNet for general suggestions
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    suggestions.append(lemma.name())
        
        return list(set(suggestions))[:3]

class EnhancedSourceTracker:
    """Track and manage document sources for proper attribution"""
    
    def __init__(self):
        self.document_sources = {}
        self.chunk_sources = {}
        self.current_document = None
    
    def add_document(self, filename: str, document_id: str):
        """Add a new document to tracking"""
        self.document_sources[document_id] = {
            'filename': filename,
            'upload_time': time.time(),
            'chunks': [],
            'tables': [],
            'images': []
        }
        self.current_document = document_id
    
    def add_chunk(self, chunk_id: str, page_num: int, section: str = None):
        """Add a chunk with source information"""
        if self.current_document:
            chunk_info = {
                'chunk_id': chunk_id,
                'page_num': page_num,
                'section': section,
                'document_id': self.current_document
            }
            self.chunk_sources[chunk_id] = chunk_info
            self.document_sources[self.current_document]['chunks'].append(chunk_info)
    
    def get_source_info(self, chunk_id: str) -> Dict:
        """Get source information for a chunk"""
        if chunk_id in self.chunk_sources:
            chunk_info = self.chunk_sources[chunk_id]
            doc_info = self.document_sources.get(chunk_info['document_id'], {})
            return {
                'filename': doc_info.get('filename', 'Unknown'),
                'page': chunk_info.get('page_num', 'Unknown'),
                'section': chunk_info.get('section', 'Unknown'),
                'upload_time': doc_info.get('upload_time', 'Unknown')
            }
        return {'filename': 'Unknown', 'page': 'Unknown', 'section': 'Unknown'}

class EnhancedPDFChatbot:
    """Enhanced PDF chatbot with better document understanding and source attribution"""
    
    def __init__(self,
                 google_api_key: str = None,
                 qdrant_url: str = None,
                 qdrant_api_key: str = None,
                 collection_name: str = "enhanced_pdf_documents",
                 query_config: EnhancedQueryConfig = None,
                 retrieval_config: EnhancedRetrievalConfig = None,
                 chunking_config: EnhancedChunkingConfig = None,
                 response_config: EnhancedResponseConfig = None):
        
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
        self.query_config = query_config or EnhancedQueryConfig()
        self.retrieval_config = retrieval_config or EnhancedRetrievalConfig()
        self.chunking_config = chunking_config or EnhancedChunkingConfig()
        self.response_config = response_config or EnhancedResponseConfig()
        
        # Initialize enhanced components
        self.document_processor = DocumentProcessor()
        self.spelling_corrector = SpellingCorrector()
        self.source_tracker = EnhancedSourceTracker()
        
        self.documents = []
        self.hybrid_retriever = None
        self.llm_cache = LLMCache()
        
        # Initialize web crawler if available
        self.web_processor = None
        if WEB_CRAWLING_AVAILABLE:
            try:
                self.web_processor = WebContentProcessor()
                print("Web crawling capabilities enabled")
            except Exception as e:
                print(f"Web crawling initialization failed: {e}")
                self.web_processor = None
        
        # Initialize Qdrant client
        if qdrant_url and qdrant_api_key:
            try:
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                print("Connected to Qdrant")
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
        
        # Enhanced system prompt for better responses
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert document and web content analyst with advanced capabilities in analyzing PDFs, websites, charts, tables, and various types of content. Your responses must be:

**CONVERSATION CONTEXT AWARENESS:**
- Pay attention to the conversation history and context
- If the user is asking for specific lists or names, provide ONLY those
- If the user asks for "names" or "list", focus on extracting and listing the specific entities
- Maintain focus on the current question without repeating previous information unless specifically asked

**RESPONSE STRUCTURE:**
1. **Direct Answer** (2-3 sentences) - Provide a clear, focused answer to the question
2. **Detailed Analysis** - Use bullet points and structured format
3. **Source Citations** - Always cite specific sources using the provided source information
4. **Key Insights** - Highlight the most important findings
5. **Confidence Level** - Indicate your confidence in the answer

**CONTENT ANALYSIS CAPABILITIES:**
- **Web Content**: Extract and analyze information from websites, articles, and online sources
- **Tables**: Extract and analyze tabular data with proper formatting
- **Charts/Graphs**: Interpret visual data and trends
- **Lists and Data**: Extract names, locations, capacities, and other structured information
- **Text Analysis**: Identify key themes, entities, and relationships

**FORMATTING REQUIREMENTS:**
- Use **bold** for key metrics and numbers
- Use bullet points (•) for lists
- Use tables for comparative data
- Include source URLs and document references
- Keep responses focused and relevant

**SOURCE ATTRIBUTION:**
Always cite your sources using the provided source information in the SOURCE INFORMATION section below.

**CONFIDENCE SCORING:**
- High (90-100%): Clear, specific data available
- Medium (70-89%): Good data with some gaps
- Low (50-69%): Limited or unclear information

**SPECIAL INSTRUCTIONS FOR LISTING QUESTIONS:**
- When asked for "names" or "list", provide ONLY the requested information in a clear, organized format
- Do not include explanations unless specifically requested
- Focus on extracting the specific entities mentioned in the documents
- Use bullet points or numbered lists for clarity
- For sports/arena questions, extract team names, arena names, locations, and capacities

**WEB CONTENT ANALYSIS:**
- Extract information from website content including titles, lists, and structured data
- Identify entities like team names, venue names, locations, and numerical data
- Present information in organized formats like tables or lists when requested

CONTEXT FROM DOCUMENT:
{context}

EXPANDED QUERY:
{expanded_query}

SOURCE INFORMATION:
{source_info}

CONVERSATION CONTEXT:
{conversation_context}

Remember: Provide focused, accurate answers with proper source attribution. If information is not available, clearly state this and suggest what additional documents might help.
"""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def upload_pdf_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """Enhanced PDF upload with table and image extraction"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Store PDF file for later serving
            pdf_dir = os.path.join(tempfile.gettempdir(), 'rag_pdfs')
            os.makedirs(pdf_dir, exist_ok=True)
            stored_pdf_path = os.path.join(pdf_dir, filename)
            
            # Copy PDF to storage directory
            with open(stored_pdf_path, 'wb') as f:
                f.write(file_bytes)
            
            # Generate unique document ID
            document_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()
            self.source_tracker.add_document(filename, document_id)
            
            # Extract tables and images
            tables = self.document_processor.extract_tables_from_pdf(tmp_path)
            images = self.document_processor.extract_images_from_pdf(tmp_path)
            
            # Load and process text
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Enhanced chunking with source tracking
            chunks = self._enhanced_chunking(docs, document_id, tables, images)
            self.documents = chunks

            if not self.vector_store and self.qdrant_client:
                self._setup_vector_store()

            if self.vector_store:
                self.vector_store.add_documents(chunks)

            # Reinitialize hybrid retriever with new documents
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=self.documents,
                config=self.retrieval_config
            )

            return f"{filename} uploaded and indexed with {len(chunks)} chunks, {len(tables)} tables, and {len(images)} images."
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def _enhanced_chunking(self, docs: List[Document], document_id: str, tables: List[Dict], images: List[Dict]) -> List[Document]:
        """Enhanced chunking with table and image integration"""
        enhanced_chunks = []
        
        for doc in docs:
            page_num = doc.metadata.get('page', 0)
            
            # Add table data to relevant pages
            page_tables = [t for t in tables if t['page'] == page_num + 1]
            table_text = ""
            for table in page_tables:
                table_text += f"\n\nTABLE {table['table_num']}:\n{table['text']}\n"
            
            # Add image data to relevant pages
            page_images = [img for img in images if img['page'] == page_num + 1]
            image_text = ""
            for img in page_images:
                image_text += f"\n\nIMAGE {img['image_num']}: {img['description']}\n"
                if img['ocr_text']:
                    image_text += f"OCR Text: {img['ocr_text']}\n"
            
            # Combine text with tables and images
            enhanced_content = doc.page_content + table_text + image_text
            
            # Create enhanced document
            enhanced_doc = Document(
                page_content=enhanced_content,
                metadata={
                    **doc.metadata,
                    'document_id': document_id,
                    'has_tables': len(page_tables) > 0,
                    'has_images': len(page_images) > 0,
                    'table_count': len(page_tables),
                    'image_count': len(page_images)
                }
            )
            
            enhanced_chunks.append(enhanced_doc)
            
            # Track source information
            chunk_id = hashlib.md5(enhanced_content.encode()).hexdigest()
            self.source_tracker.add_chunk(chunk_id, page_num + 1)
        
        return enhanced_chunks
    
    def query(self, messages: List[BaseMessage]) -> str:
        """Enhanced query processing with spelling correction and source attribution"""
        if not self.documents:
            raise ValueError("No documents found. Please upload a PDF or add a website first.")

        if not self.hybrid_retriever:
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=self.documents,
                config=self.retrieval_config
            )

        latest_human_msg = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        if not latest_human_msg:
            raise ValueError("No user message found")

        # Enhanced context analysis from conversation history
        conversation_context = self._analyze_conversation_context(messages)
        
        # Spelling correction
        if self.query_config.enable_spelling_correction:
            corrected_query = self.spelling_corrector.correct_spelling(latest_human_msg)
            if corrected_query != latest_human_msg:
                print(f"Query corrected: '{latest_human_msg}' -> '{corrected_query}'")
                latest_human_msg = corrected_query

        # Enhanced query expansion with conversation context
        expanded_info = self._expand_query_with_context(latest_human_msg, conversation_context)
        expanded_query = expanded_info["expanded_query"]

        # Enhanced retrieval with conversation context
        if conversation_context["question_type"] == "listing" and "companies" in conversation_context["focus_areas"]:
            # Use specialized retrieval for company listing questions
            retrieved_docs = self._enhanced_retrieval_for_listing(expanded_query, conversation_context)
        else:
            retrieved_docs = self.hybrid_retriever.hybrid_search(expanded_query)
        
        # Prepare context with source information and conversation focus
        context_parts = []
        source_info = []
        structured_sources = []
        
        for i, doc in enumerate(retrieved_docs):
            chunk_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            source = self.source_tracker.get_source_info(chunk_id)
            
            context_parts.append(f"[SOURCE {i+1}]\n{doc.page_content}")
            
            # Handle different source types
            if doc.metadata.get('source_type') == 'website':
                # Web source
                source_info.append(f"Source {i+1}: {doc.metadata.get('source', 'Unknown URL')}, Title: {doc.metadata.get('title', 'Unknown')}")
                structured_sources.append({
                    "source_id": i + 1,
                    "filename": doc.metadata.get('source', 'Unknown URL'),
                    "url": doc.metadata.get('source', 'Unknown URL'),
                    "page": 1,  # Web pages don't have page numbers
                    "section": doc.metadata.get('title', 'Web Page'),
                    "chunk_id": chunk_id,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            else:
                # PDF source
                source_info.append(f"Source {i+1}: {source['filename']}, Page {source['page']}, Section: {source['section']}")
                structured_sources.append({
                    "source_id": i + 1,
                    "filename": source['filename'],
                    "page": source['page'],
                    "section": source['section'],
                    "chunk_id": chunk_id,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        context = "\n\n".join(context_parts)
        source_summary = "\n".join(source_info)

        # Enhanced chain with conversation context
        chain = self.prompt_template | self.llm | StrOutputParser()

        response = chain.invoke({
            "messages": messages,
            "context": context,
            "expanded_query": expanded_query,
            "source_info": source_summary,
            "conversation_context": conversation_context
        })

        # Ensure sources are always returned if documents were retrieved
        if retrieved_docs and not structured_sources:
            # Fallback: create basic source info for retrieved documents
            for i, doc in enumerate(retrieved_docs):
                chunk_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                structured_sources.append({
                    "source_id": i + 1,
                    "filename": doc.metadata.get('source', 'Unknown'),
                    "url": doc.metadata.get('source', 'Unknown'),
                    "page": 1,
                    "section": doc.metadata.get('title', 'Document'),
                    "chunk_id": chunk_id,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        return {
            "response": response,
            "sources": structured_sources,
            "query_info": expanded_info
        }
    
    def _analyze_conversation_context(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Analyze conversation history to understand context and focus"""
        context = {
            "topic": None,
            "entities": [],
            "question_type": "general",
            "previous_questions": [],
            "focus_areas": []
        }
        
        # Extract previous questions and topics
        for msg in messages:
            if isinstance(msg, HumanMessage):
                question = msg.content.lower()
                context["previous_questions"].append(question)
                
                # Identify main topic
                if "adani" in question:
                    context["topic"] = "Adani Group"
                    context["entities"].append("Adani Group")
                
                # Identify question type
                if any(word in question for word in ["name", "list", "what are", "give me"]):
                    context["question_type"] = "listing"
                elif any(word in question for word in ["how", "why", "explain"]):
                    context["question_type"] = "explanation"
                elif any(word in question for word in ["when", "date", "time"]):
                    context["question_type"] = "temporal"
                elif any(word in question for word in ["where", "location"]):
                    context["question_type"] = "location"
                
                # Extract focus areas
                if "company" in question or "companies" in question:
                    context["focus_areas"].append("companies")
                if "group" in question:
                    context["focus_areas"].append("group structure")
                if "stakeholder" in question:
                    context["focus_areas"].append("stakeholders")
                if "financial" in question or "revenue" in question:
                    context["focus_areas"].append("financial")
        
        return context
    
    def _expand_query_with_context(self, query: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced query expansion that considers conversation context"""
        if not self.query_config.enable_llm_expansion or len(query) <= self.query_config.expansion_threshold:
            return {
                "original_query": query,
                "expanded_terms": [],
                "expanded_query": query,
                "expansion_used": False
            }
        
        cache_key = hashlib.md5(f"{query}_{str(conversation_context)}".encode()).hexdigest()
        
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
            # Context-aware expansion prompt
            context_info = ""
            if conversation_context["topic"]:
                context_info += f"Topic: {conversation_context['topic']}\n"
            if conversation_context["question_type"]:
                context_info += f"Question Type: {conversation_context['question_type']}\n"
            if conversation_context["focus_areas"]:
                context_info += f"Focus Areas: {', '.join(conversation_context['focus_areas'])}\n"
            
            expansion_prompt = f"""
            Analyze this business query with conversation context and suggest {self.query_config.max_expanded_terms} related terms:
            
            Query: "{query}"
            
            Conversation Context:
            {context_info}
            
            Consider:
            1. Financial terminology (revenue, profit, EBITDA, cash flow, etc.)
            2. Business processes and operations
            3. Industry-specific language and jargon
            4. Regulatory and compliance terms
            5. Strategic and management terms
            6. Market and competitive terms
            7. Table and chart related terms
            8. Image and visual data terms
            9. Company names and corporate entities
            10. Stakeholder and investor terms
            
            Focus on terms that would help answer the specific question type and topic.
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
    
    def _setup_vector_store(self):
        """Setup vector store with enhanced configuration"""
        if not self.qdrant_client:
            print("No Qdrant client available, using fallback only")
            return
        
        try:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' exists")
            except UnexpectedResponse as e:
                if "doesn't exist" in str(e):
                    print(f"📁 Creating enhanced collection: {self.collection_name}")
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=768,
                            distance=Distance.COSINE
                        )
                    )
                    print(f"Enhanced collection created")
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
            print(f"Enhanced vector store initialized")
        except Exception as e:
            print(f"Vector store error: {e}")
            self.vector_store = None
    
    def clear_collection(self):
        """Clear collection and reset source tracking"""
        if self.qdrant_client:
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                self.documents = []
                self.hybrid_retriever = None
                self.llm_cache.clear()
                self.source_tracker = EnhancedSourceTracker()  # Reset source tracking
                return f"Enhanced collection {self.collection_name} cleared successfully"
            except Exception as e:
                return f"Error clearing collection: {e}"
        return "No Qdrant client available"
    
    def get_stats(self):
        """Get enhanced statistics"""
        cache_stats = self.llm_cache.get_stats()
        return {
            "documents_count": len(self.documents),
            "collection_name": self.collection_name,
            "cache_stats": cache_stats,
            "vector_store_available": self.vector_store is not None,
            "qdrant_available": self.qdrant_client is not None,
            "source_tracking": {
                "documents_tracked": len(self.source_tracker.document_sources),
                "chunks_tracked": len(self.source_tracker.chunk_sources)
            },
            "enhanced_features": {
                "spelling_correction": self.query_config.enable_spelling_correction,
                "table_extraction": True,
                "image_processing": True,
                "source_attribution": True
            }
        }

    def _enhanced_retrieval_for_listing(self, query: str, conversation_context: Dict[str, Any]) -> List[Document]:
        """Enhanced retrieval specifically for listing questions to find company names and entities"""
        # First, get standard retrieval results
        standard_docs = self.hybrid_retriever.hybrid_search(query)
        
        # Then, search for documents that contain company names and entities
        entity_keywords = [
            "company", "companies", "group", "limited", "ltd", "corporation", "corp",
            "enterprises", "holdings", "ventures", "industries", "energy", "power",
            "ports", "gas", "cement", "steel", "mining", "logistics", "real estate"
        ]
        
        # Add topic-specific keywords
        if conversation_context["topic"] == "Adani Group":
            entity_keywords.extend([
                "adani", "enterprises", "power", "ports", "gas", "green", "total",
                "wilmar", "acc", "ambuja", "cement", "transmission"
            ])
        
        # Search for documents containing entity keywords
        entity_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            if any(keyword in content_lower for keyword in entity_keywords):
                entity_docs.append(doc)
        
        # Combine and deduplicate results
        all_docs = standard_docs + entity_docs
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
        
        # Return top results
        return unique_docs[:self.retrieval_config.final_top_k]

    def add_website(self, url: str, crawl_mode: str = "single") -> str:
        """Add website content to the knowledge base"""
        if not self.web_processor:
            raise ValueError("Web crawling is not available. Please install required dependencies.")
        
        try:
            print(f"🌐 Processing website: {url}")
            
            # Process the website
            web_docs = self.web_processor.process_website(url, crawl_mode)
            
            if not web_docs:
                return f"No content extracted from {url}"
            
            # Generate unique document ID for the website
            website_id = hashlib.md5(f"{url}_{time.time()}".encode()).hexdigest()
            
            # Add website to source tracker
            self.source_tracker.add_document(url, website_id)
            
            # Process each web document
            processed_docs = []
            for doc in web_docs:
                # Create enhanced document with web-specific metadata
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'document_id': website_id,
                        'source_type': 'website',
                        'url': doc.metadata.get('source', url),
                        'title': doc.metadata.get('title', 'Unknown'),
                        'domain': doc.metadata.get('domain', 'Unknown'),
                        'depth': doc.metadata.get('depth', 0)
                    }
                )
                processed_docs.append(enhanced_doc)
                
                # Track source information for web content
                chunk_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                
                # Add chunk to source tracker
                self.source_tracker.add_chunk(
                    chunk_id, 
                    doc.metadata.get('depth', 0) + 1,
                    doc.metadata.get('title', 'Web Page')
                )
            
            # Add to existing documents
            self.documents.extend(processed_docs)
            
            # Ensure vector store is set up
            if not self.vector_store and self.qdrant_client:
                self._setup_vector_store()
            
            # Update vector store if available
            if self.vector_store:
                try:
                    self.vector_store.add_documents(processed_docs)
                    print(f"✅ Added {len(processed_docs)} documents to vector store")
                except Exception as e:
                    print(f"⚠️ Vector store update failed: {e}")
                    # Fallback: use documents without vector store
                    self.vector_store = None
            
            # Reinitialize hybrid retriever with new documents
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=self.documents,
                config=self.retrieval_config
            )
            
            # Debug: Check if BM25 is set up
            if hasattr(self.hybrid_retriever, 'bm25') and self.hybrid_retriever.bm25:
                print(f"✅ BM25 index set up with {len(self.documents)} documents")
            else:
                print(f"⚠️ BM25 index not set up properly")
            
            return f"Website {url} processed successfully. Added {len(processed_docs)} pages to knowledge base."
            
        except Exception as e:
            print(f"Error processing website {url}: {e}")
            return f"Failed to process website {url}: {str(e)}"
    
    def add_single_webpage(self, url: str) -> str:
        """Add a single webpage to the knowledge base"""
        return self.add_website(url, crawl_mode="single")
    
    def add_website_with_crawling(self, url: str) -> str:
        """Add a website with multi-page crawling"""
        return self.add_website(url, crawl_mode="crawl")

# Reuse existing classes with minor modifications
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

# Import and reuse other classes from the original file
from pdf_chatbot import (
    ContextAwareChunker, AdvancedReranker, FallbackRetriever, HybridRetriever
) 