#!/usr/bin/env python3
"""
Web Crawler for RAG System
Extracts content from websites and multiple pages
"""

import requests
import time
import hashlib
import re
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_core.documents import Document
import logging

class WebCrawler:
    """Advanced web crawler for extracting content from websites"""
    
    def __init__(self, max_pages: int = 10, max_depth: int = 3, delay: float = 1.0):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Selenium for JavaScript-heavy sites
        self.driver = None
        self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver for JavaScript rendering"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Selenium: {e}")
            self.driver = None
    
    def crawl_website(self, start_url: str) -> List[Document]:
        """Crawl a website starting from the given URL"""
        self.logger.info(f"Starting crawl of: {start_url}")
        
        # Normalize the URL
        start_url = self._normalize_url(start_url)
        base_domain = urlparse(start_url).netloc
        
        documents = []
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        
        while urls_to_visit and len(documents) < self.max_pages:
            current_url, depth = urls_to_visit.pop(0)
            
            if depth > self.max_depth:
                continue
                
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            try:
                # Extract content from the page
                page_content = self._extract_page_content(current_url)
                
                if page_content and page_content.strip():
                    # Create document
                    doc = Document(
                        page_content=page_content,
                        metadata={
                            'source': current_url,
                            'type': 'webpage',
                            'title': self._extract_title(current_url),
                            'domain': base_domain,
                            'depth': depth,
                            'timestamp': time.time()
                        }
                    )
                    documents.append(doc)
                    self.logger.info(f"Extracted content from: {current_url}")
                
                # Find links to other pages on the same domain
                if depth < self.max_depth:
                    links = self._extract_links(current_url, base_domain)
                    for link in links:
                        if link not in self.visited_urls:
                            urls_to_visit.append((link, depth + 1))
                
                # Respect robots.txt and add delay
                time.sleep(self.delay)
                
            except Exception as e:
                self.logger.error(f"Error crawling {current_url}: {e}")
                continue
        
        self.logger.info(f"Crawl completed. Extracted {len(documents)} pages from {start_url}")
        return documents
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and ensuring proper format"""
        url, _ = urldefrag(url)
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url
    
    def _extract_page_content(self, url: str) -> Optional[str]:
        """Extract text content from a web page"""
        try:
            # Try with requests first (faster)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up the text
            text = self._clean_text(text)
            
            # If content seems too short, try with Selenium
            if len(text.strip()) < 100 and self.driver:
                return self._extract_with_selenium(url)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Failed to extract content with requests from {url}: {e}")
            
            # Fallback to Selenium
            if self.driver:
                return self._extract_with_selenium(url)
            
            return None
    
    def _extract_with_selenium(self, url: str) -> Optional[str]:
        """Extract content using Selenium for JavaScript-heavy sites"""
        try:
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source and parse
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            text = soup.get_text()
            return self._clean_text(text)
            
        except Exception as e:
            self.logger.error(f"Selenium extraction failed for {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common web artifacts
        text = re.sub(r'Cookie|Privacy Policy|Terms of Service|Contact Us', '', text, flags=re.IGNORECASE)
        
        # Remove very short lines
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        
        return '\n'.join(lines).strip()
    
    def _extract_title(self, url: str) -> str:
        """Extract page title"""
        try:
            response = self.session.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            return title.get_text().strip() if title else url
        except:
            return url
    
    def _extract_links(self, url: str, base_domain: str) -> List[str]:
        """Extract links from a page that belong to the same domain"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Only include links from the same domain
                if urlparse(full_url).netloc == base_domain:
                    # Remove fragments and query parameters for deduplication
                    clean_url, _ = urldefrag(full_url)
                    if clean_url not in self.visited_urls:
                        links.append(clean_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            self.logger.warning(f"Failed to extract links from {url}: {e}")
            return []
    
    def extract_single_page(self, url: str) -> Optional[Document]:
        """Extract content from a single web page"""
        try:
            url = self._normalize_url(url)
            content = self._extract_page_content(url)
            
            if content:
                return Document(
                    page_content=content,
                    metadata={
                        'source': url,
                        'type': 'webpage',
                        'title': self._extract_title(url),
                        'domain': urlparse(url).netloc,
                        'depth': 0,
                        'timestamp': time.time()
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract single page {url}: {e}")
            return None
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
        self.session.close()

class WebContentProcessor:
    """Process web content for RAG system"""
    
    def __init__(self):
        self.crawler = WebCrawler()
    
    def process_website(self, url: str, crawl_mode: str = "single") -> List[Document]:
        """Process a website URL and return documents"""
        if crawl_mode == "single":
            # Extract single page
            doc = self.crawler.extract_single_page(url)
            return [doc] if doc else []
        else:
            # Crawl multiple pages
            return self.crawler.crawl_website(url)
    
    def close(self):
        """Clean up resources"""
        self.crawler.close() 