from flask import Flask, request, jsonify, render_template_string, send_file
from enhanced_pdf_chatbot import EnhancedPDFChatbot
import json
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from enhanced_config import get_enhanced_chatbot_config as get_chatbot_config, validate_enhanced_config as validate_config, get_performance_config
from dotenv import load_dotenv
import os
import traceback
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*ARC4.*")
warnings.filterwarnings("ignore", message=".*getaddrinfo failed.*")
warnings.filterwarnings("ignore", message=".*DEPRECATED_ENDPOINT.*")
warnings.filterwarnings("ignore", message=".*PHONE_REGISTRATION_ERROR.*")

# Suppress NLTK download messages
import logging
logging.getLogger('nltk').setLevel(logging.ERROR)

# Suppress other noisy loggers
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('WDM').setLevel(logging.WARNING)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Validate configuration
try:
    validate_config()
    print("✅ Configuration validated successfully")
except Exception as e:
    print(f"⚠️ Configuration warning: {e}")

# Initialize chatbot with configuration
try:
    chatbot_config = get_chatbot_config()
    performance_config = get_performance_config()
    chatbot = EnhancedPDFChatbot(**chatbot_config)
    print("Enhanced chatbot initialized successfully")
except Exception as e:
    print(f"Failed to initialize chatbot: {e}")
    raise

@app.route('/ui')
def serve_ui():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Advanced RAG PDF Chatbot</title>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        * { box-sizing: border-box; }
        body { 
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
          margin: 0; padding: 20px; 
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
        }
        .container {
          max-width: 1200px;
          margin: 0 auto;
          background: white;
          border-radius: 16px;
          box-shadow: 0 20px 40px rgba(0,0,0,0.1);
          overflow: hidden;
          backdrop-filter: blur(10px);
        }
        .header {
          background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
          color: white;
          padding: 30px;
          text-align: center;
        }
        .header h1 {
          margin: 0 0 10px 0;
          font-size: 2.5em;
          font-weight: 700;
        }
        .header p {
          margin: 0;
          font-size: 1.1em;
          opacity: 0.9;
        }
        .stats-bar {
          background: #f8fafc;
          padding: 15px 30px;
          border-bottom: 1px solid #e2e8f0;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.9em;
          color: #64748b;
        }
        .stat-item {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .stat-value {
          font-weight: 600;
          color: #1e293b;
        }
        .upload-section {
          padding: 30px;
          border-bottom: 1px solid #e2e8f0;
          background: #fafbfc;
        }
        .upload-container {
          display: flex;
          gap: 15px;
          align-items: center;
          flex-wrap: wrap;
        }
        .file-input-wrapper {
          position: relative;
          flex: 1;
          min-width: 200px;
        }
        .file-input-wrapper input[type="file"] {
          width: 100%;
          padding: 12px;
          border: 2px dashed #cbd5e1;
          border-radius: 8px;
          background: white;
          cursor: pointer;
          transition: all 0.3s ease;
        }
        .file-input-wrapper input[type="file"]:hover {
          border-color: #2563eb;
          background: #f0f9ff;
        }
        .upload-section button {
          padding: 12px 24px;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.3s ease;
          min-width: 120px;
        }
        .upload-section button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        }
        .upload-section button:disabled {
          background: #9ca3af;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        #chatbox { 
          height: 600px; 
          overflow-y: auto; 
          padding: 30px;
          background: #fafbfc;
          scroll-behavior: smooth;
        }
        .message {
          margin: 20px 0;
          padding: 20px;
          border-radius: 12px;
          max-width: 85%;
          position: relative;
          animation: fadeInUp 0.3s ease;
        }
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
          background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
          margin-left: auto;
          text-align: right;
          border-bottom-right-radius: 4px;
        }
        .bot-message {
          background: white;
          border: 1px solid #e2e8f0;
          border-bottom-left-radius: 4px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .message-header {
          font-weight: 600;
          margin-bottom: 12px;
          font-size: 0.9em;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .user-message .message-header {
          color: #1d4ed8;
          justify-content: flex-end;
        }
        .bot-message .message-header {
          color: #059669;
        }
        .message-content {
          line-height: 1.6;
          color: #1f2937;
        }
        .query-info {
          background: #f1f5f9;
          padding: 12px;
          border-radius: 8px;
          margin-bottom: 15px;
          font-size: 0.85em;
          color: #475569;
          border-left: 4px solid #3b82f6;
        }
        .query-info strong {
          color: #1e293b;
        }
        .input-section {
          padding: 30px;
          border-top: 1px solid #e2e8f0;
          background: white;
        }
        .input-container {
          display: flex;
          gap: 15px;
          align-items: flex-end;
        }
        .input-wrapper {
          flex: 1;
          position: relative;
        }
        #userInput {
          width: 100%;
          padding: 16px 20px;
          border: 2px solid #e2e8f0;
          border-radius: 12px;
          font-size: 16px;
          transition: all 0.3s ease;
          resize: vertical;
          min-height: 50px;
          max-height: 150px;
        }
        #userInput:focus {
          outline: none;
          border-color: #2563eb;
          box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        #sendBtn {
          padding: 16px 32px;
          background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
          color: white;
          border: none;
          border-radius: 12px;
          cursor: pointer;
          font-size: 16px;
          font-weight: 600;
          transition: all 0.3s ease;
          min-width: 120px;
        }
        #sendBtn:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(37, 99, 235, 0.3);
        }
        #sendBtn:disabled {
          background: #9ca3af;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        .loading {
          display: none;
          color: #64748b;
          font-style: italic;
          margin-top: 10px;
          text-align: center;
        }
        .status {
          padding: 15px;
          margin-top: 15px;
          border-radius: 8px;
          font-size: 14px;
          font-weight: 500;
        }
        .status.success {
          background: #dcfce7;
          color: #166534;
          border: 1px solid #bbf7d0;
        }
        .status.error {
          background: #fef2f2;
          color: #991b1b;
          border: 1px solid #fecaca;
        }
        .status.info {
          background: #dbeafe;
          color: #1e40af;
          border: 1px solid #bfdbfe;
        }
        .clear-btn {
          padding: 8px 16px;
          background: #ef4444;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.9em;
          transition: all 0.3s ease;
        }
        .clear-btn:hover {
          background: #dc2626;
        }
        .sources-section {
          margin-top: 15px;
          padding: 12px;
          background: #f8fafc;
          border-radius: 8px;
          border-left: 4px solid #10b981;
        }
        .source-links {
          margin-top: 8px;
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
        .source-link {
          display: inline-block;
          padding: 6px 12px;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          color: white;
          text-decoration: none;
          border-radius: 6px;
          font-size: 0.85em;
          font-weight: 500;
          transition: all 0.3s ease;
          cursor: pointer;
        }
        .source-link:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
          text-decoration: none;
          color: white;
        }
        .source-link:active {
          transform: translateY(0);
        }
        .source-link.web-source {
          background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        }
        .source-link.web-source:hover {
          box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }
        .website-section {
          margin-top: 30px;
          padding: 25px;
          background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
          border-radius: 12px;
          border: 1px solid #bae6fd;
        }
        .website-section h3 {
          margin: 0 0 20px 0;
          color: #0369a1;
          font-size: 1.2em;
        }
        .website-container {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }
        .url-input-wrapper {
          position: relative;
        }
        .url-input-wrapper input[type="url"] {
          width: 100%;
          padding: 12px 16px;
          border: 2px solid #bae6fd;
          border-radius: 8px;
          font-size: 14px;
          transition: all 0.3s ease;
        }
        .url-input-wrapper input[type="url"]:focus {
          outline: none;
          border-color: #0369a1;
          box-shadow: 0 0 0 3px rgba(3, 105, 161, 0.1);
        }
        .crawl-options {
          display: flex;
          gap: 20px;
          align-items: center;
        }
        .crawl-options label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #0369a1;
          cursor: pointer;
        }
        .crawl-options input[type="radio"] {
          accent-color: #0369a1;
        }
        .website-section button {
          padding: 12px 24px;
          background: linear-gradient(135deg, #0369a1 0%, #0c4a6e 100%);
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.3s ease;
          align-self: flex-start;
        }
        .website-section button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(3, 105, 161, 0.3);
        }
        .website-section button:disabled {
          background: #9ca3af;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        #websiteStatus {
          margin-top: 15px;
        }
        .suggestions {
          display: flex;
          gap: 10px;
          margin-top: 15px;
          flex-wrap: wrap;
        }
        .suggestion-btn {
          padding: 8px 16px;
          background: #f1f5f9;
          border: 1px solid #e2e8f0;
          border-radius: 20px;
          cursor: pointer;
          font-size: 0.9em;
          transition: all 0.3s ease;
          color: #475569;
        }
        .suggestion-btn:hover {
          background: #e2e8f0;
          color: #1e293b;
        }
        .typing-indicator {
          display: none;
          padding: 20px;
          color: #64748b;
          font-style: italic;
        }
        .typing-dots {
          display: inline-block;
          animation: typing 1.4s infinite;
        }
        @keyframes typing {
          0%, 20% { content: "Analyzing"; }
          40% { content: "Analyzing."; }
          60% { content: "Analyzing.."; }
          80%, 100% { content: "Analyzing..."; }
        }
        .error-message {
          background: #fef2f2;
          border: 1px solid #fecaca;
          color: #991b1b;
          padding: 15px;
          border-radius: 8px;
          margin: 15px 0;
        }
        @media (max-width: 768px) {
          .container { margin: 10px; border-radius: 12px; }
          .header { padding: 20px; }
          .header h1 { font-size: 2em; }
          .upload-container { flex-direction: column; }
          .input-container { flex-direction: column; }
          .message { max-width: 95%; }
          .stats-bar { flex-direction: column; gap: 10px; }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Advanced RAG PDF Chatbot</h1>
          <p>Intelligent document analysis with hybrid retrieval and semantic understanding</p>
        </div>
        
        <div class="stats-bar">
          <div class="stat-item">
            <span>Documents:</span>
            <span class="stat-value" id="docCount">0</span>
          </div>
          <div class="stat-item">
            <span>Cache Hit Rate:</span>
            <span class="stat-value" id="cacheRate">0%</span>
          </div>
          <div class="stat-item">
            <span>Vector Store:</span>
            <span class="stat-value" id="vectorStatus">Local</span>
          </div>
          <button class="clear-btn" onclick="clearCollection()">🗑️ Clear All</button>
        </div>
        
        <div class="upload-section">
          <h3>📄 Upload PDF Document</h3>
          <div class="upload-container">
            <div class="file-input-wrapper">
              <input type="file" id="fileInput" accept=".pdf" />
            </div>
            <button onclick="uploadPDF()">Upload PDF</button>
          </div>
          <div id="uploadStatus"></div>
        </div>
        
        <div class="website-section">
          <h3>🌐 Add Website Content</h3>
          <div class="website-container">
            <div class="url-input-wrapper">
              <input type="url" id="urlInput" placeholder="Enter website URL (e.g., https://example.com)" />
            </div>
            <div class="crawl-options">
              <label>
                <input type="radio" name="crawlMode" value="single" checked> Single Page
              </label>
              <label>
                <input type="radio" name="crawlMode" value="crawl"> Multi-Page Crawl
              </label>
            </div>
            <button onclick="addWebsite()">Add Website</button>
          </div>
          <div id="websiteStatus"></div>
        </div>

        <div id="chatbox">
          <div class="message bot-message">
            <div class="message-header">
              🤖 Assistant
            </div>
            <div class="message-content">
              Welcome to the Advanced RAG PDF Chatbot!
              <br><br>
              <strong>Features:</strong>
              • Hybrid search (dense + sparse retrieval)
              • Semantic chunking with boundary detection
              • Query expansion and reranking
              • Professional formatting and citations
              • Real-time cache optimization
              <br><br>
              Upload a PDF document to get started!
            </div>
          </div>
        </div>
        
        <div class="input-section">
          <div class="input-container">
            <div class="input-wrapper">
              <textarea id="userInput" placeholder="Ask a question about your PDF document..." 
                       onkeydown="if(event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }"></textarea>
            </div>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
          </div>
          <div class="loading" id="loadingIndicator">
            <div class="typing-indicator">
              <span class="typing-dots">Analyzing your question...</span>
            </div>
          </div>
          <div class="suggestions">
            <button class="suggestion-btn" onclick="askQuestion('What are the key financial highlights?')">Financial Highlights</button>
            <button class="suggestion-btn" onclick="askQuestion('What are the main risks mentioned?')">Risk Factors</button>
            <button class="suggestion-btn" onclick="askQuestion('What is the business strategy?')">Business Strategy</button>
            <button class="suggestion-btn" onclick="askQuestion('What are the growth projections?')">Growth Projections</button>
          </div>
        </div>
      </div>

      <script>
        const chatbox = document.getElementById('chatbox');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const loadingIndicator = document.getElementById('loadingIndicator');

        function updateStats() {
          fetch('/stats')
            .then(response => response.json())
            .then(data => {
              document.getElementById('docCount').textContent = data.documents_count;
              document.getElementById('cacheRate').textContent = data.cache_stats.hit_rate;
              document.getElementById('vectorStatus').textContent = data.vector_store_available ? 'Cloud' : 'Local';
            })
            .catch(error => console.error('Error fetching stats:', error));
        }

        function showStatus(message, type = 'info') {
          const statusDiv = document.getElementById('uploadStatus');
          statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
          setTimeout(() => {
            statusDiv.innerHTML = '';
          }, 5000);
        }

        function showWebsiteStatus(message, type = 'info') {
          const statusDiv = document.getElementById('websiteStatus');
          statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
          setTimeout(() => {
            statusDiv.innerHTML = '';
          }, 5000);
        }

        function appendMessage(role, text, queryInfo = null, sources = []) {
          const messageDiv = document.createElement('div');
          messageDiv.className = `message ${role}-message`;
          
          let queryInfoHtml = '';
          if (queryInfo && queryInfo.expansion_used) {
            queryInfoHtml = `
              <div class="query-info">
                <strong>🔍 Query Enhancement:</strong> "${queryInfo.original_query}" → "${queryInfo.expanded_query}"<br>
                <strong>➕ Added terms:</strong> ${queryInfo.expanded_terms.join(', ')}
              </div>
            `;
          }
          
          let sourcesHtml = '';
          if (sources && sources.length > 0) {
            sourcesHtml = `
              <div class="sources-section">
                <strong>📚 Sources:</strong>
                <div class="source-links">
                  ${sources.map(source => {
                    if (source.filename && source.filename.endsWith('.pdf')) {
                      // PDF source
                      return `<a href="/pdf/${encodeURIComponent(source.filename)}#page=${source.page}" 
                                 target="_blank" 
                                 class="source-link" 
                                 title="Page ${source.page} - ${source.content_preview}"
                                 onclick="openSource('${source.filename}', ${source.page}, '${source.chunk_id}')">
                                Source ${source.source_id}
                              </a>`;
                    } else {
                      // Web source
                      return `<a href="${source.filename || source.url}" 
                                 target="_blank" 
                                 class="source-link web-source" 
                                 title="${source.content_preview}"
                                 onclick="openWebSource('${source.filename || source.url}', '${source.chunk_id}')">
                                Source ${source.source_id} (Web)
                              </a>`;
                    }
                  }).join(', ')}
                </div>
              </div>
            `;
          }
          
          const icon = role === 'user' ? 'User' : 'Assistant';
          const header = role === 'user' ? 'You' : 'Assistant';
          
          messageDiv.innerHTML = `
            <div class="message-header">
              ${icon} ${header}
            </div>
            ${queryInfoHtml}
            <div class="message-content">${formatResponse(text)}</div>
            ${sourcesHtml}
          `;
          
          chatbox.appendChild(messageDiv);
          chatbox.scrollTop = chatbox.scrollHeight;
        }

        function formatResponse(text) {
          return text
            .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong style="color: #1d4ed8;">$1</strong>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/\\[Section ([^\\]]+)\\]/g, '<span style="background: #fef3c7; padding: 2px 6px; border-radius: 3px; font-size: 12px; color: #92400e;">[Section $1]</span>')
            .replace(/\\n/g, '<br>')
            .replace(/(\d+\.\s)/g, '<br>$1')
            .replace(/•/g, '<br>•')
            .replace(/💰/g, '<span style="color: #059669;">💰</span>')
            .replace(/📊/g, '<span style="color: #2563eb;">📊</span>')
            .replace(/⚠️/g, '<span style="color: #dc2626;">⚠️</span>')
            .replace(/🎯/g, '<span style="color: #7c3aed;">🎯</span>');
        }

        function setLoading(loading) {
          sendBtn.disabled = loading;
          loadingIndicator.style.display = loading ? 'block' : 'none';
          sendBtn.textContent = loading ? 'Processing...' : 'Send';
        }

        function askQuestion(question) {
          userInput.value = question;
          sendMessage();
        }

        function openSource(filename, page, chunkId) {
          // Open PDF in new tab with page parameter
          const pdfUrl = `/pdf/${encodeURIComponent(filename)}#page=${page}`;
          window.open(pdfUrl, '_blank');
          
          // Optional: Show source details in a modal or tooltip
          console.log(`Opening source: ${filename}, Page ${page}, Chunk: ${chunkId}`);
        }

        function openWebSource(url, chunkId) {
          // Open web page in new tab
          window.open(url, '_blank');
          
          // Optional: Show source details in a modal or tooltip
          console.log(`Opening web source: ${url}, Chunk: ${chunkId}`);
        }

        async function uploadPDF() {
          const file = document.getElementById('fileInput').files[0];
          if (!file || !file.name.endsWith('.pdf')) {
            showStatus('Please select a valid PDF file', 'error');
            return;
          }

          const formData = new FormData();
          formData.append('file', file);

          try {
            const response = await fetch('/upload', {
              method: 'POST',
              body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
              showStatus(result.message, 'success');
              updateStats();
            } else {
              showStatus(result.error, 'error');
            }
          } catch (error) {
            showStatus('Upload failed. Please try again.', 'error');
            console.error('Upload error:', error);
          }
        }

        async function addWebsite() {
          const url = document.getElementById('urlInput').value.trim();
          if (!url) {
            showWebsiteStatus('Please enter a valid website URL', 'error');
            return;
          }

          // Validate URL format
          try {
            new URL(url);
          } catch {
            showWebsiteStatus('Please enter a valid URL (e.g., https://example.com)', 'error');
            return;
          }

          const crawlMode = document.querySelector('input[name="crawlMode"]:checked').value;
          const button = event.target;
          const originalText = button.textContent;
          
          button.disabled = true;
          button.textContent = 'Processing...';

          try {
            const response = await fetch('/add-website', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                url: url,
                crawl_mode: crawlMode
              })
            });

            const result = await response.json();
            
            if (response.ok) {
              showWebsiteStatus(result.message, 'success');
              document.getElementById('urlInput').value = '';
              updateStats();
            } else {
              showWebsiteStatus(result.error, 'error');
            }
          } catch (error) {
            showWebsiteStatus('Failed to add website. Please try again.', 'error');
            console.error('Add website error:', error);
          } finally {
            button.disabled = false;
            button.textContent = originalText;
          }
        }

        async function sendMessage() {
          const message = userInput.value.trim();
          if (!message) return;

          setLoading(true);
          appendMessage('user', message);
          userInput.value = '';

          try {
            const response = await fetch('/chat', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                messages: [{'role': 'user', 'content': message}]
              })
            });

            const result = await response.json();
            
            if (response.ok) {
              appendMessage('bot', result.response, result.query_info, result.sources);
            } else {
                             appendMessage('bot', `Error: ${result.error}`);
            }
          } catch (error) {
                         appendMessage('bot', 'Network error. Please try again.');
            console.error('Chat error:', error);
          } finally {
            setLoading(false);
          }
        }

        async function clearCollection() {
          if (!confirm('Are you sure you want to clear all documents and chat history?')) {
            return;
          }

          try {
            const response = await fetch('/clear', {
              method: 'POST'
            });

            const result = await response.json();
            
            if (response.ok) {
              showStatus(result.message, 'success');
              chatbox.innerHTML = `
                <div class="message bot-message">
                  <div class="message-header">🤖 Assistant</div>
                  <div class="message-content">
                    Collection cleared successfully! Upload a new PDF to get started.
                  </div>
                </div>
              `;
              updateStats();
            } else {
              showStatus(result.error, 'error');
            }
          } catch (error) {
            showStatus('Failed to clear collection.', 'error');
            console.error('Clear error:', error);
          }
        }

        // Update stats on page load
        updateStats();
        
        // Auto-resize textarea
        userInput.addEventListener('input', function() {
          this.style.height = 'auto';
          this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
      </script>
    </body>
    </html>
    """
    return html

@app.route('/')
def index():
    return jsonify({
        "message": "Advanced RAG PDF Chatbot API",
        "version": "2.0",
        "features": [
            "Hybrid search (dense + sparse retrieval)",
            "Semantic chunking with boundary detection", 
            "Query expansion and reranking",
            "Professional formatting and citations",
            "Real-time cache optimization"
        ],
        "endpoints": {
            "/ui": "Web interface",
            "/upload": "Upload PDF (POST)",
            "/chat": "Chat endpoint (POST)",
            "/clear": "Clear collection (POST)",
            "/stats": "Get statistics (GET)"
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        file_bytes = file.read()
        if len(file_bytes) > performance_config['max_file_size']:
            return jsonify({"error": "File too large. Maximum size is 16MB"}), 413
        
        result = chatbot.upload_pdf_from_bytes(file_bytes, str(file.filename))
        return jsonify({"message": result})
    
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({"error": "No messages provided"}), 400
        
        messages = []
        for msg in data['messages']:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        result = chatbot.query(messages)
        
        # Handle both old string format and new dict format
        if isinstance(result, str):
            return jsonify({
                "response": result,
                "sources": [],
                "query_info": None
            })
        else:
            return jsonify({
                "response": result["response"],
                "sources": result["sources"],
                "query_info": result["query_info"]
            })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

@app.route('/clear', methods=['POST'])
def clear_collection():
    try:
        result = chatbot.clear_collection()
        return jsonify({"message": result})
    except Exception as e:
        app.logger.error(f"Clear error: {str(e)}")
        return jsonify({"error": f"Failed to clear collection: {str(e)}"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = chatbot.get_stats()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Stats error: {str(e)}")
        return jsonify({"error": f"Failed to get stats: {str(e)}"}), 500

@app.route('/pdf/<filename>')
def serve_pdf(filename):
    """Serve PDF files for source citations"""
    try:
        # Security: Only allow PDF files
        if not filename.lower().endswith('.pdf'):
            return jsonify({"error": "Invalid file type"}), 400
        
        # For now, we'll serve from a temporary directory
        # In production, you'd want to store PDFs securely
        import tempfile
        import os
        
        # Create a temporary directory for PDFs if it doesn't exist
        pdf_dir = os.path.join(tempfile.gettempdir(), 'rag_pdfs')
        os.makedirs(pdf_dir, exist_ok=True)
        
        pdf_path = os.path.join(pdf_dir, filename)
        
        if not os.path.exists(pdf_path):
            return jsonify({"error": "PDF not found"}), 404
        
        return send_file(pdf_path, mimetype='application/pdf')
    
    except Exception as e:
        app.logger.error(f"PDF serve error: {str(e)}")
        return jsonify({"error": f"Failed to serve PDF: {str(e)}"}), 500

@app.route('/source/<chunk_id>')
def get_source_details(chunk_id):
    """Get detailed source information for a specific chunk"""
    try:
        # This would return the exact location and content of the source
        # For now, return basic info
        return jsonify({
            "chunk_id": chunk_id,
            "message": "Source details endpoint - to be implemented with PDF highlighting"
        })
    except Exception as e:
        app.logger.error(f"Source details error: {str(e)}")
        return jsonify({"error": f"Failed to get source details: {str(e)}"}), 500

@app.route('/add-website', methods=['POST'])
def add_website():
    """Add website content to the knowledge base"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400
        
        url = data['url']
        crawl_mode = data.get('crawl_mode', 'single')  # 'single' or 'crawl'
        
        if crawl_mode == 'crawl':
            result = chatbot.add_website_with_crawling(url)
        else:
            result = chatbot.add_single_webpage(url)
        
        return jsonify({"message": result})
    
    except Exception as e:
        app.logger.error(f"Add website error: {str(e)}")
        return jsonify({"error": f"Failed to add website: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error. Please try again."}), 500

if __name__ == '__main__':
    print("Enhanced RAG PDF Chatbot Starting...")
    print("Features: Advanced document processing, conversation memory, source attribution")
    print("Web Interface: http://localhost:5000/ui")
    print("API Endpoints: /upload, /chat, /clear, /stats")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)