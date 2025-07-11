from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import tempfile
import time
from pdf_chatbot import FreePDFChatbot

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize free chatbot (no API key needed!)
chatbot = FreePDFChatbot()

# Minimal HTML template with installation instructions
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Free PDF RAG Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
        .messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; }
        .bot { background-color: #f3e5f5; }
        .expansion-info { font-size: 0.9em; color: #666; margin-top: 5px; }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; margin: 5px; }
        .stats { font-size: 0.9em; color: #666; }
        .install-info { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .success { color: green; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Free PDF RAG Chatbot</h1>
        <p><strong>✅ No API Keys Required!</strong> Uses free Hugging Face models.</p>
        
        <div class="install-info">
            <h3>Required Dependencies:</h3>
            <code>pip install flask sentence-transformers transformers torch faiss-cpu PyPDF2 scikit-learn</code>
            <br><br>
            <strong>Note:</strong> First run will download ~500MB of AI models. This is normal and only happens once.
        </div>
        
        <div class="section">
            <h3>Upload PDF</h3>
            <form id="upload-form" enctype="multipart/form-data">
                <input type="file" name="pdf" accept=".pdf" required>
                <button type="submit">Upload PDF</button>
            </form>
            <div id="upload-status"></div>
        </div>
        
        <div class="section">
            <h3>Chat</h3>
            <div id="messages" class="messages">
                <div class="message bot">
                    <strong>Bot:</strong> Upload a PDF to start chatting! I'll use smart query expansion to find better answers.
                </div>
            </div>
            <div>
                <input type="text" id="question" placeholder="Ask a question about your PDF...">
                <button onclick="sendQuestion()">Send</button>
            </div>
        </div>
        
        <div class="section">
            <h3>Cost Optimization Stats</h3>
            <button onclick="loadStats()">Refresh Stats</button>
            <div id="stats-display"></div>
        </div>
    </div>

    <script>
        // Upload PDF
        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const status = document.getElementById('upload-status');
            
            status.innerHTML = '⏳ Uploading and processing PDF... This may take a moment.';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    status.innerHTML = `✅ ${result.message}`;
                    status.className = 'success';
                } else {
                    status.innerHTML = `❌ ${result.message}`;
                    status.className = 'error';
                }
            } catch (error) {
                status.innerHTML = '❌ Upload failed: ' + error.message;
                status.className = 'error';
            }
        });
        
        // Send question
        async function sendQuestion() {
            const question = document.getElementById('question').value;
            if (!question.trim()) return;
            
            const messagesDiv = document.getElementById('messages');
            
            // Add user message
            messagesDiv.innerHTML += `
                <div class="message user">
                    <strong>You:</strong> ${question}
                </div>
            `;
            
            // Clear input
            document.getElementById('question').value = '';
            
            // Add loading message
            messagesDiv.innerHTML += `
                <div class="message bot" id="loading">
                    <strong>Bot:</strong> <em>🤔 Thinking and expanding your query...</em>
                </div>
            `;
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({question: question})
                });
                
                const result = await response.json();
                
                // Remove loading message
                document.getElementById('loading').remove();
                
                // Add confidence indicator
                const confidenceColor = result.confidence > 0.7 ? 'green' : result.confidence > 0.4 ? 'orange' : 'red';
                const confidenceText = result.confidence ? `(Confidence: ${(result.confidence * 100).toFixed(1)}%)` : '';
                
                // Add expansion info
                const expansionInfo = result.expanded_terms && result.expanded_terms.length > 1 ? 
                    `<div class="expansion-info">
                        <strong>🔍 Expansion Method:</strong> ${result.expansion_method}<br>
                        <strong>📝 Search Terms:</strong> ${result.expanded_terms.join(', ')}
                    </div>` : '';
                
                messagesDiv.innerHTML += `
                    <div class="message bot">
                        <strong>Bot:</strong> ${result.answer} 
                        <span style="color: ${confidenceColor}; font-size: 0.8em;">${confidenceText}</span>
                        ${expansionInfo}
                    </div>
                `;
                
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
            } catch (error) {
                document.getElementById('loading').remove();
                messagesDiv.innerHTML += `
                    <div class="message bot">
                        <strong>Bot:</strong> ❌ Error: ${error.message}
                    </div>
                `;
            }
        }
        
        // Load stats
        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                
                document.getElementById('stats-display').innerHTML = `
                    <div class="stats">
                        <h4>💰 Cost Optimization (Free Model):</h4>
                        <p>✅ Synonym DB usage: ${stats.cost_savings.synonym_db_usage} (Free)</p>
                        <p>🔄 Similarity usage: ${stats.cost_savings.similarity_usage} (Free)</p>
                        <p>💾 Cache usage: ${stats.cost_savings.cache_usage} (Free)</p>
                        <p><strong>Total queries processed: ${stats.total_queries}</strong></p>
                        
                        <h4>📊 Session Stats:</h4>
                        <p>Synonym hits: ${stats.session_stats.synonym_hits}</p>
                        <p>Similarity hits: ${stats.session_stats.similarity_hits}</p>
                        <p>Cache hits: ${stats.session_stats.cache_hits}</p>
                        
                        <p><strong>💡 All processing is FREE - no API costs!</strong></p>
                    </div>
                `;
            } catch (error) {
                document.getElementById('stats-display').innerHTML = '❌ Error loading stats: ' + error.message;
            }
        }
        
        // Enter key support
        document.getElementById('question').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendQuestion();
            }
        });
        
        // Load initial stats
        loadStats();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'message': 'No PDF file provided'})
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'message': 'Please upload a PDF file'})
        
        # Save file temporarily - Windows-safe approach
        filename = secure_filename(file.filename)
        tmp_path = os.path.join(tempfile.gettempdir(), f"temp_pdf_{os.getpid()}_{filename}")
        
        try:
            file.save(tmp_path)
            
            # Load PDF into chatbot
            success = chatbot.load_pdf(tmp_path)
            
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'PDF "{filename}" uploaded and processed successfully! Ready for questions.'
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': 'Failed to process PDF. Please check the file format.'
                })
                
        finally:
            # Clean up temp file (Windows-safe)
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except PermissionError:
                # File still in use, try again after delay
                time.sleep(0.5)
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # Let OS handle cleanup
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/query', methods=['POST'])
def query_pdf():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'answer': 'Please provide a question'})
        
        # Get response with expansion info
        result = chatbot.answer_question(question)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'answer': f'Error: {str(e)}',
            'expanded_terms': [],
            'expansion_method': 'error'
        })

@app.route('/stats')
def get_stats():
    try:
        stats = chatbot.get_expansion_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Free PDF RAG Chatbot is running'})

if __name__ == '__main__':
    print("🚀 Starting Free PDF RAG Chatbot...")
    print("📋 Required packages: flask sentence-transformers transformers torch faiss-cpu PyPDF2 scikit-learn")
    print("⚠️  First run will download AI models (~500MB). This is normal!")
    print("💰 No API keys required - everything runs locally!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)