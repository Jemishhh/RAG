from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import json
from pdf_chatbot import PDFChatbot  # Import our chatbot class

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize chatbot
chatbot = PDFChatbot(collection_name="web_pdf_docs")

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 2px dashed #dee2e6;
        }
        .chat-section {
            margin-top: 30px;
        }
        .message {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 70%;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
        }
        .bot-message {
            background-color: #e9ecef;
            color: #333;
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            background-color: #fafafa;
        }
        .input-group {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .file-input {
            padding: 10px;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            background-color: white;
            margin-right: 10px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .loading {
            display: none;
            color: #6c757d;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 PDF Chatbot</h1>
        
        <div class="upload-section">
            <h3>Upload PDF Document</h3>
            <div style="display: flex; align-items: center; gap: 10px;">
                <input type="file" id="pdfFile" accept=".pdf" class="file-input">
                <button onclick="uploadPDF()">Upload PDF</button>
            </div>
            <div id="uploadStatus"></div>
        </div>
        
        <div class="chat-section">
            <h3>Chat with your PDF</h3>
            <div id="chatContainer" class="chat-container"></div>
            <div class="loading" id="loadingIndicator">Bot is typing...</div>
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Ask a question about your PDF..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()" id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        let threadId = 'web_' + Math.random().toString(36).substr(2, 9);
        
        function uploadPDF() {
            const fileInput = document.getElementById('pdfFile');
            const file = fileInput.files[0];
            const statusDiv = document.getElementById('uploadStatus');
            
            if (!file) {
                showStatus('Please select a PDF file', 'error');
                return;
            }
            
            if (file.type !== 'application/pdf') {
                showStatus('Please select a valid PDF file', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('pdf', file);
            
            showStatus('Uploading and processing PDF...', 'info');
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(data.message, 'success');
                    fileInput.value = '';
                } else {
                    showStatus(data.message, 'error');
                }
            })
            .catch(error => {
                showStatus('Error uploading file: ' + error, 'error');
            });
        }
        
        function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            messageInput.value = '';
            
            const sendButton = document.getElementById('sendButton');
            const loadingIndicator = document.getElementById('loadingIndicator');
            
            sendButton.disabled = true;
            loadingIndicator.style.display = 'block';
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    thread_id: threadId
                })
            })
            .then(response => response.json())
            .then(data => {
                addMessage(data.response, 'bot');
                sendButton.disabled = false;
                loadingIndicator.style.display = 'none';
            })
            .catch(error => {
                addMessage('Error: ' + error, 'bot');
                sendButton.disabled = false;
                loadingIndicator.style.display = 'none';
            });
        }
        
        function addMessage(text, sender) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
            setTimeout(() => {
                statusDiv.innerHTML = '';
            }, 5000);
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Initialize
        addMessage('Hello! Upload a PDF document and start asking questions about it.', 'bot');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['pdf']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and file.filename.lower().endswith('.pdf'):
            # Read file as bytes
            pdf_bytes = file.read()
            
            # Process PDF
            result = chatbot.upload_pdf_from_bytes(pdf_bytes, file.filename)
            
            if result.startswith('Successfully'):
                return jsonify({'success': True, 'message': result})
            else:
                return jsonify({'success': False, 'message': result})
        else:
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload a PDF.'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        thread_id = data.get('thread_id', 'default')
        
        if not message:
            return jsonify({'response': 'Please provide a message'})
        
        # Get response from chatbot
        response = chatbot.chat(message, thread_id)
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    try:
        data = request.get_json()
        message = data.get('message')
        thread_id = data.get('thread_id', 'default')
        
        if not message:
            return jsonify({'response': 'Please provide a message'})
        
        def generate():
            for chunk in chatbot.stream_chat(message, thread_id):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        return generate(), {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

@app.route('/clear', methods=['POST'])
def clear_collection():
    try:
        result = chatbot.clear_collection()
        return jsonify({'success': True, 'message': result})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    print("Starting PDF Chatbot Server...")
    print("Make sure you have:")
    print("1. Qdrant server running (docker run -p 6333:6333 qdrant/qdrant)")
    print("2. Your Google API key set as environment variable")
    print("3. All required packages installed")
    print("\nAccess the app at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)