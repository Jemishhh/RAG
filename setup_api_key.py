#!/usr/bin/env python3
"""
Setup script for Google API key
"""

import os
import sys

def setup_api_key():
    """Help user set up Google API key"""
    print("🔑 Google API Key Setup")
    print("=" * 50)
    
    print("\n❌ Current API key has expired!")
    print("\n📋 To get a new API key:")
    print("1. Go to: https://aistudio.google.com/")
    print("2. Sign in with your Google account")
    print("3. Click 'Get API key'")
    print("4. Create a new API key")
    print("5. Copy the new key")
    
    print("\n🛠️ Setup Options:")
    print("A) Set as environment variable (recommended)")
    print("B) Update config files directly")
    print("C) Create .env file")
    
    choice = input("\nChoose option (A/B/C): ").upper()
    
    if choice == 'A':
        api_key = input("\nEnter your new API key: ").strip()
        if api_key:
            # Set environment variable for current session
            os.environ['GOOGLE_API_KEY'] = api_key
            print(f"✅ API key set for current session")
            print(f"🔑 Key: {api_key[:20]}...")
            print("\n💡 To make permanent, add to your system environment variables")
            
    elif choice == 'B':
        api_key = input("\nEnter your new API key: ").strip()
        if api_key:
            update_config_files(api_key)
            
    elif choice == 'C':
        api_key = input("\nEnter your new API key: ").strip()
        if api_key:
            create_env_file(api_key)
            
    else:
        print("❌ Invalid choice")
        return
    
    print("\n✅ Setup complete! Restart your application to use the new key.")

def update_config_files(api_key):
    """Update config files with new API key"""
    try:
        # Update enhanced_config.py
        with open('enhanced_config.py', 'r') as f:
            content = f.read()
        
        content = content.replace(
            "os.environ.get('GOOGLE_API_KEY', 'AIzaSyBkwMdGH4dFP8DZL_kCsO3GfsfbZMwAKh0')",
            f"'{api_key}'"
        )
        
        with open('enhanced_config.py', 'w') as f:
            f.write(content)
        
        # Update config.py
        with open('config.py', 'r') as f:
            content = f.read()
        
        content = content.replace(
            'os.environ.get("GOOGLE_API_KEY", "AIzaSyBkwMdGH4dFP8DZL_kCsO3GfsfbZMwAKh0")',
            f'"{api_key}"'
        )
        
        with open('config.py', 'w') as f:
            f.write(content)
        
        print(f"✅ Config files updated with new API key")
        print(f"🔑 Key: {api_key[:20]}...")
        
    except Exception as e:
        print(f"❌ Error updating config files: {e}")

def create_env_file(api_key):
    """Create .env file with API key"""
    try:
        env_content = f"GOOGLE_API_KEY={api_key}\n"
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print(f"✅ .env file created with API key")
        print(f"🔑 Key: {api_key[:20]}...")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def test_api_key():
    """Test if current API key works"""
    print("\n🧪 Testing API key...")
    
    try:
        from enhanced_pdf_chatbot import EnhancedPDFChatbot
        from enhanced_config import get_enhanced_chatbot_config
        
        config = get_enhanced_chatbot_config()
        print(f"🔑 Using key: {config['google_api_key'][:20]}...")
        
        chatbot = EnhancedPDFChatbot(**config)
        print("✅ API key is valid!")
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")

if __name__ == "__main__":
    setup_api_key()
    
    # Test the key if user wants
    test_choice = input("\nTest the API key? (y/n): ").lower()
    if test_choice == 'y':
        test_api_key() 