#!/usr/bin/env python3
"""
Test the spelling correction fix
"""

from enhanced_pdf_chatbot import EnhancedPDFChatbot
from enhanced_config import get_enhanced_chatbot_config
from langchain_core.messages import HumanMessage

def test_fix():
    """Test that spelling correction is disabled"""
    print("Testing Spelling Correction Fix")
    print("=" * 40)
    
    try:
        # Initialize chatbot with new config
        config = get_enhanced_chatbot_config()
        chatbot = EnhancedPDFChatbot(**config)
        print("Chatbot initialized")
        
        # Check if spelling correction is disabled
        print(f"Spelling correction enabled: {chatbot.query_config.enable_spelling_correction}")
        
        if not chatbot.query_config.enable_spelling_correction:
            print("✅ Spelling correction is disabled")
        else:
            print("❌ Spelling correction is still enabled")
        
        # Add NBA website
        nba_url = "https://www.nba.com/news/all-30-nba-arenas-by-team"
        result = chatbot.add_single_webpage(nba_url)
        print(f"Website added: {result}")
        
        # Test query without spelling correction
        test_query = "What are the NBA arenas mentioned?"
        messages = [HumanMessage(content=test_query)]
        
        # Get the latest human message
        latest_human_msg = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        # Check if spelling correction would be applied
        if chatbot.query_config.enable_spelling_correction:
            corrected_query = chatbot.spelling_corrector.correct_spelling(latest_human_msg)
            if corrected_query != latest_human_msg:
                print(f"❌ Query would be corrected: '{latest_human_msg}' -> '{corrected_query}'")
            else:
                print(f"✅ Query would not be corrected: '{latest_human_msg}'")
        else:
            print(f"✅ Query will not be corrected: '{latest_human_msg}'")
        
        # Test actual query
        print("Testing actual query...")
        response = chatbot.query(messages)
        
        if isinstance(response, dict):
            print(f"Response sources count: {len(response.get('sources', []))}")
            if response.get('sources'):
                print("✅ SUCCESS! Sources found!")
                for i, source in enumerate(response['sources']):
                    print(f"Source {i+1}: {source}")
            else:
                print("❌ No sources in response")
        else:
            print(f"Response is string: {response[:200]}...")
        
        print("Test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fix() 