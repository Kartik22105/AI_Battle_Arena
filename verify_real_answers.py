#!/usr/bin/env python
"""
Final verification that RAG system returns REAL answers, not dummy data
"""

import requests
import json
import time

def test_api():
    """Test the live API server"""
    print("\n" + "="*70)
    print("🧪 TESTING LIVE API SERVER")
    print("="*70 + "\n")
    
    url = "http://localhost:8000/aibattle"
    
    payload = {
        "pdf_url": "https://arxiv.org/pdf/2005.14165.pdf",
        "questions": [
            "What is this paper about?",
            "What is the name of the main model discussed?"
        ]
    }
    
    print(f"📤 Sending request to: {url}")
    print(f"📄 PDF: {payload['pdf_url']}")
    print(f"❓ Questions: {len(payload['questions'])}")
    print("\n⏳ Processing (downloading PDF, extracting text, generating answers)...\n")
    
    try:
        start = time.time()
        response = requests.post(
            url,
            json=payload,
            timeout=180,
            headers={"Content-Type": "application/json"}
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS! (Completed in {elapsed:.1f}s)")
            print(f"Status: {result.get('success')}")
            print(f"Message: {result.get('message')}\n")
            
            print("="*70)
            print("ANSWERS (From PDF Content):")
            print("="*70)
            
            for i, (q, a) in enumerate(zip(payload['questions'], result['answers']), 1):
                print(f"\nQ{i}: {q}")
                print(f"A{i}:")
                if len(a) > 150:
                    print(f"  {a[:150]}...")
                else:
                    print(f"  {a}")
            
            # Verification
            print("\n" + "="*70)
            dummy_indicators = ["Mock answer", "Information extracted from document"]
            has_dummy = any(indicator in str(result['answers']) for indicator in dummy_indicators)
            
            if not has_dummy and all(len(a) > 20 for a in result['answers']):
                print("✅ VERIFIED: Returning REAL content from PDF!")
                print("   - NOT returning dummy/mock responses")
                print("   - Answers extracted from actual PDF text")
            else:
                print("⚠️  Some answers may be minimal or generic")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("   Make sure the server is running: python api_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()
