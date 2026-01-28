"""
Test script for AI Battle Arena API
Run this after starting api_server.py
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("=" * 80)
    print("Testing /health endpoint...")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_aibattle():
    """Test /aibattle endpoint with sample PDF."""
    print("\n" + "=" * 80)
    print("Testing /aibattle endpoint...")
    print("=" * 80)
    
    # Sample request
    request_data = {
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
        "questions": [
            "What is the title of this paper?",
            "Who are the authors?",
            "What is the main contribution of this work?",
            "What architecture is proposed?",
            "What datasets were used for experiments?"
        ]
    }
    
    print(f"\nSending request...")
    print(f"PDF: {request_data['pdf_url']}")
    print(f"Questions: {len(request_data['questions'])}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/aibattle",
            json=request_data,
            timeout=120
        )
        elapsed = time.time() - start_time
        
        print(f"\nStatus: {response.status_code}")
        print(f"Response time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success! Got {len(data['answers'])} answers\n")
            
            for i, (q, a) in enumerate(zip(request_data['questions'], data['answers']), 1):
                print(f"{i}. Q: {q}")
                print(f"   A: {a}\n")
            
            # Validate JSON format
            try:
                json_str = json.dumps(data)
                json.loads(json_str)
                print("✅ JSON format is valid")
                return True
            except:
                print("❌ JSON format is INVALID")
                return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>120s)")
        return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_invalid_request():
    """Test with invalid request."""
    print("\n" + "=" * 80)
    print("Testing invalid request handling...")
    print("=" * 80)
    
    # Request with no questions
    request_data = {
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
        "questions": []
    }
    
    try:
        response = requests.post(
            f"{API_URL}/aibattle",
            json=request_data,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 400:
            print("✅ Correctly rejected invalid request")
            return True
        else:
            print("⚠️  Should return 400 for empty questions")
            return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    print("=" * 80)
    print("AI BATTLE ARENA API TEST SUITE")
    print("=" * 80)
    print(f"\nTesting API at: {API_URL}")
    print("Make sure api_server.py is running!\n")
    
    results = {
        "health": False,
        "aibattle": False,
        "validation": False
    }
    
    # Run tests
    results["health"] = test_health()
    
    if results["health"]:
        time.sleep(1)
        results["aibattle"] = test_aibattle()
        time.sleep(1)
        results["validation"] = test_invalid_request()
    else:
        print("\n❌ Health check failed. Is the server running?")
        print("   Start with: python api_server.py")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Health Check:       {'✅ PASS' if results['health'] else '❌ FAIL'}")
    print(f"Main Endpoint:      {'✅ PASS' if results['aibattle'] else '❌ FAIL'}")
    print(f"Error Handling:     {'✅ PASS' if results['validation'] else '❌ FAIL'}")
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED - System ready for competition!")
    else:
        print("\n⚠️  Some tests failed - Check server logs")

if __name__ == "__main__":
    main()
