"""
Test script to check JSON format from backend
"""
import requests
import json

# Test the API endpoint
url = "http://localhost:8000/aibattle"

# Simple test payload
payload = {
    "pdf_url": "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-12r1.pdf",
    "questions": ["What is computer security?"]
}

print("=" * 70)
print("TESTING BACKEND JSON FORMAT")
print("=" * 70)
print(f"\n📤 REQUEST:")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload)
    
    print(f"\n📥 RESPONSE:")
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    
    print(f"\n📄 RAW RESPONSE:")
    print(response.text)
    
    print(f"\n✅ PARSED JSON:")
    response_json = response.json()
    print(json.dumps(response_json, indent=2))
    
    print(f"\n🔍 RESPONSE STRUCTURE:")
    print(f"Keys in response: {list(response_json.keys())}")
    print(f"Type of 'answers': {type(response_json.get('answers'))}")
    if 'answers' in response_json:
        print(f"Number of answers: {len(response_json['answers'])}")
        print(f"\nFirst answer preview: {response_json['answers'][0][:200]}...")
    
    print("\n" + "=" * 70)
    print("✅ BACKEND IS RETURNING VALID JSON FORMAT")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("=" * 70)
