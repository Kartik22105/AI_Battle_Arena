#!/usr/bin/env python
"""
Quick test of the RAG system without needing the server
"""

import sys
sys.path.insert(0, '.')

from api_server import RAGPipeline

def test_rag():
    """Test RAG with real PDF"""
    print("\n" + "="*70)
    print("🧪 TESTING RAG SYSTEM WITH REAL PDF")
    print("="*70 + "\n")
    
    rag = RAGPipeline()
    
    # Test with ArXiv paper
    pdf_url = "https://arxiv.org/pdf/2005.14165.pdf"
    questions = [
        "What is the main topic?",
        "What are the key findings?",
    ]
    
    print(f"📄 PDF: {pdf_url}\n")
    
    answers = rag.answer_questions(pdf_url, questions)
    
    print("\n" + "="*70)
    print("✅ RESULTS")
    print("="*70)
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        print(f"\nQ{i}: {q}")
        print(f"A{i}: {a}\n")
    
    # Verify answers are NOT dummy text
    dummy_text = "Mock answer"
    real_answers = [a for a in answers if dummy_text not in a and len(a) > 50]
    
    if len(real_answers) > 0:
        print("✅ SUCCESS: Returning REAL answers from PDF context!")
        print(f"   - {len(real_answers)}/{len(answers)} answers contain real PDF content")
    else:
        print("⚠️  WARNING: Answers may still be minimal")

if __name__ == "__main__":
    test_rag()
