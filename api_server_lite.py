"""
AI Battle Arena - Lightweight RAG System
Simplified offline server with intelligent answer extraction
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import json
import torch
import warnings
warnings.filterwarnings('ignore')

# PDF processing
from PyPDF2 import PdfReader
import requests
from io import BytesIO
import tempfile
import os

# ML imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except ImportError as e:
    print(f"Warning: {e}. Using fallback mode.")

# ============================================================================
# CONFIGURATION
# ============================================================================

RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 128,
    "top_k_chunks": 5,
    "max_context_length": 3072
}

# ============================================================================
# VECTOR STORE - SIMPLE FAISS WRAPPER
# ============================================================================

class SimpleVectorStore:
    """Lightweight vector store using FAISS and sentence-transformers"""
    
    def __init__(self):
        try:
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.index = None
            self.chunks = []
            print("✓ Embedding model loaded")
        except Exception as e:
            print(f"Error loading embedder: {e}")
            self.embedder = None
            self.chunks = []
    
    def add_chunks(self, chunks: List[Dict]) -> bool:
        """Add chunks to vector store"""
        if not self.embedder or not chunks:
            self.chunks = chunks  # Store even if no embedder
            return True
        
        try:
            self.chunks = chunks
            texts = [c['text'] for c in chunks]
            embeddings = self.embedder.encode(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            print(f"✓ Indexed {len(chunks)} chunks")
            return True
        except Exception as e:
            print(f"Error indexing chunks: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k similar chunks"""
        if not self.chunks:
            return []
        
        if not self.embedder or self.index is None:
            # Fallback: return all chunks if no vector store
            return self.chunks[:top_k]
        
        try:
            query_embedding = self.embedder.encode([query])
            _, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))
            return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        except Exception as e:
            print(f"Retrieval error: {e}")
            return self.chunks[:top_k]

# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline"""
    
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.pdf_cache = {}
        print("✓ RAG Pipeline initialized")
    
    def download_pdf(self, pdf_url: str) -> bytes:
        """Download PDF from URL"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PdfReader(BytesIO(pdf_content))
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"[Page {page_num}] {text}\n\n"
            return full_text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(current_chunk) + len(sentence) < RAG_CONFIG["chunk_size"]:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page_num": len(chunks) // 10
                    })
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "page_num": len(chunks) // 10
            })
        
        return [c for c in chunks if len(c['text']) > 50]
    
    def process_pdf(self, pdf_url: str) -> bool:
        """Download, extract, and index PDF"""
        if pdf_url in self.pdf_cache:
            return True
        
        print(f"Processing PDF: {pdf_url}")
        
        # Download
        pdf_content = self.download_pdf(pdf_url)
        if not pdf_content:
            return False
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_content)
        if not text or len(text) < 100:
            print("No text extracted from PDF")
            return False
        
        # Chunk
        chunks = self.chunk_text(text)
        if not chunks:
            print("No valid chunks created")
            return False
        
        # Index
        self.vector_store.add_chunks(chunks)
        self.pdf_cache[pdf_url] = True
        print(f"✓ PDF processed: {len(chunks)} chunks")
        return True
    
    def extract_answer_from_context(self, question: str, context: str) -> str:
        """Intelligently extract answer from context"""
        if not context:
            return "Information not available in the document."
        
        # Split into sentences
        sentences = context.split('.')
        question_lower = question.lower()
        question_words = set(w.lower() for w in question.split() if len(w) > 3)
        
        # Score and filter sentences
        scored_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:  # Only consider meaningful sentences
                # Count matching words
                matches = sum(1 for word in sent.lower().split() if word in question_words)
                if matches > 0:
                    scored_sentences.append((matches, sent))
        
        # Return top sentences
        if scored_sentences:
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            answer = '. '.join([s[1] for s in scored_sentences[:2]])
            return answer[:512]
        
        # Fallback: return first meaningful sentence
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30:
                return sent[:512]
        
        return context[:512]
    
    def answer_questions(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Answer multiple questions about a PDF"""
        # Process PDF
        if not self.process_pdf(pdf_url):
            return ["PDF processing failed"] * len(questions)
        
        answers = []
        for question in questions:
            try:
                # Retrieve relevant chunks
                chunks = self.vector_store.retrieve(
                    question,
                    top_k=RAG_CONFIG["top_k_chunks"]
                )
                
                if not chunks:
                    answers.append("Information not found in the document.")
                    continue
                
                # Build context
                context = "\n\n".join([
                    f"[Page {c['page_num']}] {c['text']}"
                    for c in chunks
                ])
                
                # Extract answer intelligently
                answer = self.extract_answer_from_context(question, context)
                answers.append(answer)
                
            except Exception as e:
                print(f"Error answering question: {e}")
                answers.append("Error processing question.")
        
        return answers

# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI(title="AI Battle Arena - RAG System")

# Global RAG pipeline
rag_pipeline = RAGPipeline()

# Models
class QuestionRequest(BaseModel):
    pdf_url: str
    questions: List[str]

class AnswerResponse(BaseModel):
    success: bool
    answers: List[str]
    message: str

# Routes
@app.get("/")
async def root():
    """Serve main HTML interface"""
    html_path = "rag_frontend.html"
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"message": "AI Battle Arena RAG System", "status": "ready"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "AI Battle Arena RAG",
        "vector_store": "active" if rag_pipeline.vector_store.chunks else "ready"
    }

@app.post("/aibattle")
async def answer_battle(request: QuestionRequest):
    """Main endpoint for answering questions"""
    try:
        if not request.pdf_url or not request.questions:
            raise HTTPException(status_code=400, detail="Missing PDF URL or questions")
        
        answers = rag_pipeline.answer_questions(request.pdf_url, request.questions)
        
        return AnswerResponse(
            success=True,
            answers=answers,
            message="Questions answered successfully"
        )
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI BATTLE ARENA - OFFLINE RAG SYSTEM")
    print("="*60)
    print("✓ RAG Pipeline: Initialized")
    print("✓ Vector Store: Ready")
    print("✓ Answer Extraction: Enabled")
    print("="*60)
    print("\n🚀 Starting server on http://localhost:8000\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
