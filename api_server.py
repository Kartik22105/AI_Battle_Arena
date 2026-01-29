"""
AI Battle Arena - Ultra-lightweight RAG System
No external model dependencies - works offline immediately
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Minimal imports
from PyPDF2 import PdfReader
import requests
from io import BytesIO
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 128,
    "top_k_chunks": 5,
}

# ============================================================================
# SIMPLE TEXT-BASED RETRIEVAL
# ============================================================================

class SimpleTextRetrieval:
    """Ultra-simple text-based retrieval without ML"""
    
    def __init__(self):
        self.chunks = []
    
    def add_chunks(self, chunks: List[Dict]) -> bool:
        """Store chunks for retrieval"""
        self.chunks = chunks
        return True
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant chunks based on keyword matching"""
        if not self.chunks:
            return []
        
        query_words = set(w.lower() for w in query.split() if len(w) > 3)
        
        # Score chunks based on keyword overlap
        scored_chunks = []
        for chunk in self.chunks:
            text = chunk['text'].lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top-k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]

# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """Lightweight RAG without ML dependencies"""
    
    def __init__(self):
        self.retrieval = SimpleTextRetrieval()
        self.pdf_cache = {}
        print("✓ RAG Pipeline: Initialized (Text-based)")
    
    def download_pdf(self, pdf_url: str) -> bytes:
        """Download PDF from URL"""
        try:
            print(f"  Downloading: {pdf_url[:60]}...")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            print(f"  ✓ Downloaded {len(response.content)} bytes")
            return response.content
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PdfReader(BytesIO(pdf_content))
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        full_text += f"[Page {page_num}] {text}\n\n"
                except:
                    pass
            
            if full_text:
                print(f"  ✓ Extracted {len(full_text)} characters from PDF")
            return full_text
        except Exception as e:
            print(f"  ✗ Extraction error: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into meaningful chunks"""
        # Split by multiple delimiters
        chunks = []
        current_chunk = ""
        
        # First pass: split by periods
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(current_chunk) + len(sentence) < RAG_CONFIG["chunk_size"]:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk and len(current_chunk) > 30:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page_num": 0
                    })
                current_chunk = sentence
        
        if current_chunk and len(current_chunk) > 30:
            chunks.append({
                "text": current_chunk.strip(),
                "page_num": 0
            })
        
        return chunks
    
    def process_pdf(self, pdf_url: str) -> bool:
        """Download, extract, and index PDF"""
        if pdf_url in self.pdf_cache:
            print(f"  ✓ Using cached PDF")
            return True
        
        print(f"🔄 Processing PDF...")
        
        # Download
        pdf_content = self.download_pdf(pdf_url)
        if not pdf_content:
            return False
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_content)
        if not text or len(text) < 100:
            print("  ✗ No text extracted from PDF")
            return False
        
        # Chunk
        chunks = self.chunk_text(text)
        if not chunks:
            print("  ✗ No chunks created")
            return False
        
        # Index
        self.retrieval.add_chunks(chunks)
        self.pdf_cache[pdf_url] = True
        print(f"  ✓ Created {len(chunks)} chunks for indexing")
        return True
    
    def is_readable_text(self, text: str) -> bool:
        """Check if text is readable (not garbled/special characters)"""
        if len(text) < 10:
            return False
        
        # Count alphanumeric characters
        alpha_count = sum(1 for c in text if c.isalnum() or c.isspace())
        total_chars = len(text)
        
        # Text should be at least 60% alphanumeric/spaces
        return (alpha_count / total_chars) > 0.6
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing excessive special characters"""
        import re
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove standalone special characters
        text = re.sub(r'\s[^\w\s]\s', ' ', text)
        return text.strip()
    
    def extract_answer(self, question: str, context: str) -> str:
        """Extract best answer from context"""
        if not context:
            return "Information not available in the document."
        
        # Clean and split into sentences
        context = self.clean_text(context)
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        
        # Find question keywords
        question_words = set(w.lower() for w in question.split() if len(w) > 3)
        
        # Score sentences with readability check
        best_sentences = []
        for sent in sentences:
            if len(sent) < 20:  # Skip very short sentences
                continue
            
            # Skip unreadable text (garbled/special characters)
            if not self.is_readable_text(sent):
                continue
            
            sent_words = set(w.lower() for w in sent.split())
            matches = len(question_words & sent_words)
            
            # Calculate readability score
            words = sent.split()
            readable_words = sum(1 for w in words if w.isalnum() or any(c.isalpha() for c in w))
            readability = readable_words / len(words) if words else 0
            
            if matches > 0 and readability > 0.5:
                # Score combines keyword matches and readability
                score = matches * readability
                best_sentences.append((score, sent))
        
        # Return top matching sentences
        if best_sentences:
            best_sentences.sort(reverse=True, key=lambda x: x[0])
            answer = ". ".join([s[1] for s in best_sentences[:3]])
            if len(answer) > 512:
                answer = answer[:512] + "..."
            return answer
        
        # Fallback: find longest readable sentence
        readable_sentences = [s for s in sentences if len(s) > 30 and self.is_readable_text(s)]
        if readable_sentences:
            longest = max(readable_sentences, key=len)
            if len(longest) > 512:
                longest = longest[:512] + "..."
            return longest
        
        return "No relevant readable information found in the document."
    
    def answer_questions(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Answer multiple questions about a PDF"""
        # Process PDF
        if not self.process_pdf(pdf_url):
            return ["Failed to process PDF"] * len(questions)
        
        answers = []
        print(f"\n📝 Answering {len(questions)} questions...\n")
        
        for i, question in enumerate(questions, 1):
            try:
                print(f"  Q{i}: {question[:50]}...")
                
                # Retrieve relevant chunks
                chunks = self.retrieval.retrieve(
                    question,
                    top_k=RAG_CONFIG["top_k_chunks"]
                )
                
                if not chunks:
                    answer = "Information not found in the document."
                else:
                    # Build context from chunks
                    context = "\n".join([c['text'] for c in chunks])
                    
                    # Extract answer
                    answer = self.extract_answer(question, context)
                
                answers.append(answer)
                print(f"  A{i}: {answer[:60]}..." if len(answer) > 60 else f"  A{i}: {answer}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                answers.append(f"Error: {str(e)}")
        
        print()
        return answers

# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI(title="AI Battle Arena - Offline RAG")

# Global RAG
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
    """Serve HTML interface"""
    if os.path.exists("rag_frontend.html"):
        return FileResponse("rag_frontend.html", media_type="text/html")
    return {"message": "AI Battle Arena - RAG System", "status": "ready"}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "system": "AI Battle Arena RAG",
        "mode": "text-based retrieval"
    }

@app.post("/aibattle")
async def answer(request: QuestionRequest):
    """Answer questions endpoint"""
    try:
        if not request.pdf_url or not request.questions:
            raise HTTPException(status_code=400, detail="Missing PDF URL or questions")
        
        answers = rag_pipeline.answer_questions(request.pdf_url, request.questions)
        
        return {
            "success": True,
            "answers": answers,
            "message": "Questions answered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" 🤖 AI BATTLE ARENA - OFFLINE RAG SYSTEM")
    print("="*70)
    print(" Mode: Text-based retrieval (no ML dependencies)")
    print(" Features: PDF download, text extraction, intelligent answer extraction")
    print("="*70)
    print("\n 🌐 Server: http://localhost:8000")
    print(" 📖 Frontend: http://localhost:8000/")
    print(" 💚 Health: http://localhost:8000/health\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
