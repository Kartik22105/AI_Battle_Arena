"""
AI Battle Arena - Offline RAG System
Complete standalone server with local LLM inference
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import uvicorn
import torch
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# PDF processing
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import requests
from io import BytesIO
import tempfile
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LORA_PATH = "./final_lora_model"  # Set to None if not using fine-tuned model

RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 128,
    "top_k_chunks": 5,
    "max_context_length": 3072
}

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# ============================================================================
# PDF PROCESSOR
# ============================================================================

class PDFProcessor:
    """Extract and chunk text from PDF with page tracking."""
    
    def __init__(self, chunk_size=512, overlap=128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"PDF download failed: {e}")
            return None
    
    def extract_text(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text page by page."""
        if not pdf_bytes:
            return []
            
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            pages = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "page_num": page_num,
                        "text": text.strip(),
                        "type": "text"
                    })
            
            return pages
        except Exception as e:
            print(f"Text extraction failed: {e}")
            return []
    
    def chunk_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with page info."""
        chunks = []
        
        for page in pages:
            text = page["text"]
            words = text.split()
            
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    "text": chunk_text,
                    "page_num": page["page_num"],
                    "chunk_id": len(chunks)
                })
        
        return chunks

# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore:
    """FAISS-based retrieval system."""
    
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {embedding_model_name}")
        self.encoder = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from text chunks."""
        if not chunks:
            print("Warning: No chunks to index")
            return
            
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"✅ FAISS index built: {len(chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks."""
        if not self.index or not self.chunks:
            return []
            
        query_embedding = self.encoder.encode([query]).astype('float32')
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    **self.chunks[idx],
                    "score": float(dist)
                })
        
        return results

# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """Complete RAG system with local LLM."""
    
    def __init__(self, model, tokenizer, vector_store, pdf_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.pdf_processor = pdf_processor
        self.pdf_cache = {}
        
    def process_pdf(self, pdf_url: str) -> bool:
        """Download and process PDF, build FAISS index."""
        if pdf_url in self.pdf_cache:
            return True
        
        # Download PDF
        pdf_bytes = self.pdf_processor.download_pdf(pdf_url)
        if not pdf_bytes:
            return False
        
        # Extract and chunk text
        pages = self.pdf_processor.extract_text(pdf_bytes)
        if not pages:
            return False
            
        chunks = self.pdf_processor.chunk_text(pages)
        if not chunks:
            return False
        
        # Build FAISS index
        self.vector_store.build_index(chunks)
        
        # Cache success
        self.pdf_cache[pdf_url] = True
        return True
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using local LLM."""
        
        # Simple prompt format for better compatibility
        prompt = f"""You are a document question-answering assistant. Answer ONLY from the provided context.

Context:
{context}

Question: {question}

Answer (be concise and factual):"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=RAG_CONFIG["max_context_length"]
        ).to(self.model.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (everything after "Answer:")
            if "Answer" in response:
                answer = response.split("Answer")[-1].strip()
                answer = answer.lstrip(":(be concise and factual):").strip()
                return answer if answer else "Information not found in the document"
            else:
                return response.strip()
                
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error generating answer"
    
    def answer_questions(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Answer multiple questions for a PDF."""
        # Process PDF
        if not self.process_pdf(pdf_url):
            return ["Information not found in the document"] * len(questions)
        
        answers = []
        for question in questions:
            try:
                # Retrieve relevant chunks
                chunks = self.vector_store.retrieve(
                    question,
                    top_k=RAG_CONFIG["top_k_chunks"]
                )
                
                if not chunks:
                    answers.append("Information not found in the document")
                    continue
                
                # Build context
                context = "\n\n".join([
                    f"[Page {c['page_num']}] {c['text']}"
                    for c in chunks
                ])
                
                # Generate answer
                answer = self.generate_answer(question, context)
                answers.append(answer)
                
            except Exception as e:
                print(f"Error processing question: {e}")
                answers.append("Error processing question")
        
        return answers

# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI(title="AI Battle Arena - Offline RAG System")

# Global RAG pipeline instance
rag_pipeline = None

class QuestionRequest(BaseModel):
    pdf_url: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

@app.on_event("startup")
async def startup_event():
    """Load model and initialize pipeline on server startup."""
    global rag_pipeline
    
    print("=" * 80)
    print("LOADING AI BATTLE ARENA SYSTEM")
    print("=" * 80)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with 4-bit quantization
    print(f"Loading model: {MODEL_NAME} (4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Load LoRA weights if available
    if LORA_PATH and os.path.exists(LORA_PATH):
        print(f"Loading LoRA weights: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    
    model.eval()
    
    # Initialize components
    print("\nInitializing RAG components...")
    vector_store = VectorStore()
    pdf_processor = PDFProcessor(
        chunk_size=RAG_CONFIG["chunk_size"],
        overlap=RAG_CONFIG["chunk_overlap"]
    )
    
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        model=model,
        tokenizer=tokenizer,
        vector_store=vector_store,
        pdf_processor=pdf_processor
    )
    
    print("\n" + "=" * 80)
    print("✅ SYSTEM READY - Server listening on http://0.0.0.0:8000")
    print("=" * 80)

@app.post("/aibattle", response_model=AnswerResponse)
async def answer_questions(request: QuestionRequest):
    """Main competition endpoint."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Validate input
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        if len(request.questions) < 1:
            raise HTTPException(status_code=400, detail="At least 1 question required")
        
        # Process request
        answers = rag_pipeline.answer_questions(
            request.pdf_url,
            request.questions
        )
        
        # Ensure we return same number of answers as questions
        while len(answers) < len(request.questions):
            answers.append("Information not found in the document")
        
        return AnswerResponse(answers=answers[:len(request.questions)])
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if rag_pipeline else "initializing",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AI Battle Arena - Offline RAG System",
        "endpoint": "POST /aibattle",
        "status": "operational" if rag_pipeline else "initializing"
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
