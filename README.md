# AI Battle Arena - Offline RAG System

## 🎯 Overview

Complete **offline** Question-Answering system for PDF documents using:
- **Local LLM**: Llama-3.1-8B-Instruct (4-bit quantized)
- **RAG**: FAISS + SentenceTransformers
- **PDF Processing**: PyPDF2 + Tesseract OCR
- **API**: FastAPI with POST /aibattle endpoint

**Key Feature**: 100% offline - NO external API calls

## 🏗️ Architecture

```
Request → Download PDF → Extract Text → Chunk → Embed → FAISS Search
                                                              ↓
Response ← Generate Answer ← Local LLM ← Context ← Top-K Chunks
```

## 📋 Requirements

- **GPU**: 12-16GB VRAM (NVIDIA with CUDA)
- **RAM**: 16GB+
- **Python**: 3.8+
- **CUDA**: 11.8+ (for PyTorch)

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Install Tesseract (for OCR)

Download from: https://github.com/UB-Mannheim/tesseract/wiki

Add to PATH: `C:\Program Files\Tesseract-OCR`

### 3. Start the Server

```powershell
python api_server.py
```

The server will:
- Load Llama-3.1-8B-Instruct model (4-bit, ~5GB VRAM)
- Initialize FAISS and embeddings
- Start listening on http://0.0.0.0:8000

**Startup time**: ~2-3 minutes (one-time model loading)

## 📡 API Usage

### Endpoint: POST /aibattle

**Request:**
```json
{
  "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
  "questions": [
    "What is the title of this paper?",
    "Who are the authors?",
    "What is the main contribution?",
    "What architecture is proposed?",
    "What datasets were used?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Attention Is All You Need",
    "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
    "The paper proposes the Transformer, a novel architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions.",
    "The Transformer architecture uses multi-head self-attention and position-wise fully connected layers.",
    "WMT 2014 English-to-German and English-to-French translation tasks"
  ]
}
```

### Test with cURL

```powershell
curl -X POST "http://localhost:8000/aibattle" `
  -H "Content-Type: application/json" `
  -d '{\"pdf_url\": \"https://arxiv.org/pdf/1706.03762.pdf\", \"questions\": [\"What is the title?\", \"Who are the authors?\", \"What is the main contribution?\", \"What architecture is proposed?\", \"What datasets were used?\"]}'
```

### Health Check

```powershell
curl http://localhost:8000/health
```

## 🧪 Testing with Notebook

1. Open `ai_battle_arena_rag_system (1).ipynb`
2. Run cells sequentially
3. Use the final test cell to validate the complete system

## 🎓 Training (Optional)

The system works with the base Llama-3.1-8B-Instruct model. For improved performance:

1. **Load your dataset** (already configured for `pdf_qa_finetune.jsonl`)
2. **Run training cells 7-11** in the notebook
3. **Update `api_server.py`**: Set `LORA_PATH = "./final_lora_model"`
4. **Restart server**

Training takes ~30-60 minutes on T4 GPU with the provided 36 examples.

## 📂 Project Structure

```
.
├── api_server.py              # Complete standalone server
├── requirements.txt           # Python dependencies
├── pdf_qa_finetune.jsonl     # Training dataset (36 examples)
├── ai_battle_arena_rag_system (1).ipynb  # Full notebook
└── final_lora_model/         # Fine-tuned model (after training)
```

## ⚙️ Configuration

Edit `api_server.py` constants:

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LORA_PATH = "./final_lora_model"  # Set to None to skip LoRA

RAG_CONFIG = {
    "chunk_size": 512,        # Tokens per chunk
    "chunk_overlap": 128,     # Overlap between chunks
    "top_k_chunks": 5,        # Retrieve top-5 chunks
    "max_context_length": 3072  # Max context for LLM
}
```

## 🔧 Troubleshooting

### CUDA Out of Memory
- Ensure no other GPU processes are running
- Reduce `chunk_size` or `top_k_chunks`
- Use smaller model: `meta-llama/Llama-2-7b-chat-hf`

### Slow Inference
- Check GPU is being used: visit http://localhost:8000/health
- Ensure CUDA is properly installed
- Reduce `max_new_tokens` in generation

### PDF Download Fails
- Check internet connection
- Try a different PDF URL
- Ensure URL is direct PDF link (not webpage)

### Tesseract Not Found
- Install from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`
- Restart terminal

## 🎯 Competition Compliance

✅ **Fully Offline** - No external API calls  
✅ **Local LLM** - Llama-3.1-8B-Instruct  
✅ **POST /aibattle** - Correct endpoint  
✅ **Valid JSON** - Guaranteed format  
✅ **Error Handling** - Graceful degradation  
✅ **Context-Only Answers** - No hallucination  

## 📊 Performance

- **Startup**: ~2-3 minutes (one-time)
- **Per Request**: ~5-15 seconds (5 questions)
- **Memory**: ~5-7GB VRAM
- **Concurrent Requests**: Supported (queued)

## 🔒 Security

- No API keys required
- No external services
- All processing local
- No data persistence (unless cached)

## 📝 License

Competition submission - Educational use

## 🙏 Acknowledgments

- Meta AI (Llama-3.1)
- HuggingFace (Transformers, PEFT)
- Facebook Research (FAISS)
- Sentence-Transformers team

---

**Ready for competition! 🏆**

For questions or issues, check the notebook for detailed cell-by-cell explanations.
# kartikcblock
