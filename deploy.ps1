
# deploy.ps1 - Production deployment script for Windows

Write-Host "🚀 Deploying AI Battle Arena System..." -ForegroundColor Green

# 1. Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0 peft==0.7.1 bitsandbytes==0.41.3
pip install accelerate==0.25.0 datasets==2.16.0 sentencepiece==0.1.99
pip install faiss-cpu==1.7.4 sentence-transformers==2.2.2
pip install pypdf2==3.0.1 pdf2image==1.16.3 pillow==10.1.0
pip install fastapi==0.109.0 uvicorn==0.27.0 pydantic==2.5.3
pip install pytesseract==0.3.10 requests==2.31.0

# 2. Download model (if not cached)
Write-Host "📥 Checking model..." -ForegroundColor Yellow
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"

# 3. Apply torch optimizations
$env:TORCH_CUDNN_V8_API_ENABLED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# 4. Start server with optimizations
Write-Host "🚀 Starting server..." -ForegroundColor Green
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 300 --log-level info
