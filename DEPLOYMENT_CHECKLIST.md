# AI Battle Arena - Deployment Checklist

## 📋 Pre-Deployment (48 hours before competition)

### Environment Setup
- [ ] GPU server with 12-16GB VRAM available
- [ ] CUDA 11.8+ installed
- [ ] Python 3.8+ installed
- [ ] Internet connection for downloading model and PDFs
- [ ] Port 8000 is open and available

### Software Installation
- [ ] Clone/copy project to server
- [ ] Install PyTorch with CUDA:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- [ ] Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Install Tesseract OCR (Windows):
  - Download from: https://github.com/UB-Mannheim/tesseract/wiki
  - Add to PATH: `C:\Program Files\Tesseract-OCR`

### Model Download (One-Time, ~30GB)
- [ ] Run once to download model:
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
  AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
  ```
- [ ] Verify model cached in: `~/.cache/huggingface/hub/`

### Optional: Fine-Tuning
- [ ] Open notebook: `ai_battle_arena_rag_system (1).ipynb`
- [ ] Run cells 1-6 (setup and config)
- [ ] Run cell 7 (loads training data from pdf_qa_finetune.jsonl)
- [ ] Run cells 8-11 (model training, ~30-60 min)
- [ ] Verify `./final_lora_model/` folder created
- [ ] Update api_server.py: `LORA_PATH = "./final_lora_model"`

## 🧪 Testing

### Unit Tests
- [ ] Start server: `python api_server.py`
- [ ] Wait for: "✅ SYSTEM READY"
- [ ] Run tests: `python test_api.py`
- [ ] All tests should PASS

### Sample PDFs
Test with different PDF types:
- [ ] Academic paper (arxiv.org)
- [ ] Technical documentation
- [ ] Image-heavy PDF (OCR test)
- [ ] Large PDF (50+ pages)
- [ ] Small PDF (1-5 pages)

### Performance Benchmarks
- [ ] First request latency: <30s (includes PDF download + processing)
- [ ] Subsequent requests: <15s (PDF cached)
- [ ] 5 questions per request: <60s total
- [ ] GPU memory usage: <14GB
- [ ] CPU usage: <80%

### Edge Cases
- [ ] Empty PDF
- [ ] Invalid PDF URL
- [ ] PDF with no text (images only)
- [ ] Very long questions (>500 chars)
- [ ] Questions with no relevant context
- [ ] Concurrent requests (2-3 simultaneous)

## 🚀 Competition Day

### Pre-Competition (1 hour before)
- [ ] Server is powered on
- [ ] GPU is detected: `nvidia-smi`
- [ ] Start api_server.py
- [ ] Verify startup completes successfully
- [ ] Check health endpoint: `curl http://localhost:8000/health`
- [ ] Run one test request to warm up system
- [ ] Monitor GPU memory: should be ~5-7GB

### During Competition
- [ ] Keep terminal with server logs visible
- [ ] Monitor GPU memory with `nvidia-smi`
- [ ] Watch for error patterns in logs
- [ ] Track response times
- [ ] Note any unusual requests
- [ ] Have backup plan ready

### Monitoring Commands
```powershell
# GPU monitoring (run in separate terminal)
watch -n 1 nvidia-smi

# Server logs
# (already visible in api_server.py terminal)

# Test health
curl http://localhost:8000/health
```

## 🛠️ Troubleshooting

### Server Won't Start
1. Check CUDA: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check port: `netstat -ano | findstr :8000`
4. Check logs for error messages

### Out of Memory
1. Kill other GPU processes
2. Reduce `top_k_chunks` from 5 to 3 in api_server.py
3. Reduce `chunk_size` from 512 to 256
4. Restart server

### Slow Responses
1. Verify GPU is being used (check /health endpoint)
2. Check if PDF is being re-downloaded (should be cached)
3. Reduce `max_new_tokens` from 256 to 128
4. Check internet speed for PDF downloads

### PDF Download Fails
- Already handled in code (returns error message)
- Check internet connection
- Verify PDF URL is valid and accessible

### Invalid JSON Output
- Code already validates JSON before returning
- Should never happen with current implementation
- If occurs, check LLM output in logs

## ✅ Final Checklist (30 min before competition)

- [ ] Server is running
- [ ] Health check returns 200
- [ ] Test request succeeds
- [ ] GPU memory is stable
- [ ] Logs show no errors
- [ ] Internet connection is stable
- [ ] Have competition organizers' contact info ready
- [ ] Screenshot of successful test request saved
- [ ] Backup server ready (if available)

## 🎯 Success Criteria

### Must Have
✅ Server responds to /aibattle POST requests  
✅ Returns valid JSON format  
✅ Answers are relevant to questions  
✅ No external API calls (100% offline after model download)  
✅ Response time <2 minutes per request  
✅ Server stays up for entire competition  

### Nice to Have
⭐ Answers are highly accurate  
⭐ Response time <30s per request  
⭐ Handles 10+ concurrent requests  
⭐ Perfect uptime (no crashes)  

## 📞 Emergency Contacts

- Competition Organizers: _________________
- Team Lead: _________________
- Server Admin: _________________

## 📝 Notes Section

Use this space to note any issues, observations, or improvements during testing:

```
Date: ___________
Notes:





```

---

**Good luck! 🏆**

Remember:
- Test early and often
- Monitor continuously
- Stay calm
- Trust your preparation
