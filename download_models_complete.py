#!/usr/bin/env python3
"""
Complete model download script.
Downloads ALL required models to local cache.
RUN THIS IN A TERMINAL WITH INTERNET ACCESS.
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("🔵 DOWNLOADING REQUIRED MODELS")
print("=" * 80)
print("\nThis downloads the complete models (~30GB total)")
print("Takes 10-20 minutes depending on your connection.\n")

# Test network
print("Testing network connectivity...")
import socket
try:
    socket.create_connection(("huggingface.co", 443), timeout=10)
    print("✅ Connected to HuggingFace\n")
except OSError as e:
    print(f"❌ Cannot reach HuggingFace: {e}")
    print("Please check your internet connection.")
    sys.exit(1)

cache_dir = os.path.expanduser(r"~\.cache\huggingface\hub")
print(f"Cache directory: {cache_dir}\n")

try:
    # Import after testing network
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    
    # Download Llama model completely
    print("=" * 80)
    print("1. DOWNLOADING LLAMA-3.1-8B-INSTRUCT MODEL (~25GB)")
    print("=" * 80)
    
    print("\n   Downloading tokenizer...")
    try:
        AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("   ✅ Tokenizer cached")
    except Exception as e:
        print(f"   ⚠️  Tokenizer download: {e}")
    
    print("\n   Downloading model weights (this takes several minutes)...")
    try:
        AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=cache_dir,
            device_map="cpu",  # Don't load to GPU yet
            trust_remote_code=True
        )
        print("   ✅ Model weights cached")
    except Exception as e:
        print(f"   ❌ Model download failed: {e}")
        print("   This might be a HuggingFace permission issue.")
        print("   Make sure you have access to meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)
    
    # Download embedding model
    print("\n" + "=" * 80)
    print("2. DOWNLOADING SENTENCE-TRANSFORMERS EMBEDDING MODEL (~500MB)")
    print("=" * 80)
    
    print("\n   Downloading sentence-transformers/all-MiniLM-L6-v2...")
    try:
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("   ✅ Embedding model cached")
    except Exception as e:
        print(f"   ⚠️  Embedding model: {e}")
    
    # Verify downloads
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    llama_cache = os.path.join(cache_dir, "models--meta-llama--Llama-3.1-8B-Instruct")
    if os.path.exists(llama_cache):
        snapshots = os.path.join(llama_cache, "snapshots")
        if os.path.exists(snapshots):
            snapshot_dirs = [d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))]
            if snapshot_dirs:
                model_dir = os.path.join(snapshots, snapshot_dirs[0])
                files = os.listdir(model_dir)
                safetensors = [f for f in files if "safetensors" in f]
                print(f"\n✅ Llama model cache found")
                print(f"   Total files: {len(files)}")
                print(f"   Safetensors files: {len(safetensors)}")
                if len(safetensors) > 0:
                    print(f"   ✓ Model weights are present!")
                    
    embedding_cache = os.path.join(cache_dir, "models--sentence-transformers--all-MiniLM-L6-v2")
    if os.path.exists(embedding_cache):
        print(f"\n✅ Embedding model cache found")
    
    print("\n" + "=" * 80)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nYou can now:")
    print("1. Close this terminal")
    print("2. Return to the notebook")
    print("3. Re-run Cell 8 (model loading)")
    print("4. The notebook will work in OFFLINE MODE!")
    print("\n✨ Models are cached locally - no network needed after this!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Download failed: {type(e).__name__}: {e}")
    print("\nIf you don't have HuggingFace access:")
    print("1. Ask an admin with internet access to run this script")
    print("2. Copy the ~/.cache/huggingface/hub folder from that machine")
    print("3. Place it in your home directory on this machine")
    sys.exit(1)
