#!/usr/bin/env python3
"""
One-time model download script.
Downloads Llama-3.1-8B-Instruct to local cache.
Run this ONCE in a terminal, then the notebook will work in offline mode.
"""

import os
import sys

print("=" * 80)
print("🔵 DOWNLOADING LLAMA-3.1-8B-INSTRUCT MODEL")
print("=" * 80)
print("\nThis is a ONE-TIME operation (takes 5-10 minutes).")
print("After this completes, the notebook will work in offline mode.\n")

# Ensure network access is available
import socket
try:
    socket.create_connection(("huggingface.co", 443), timeout=5)
    print("✅ Network connection to HuggingFace verified\n")
except OSError as e:
    print(f"❌ Cannot reach HuggingFace: {e}")
    print("   Please check your internet connection and try again.")
    sys.exit(1)

# Download tokenizer
print("📥 Downloading tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True
    )
    print("✅ Tokenizer downloaded and cached\n")
except Exception as e:
    print(f"❌ Failed to download tokenizer: {e}")
    sys.exit(1)

# Download model
print("📥 Downloading model (this may take 10-15 minutes)...")
print("   Model size: ~16 GB")
try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="cpu",  # Download to CPU first, then notebook can use GPU
        trust_remote_code=True
    )
    print("✅ Model downloaded and cached\n")
except Exception as e:
    print(f"❌ Failed to download model: {e}")
    sys.exit(1)

# Download embedding model
print("📥 Downloading sentence-transformers/all-MiniLM-L6-v2...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model downloaded and cached\n")
except Exception as e:
    print(f"❌ Failed to download embedding model: {e}")
    sys.exit(1)

print("=" * 80)
print("✅ ALL MODELS CACHED SUCCESSFULLY")
print("=" * 80)
print("\nNow you can:")
print("1. Close this terminal")
print("2. Return to the notebook")
print("3. Re-run Cell 8 (it will use the local cached models)")
print("\n✨ The notebook will now work in OFFLINE MODE!")
print("=" * 80)
