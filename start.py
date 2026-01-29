#!/usr/bin/env python3
"""
AI Battle Arena - Simple Python Launcher
Run this to start the entire system
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("🚀 AI BATTLE ARENA - LAUNCHER".center(80))
    print("="*80 + "\n")
    
    # Check if we're in the right directory
    if not Path("api_server.py").exists():
        print("❌ ERROR: api_server.py not found!")
        print("   Make sure you're in the project root directory")
        sys.exit(1)
    
    print("✅ Project files found")
    print("✅ Starting server...\n")
    
    try:
        # Start the server
        subprocess.run([sys.executable, "api_server.py"])
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
