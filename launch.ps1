# AI Battle Arena - Quick Launcher
# Run this script to start the server with all checks

Write-Host "=" -NoNewline; 1..79 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "AI BATTLE ARENA - SYSTEM LAUNCHER" -ForegroundColor Green
Write-Host "=" -NoNewline; 1..79 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host ""

# Check Python
Write-Host "1. Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python not found! Install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check CUDA
Write-Host "`n2. Checking CUDA/GPU..." -ForegroundColor Yellow
try {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) {
        $gpu = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        Write-Host "   ✅ GPU: $gpu" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  No NVIDIA GPU detected - will use CPU (slow)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️  nvidia-smi not found - GPU may not be available" -ForegroundColor Yellow
}

# Check dependencies
Write-Host "`n3. Checking dependencies..." -ForegroundColor Yellow
$required = @("torch", "transformers", "fastapi", "faiss", "sentence_transformers")
$missing = @()

foreach ($pkg in $required) {
    $check = python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing += $pkg
    }
}

if ($missing.Count -eq 0) {
    Write-Host "   ✅ All required packages installed" -ForegroundColor Green
} else {
    Write-Host "   ❌ Missing packages: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "   Run: pip install -r requirements.txt" -ForegroundColor Yellow
    
    $install = Read-Host "`n   Install now? (y/N)"
    if ($install -eq 'y' -or $install -eq 'Y') {
        Write-Host "`n   Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   ❌ Installation failed!" -ForegroundColor Red
            exit 1
        }
    } else {
        exit 1
    }
}

# Check Tesseract
Write-Host "`n4. Checking Tesseract OCR..." -ForegroundColor Yellow
try {
    tesseract --version 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Tesseract installed" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Tesseract not found (OCR won't work)" -ForegroundColor Yellow
        Write-Host "   Download from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️  Tesseract not found (OCR won't work)" -ForegroundColor Yellow
}

# Check port
Write-Host "`n5. Checking port 8000..." -ForegroundColor Yellow
$portCheck = netstat -ano | Select-String ":8000"
if ($portCheck) {
    Write-Host "   ⚠️  Port 8000 is already in use!" -ForegroundColor Yellow
    Write-Host "   Kill the process or change the port in api_server.py" -ForegroundColor Yellow
    $continue = Read-Host "`n   Continue anyway? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        exit 1
    }
} else {
    Write-Host "   ✅ Port 8000 is available" -ForegroundColor Green
}

# Check api_server.py exists
Write-Host "`n6. Checking api_server.py..." -ForegroundColor Yellow
if (Test-Path "api_server.py") {
    Write-Host "   ✅ api_server.py found" -ForegroundColor Green
} else {
    Write-Host "   ❌ api_server.py not found!" -ForegroundColor Red
    exit 1
}

# Ready to launch
Write-Host "`n" + ("=" * 80)
Write-Host "✅ ALL CHECKS PASSED - READY TO LAUNCH" -ForegroundColor Green
Write-Host ("=" * 80)
Write-Host "`nStarting server..." -ForegroundColor Yellow
Write-Host "This will take 2-3 minutes for first-time model loading.`n" -ForegroundColor Yellow

# Set environment variables
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# Start server
python api_server.py
