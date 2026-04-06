$py = "E:\project\qwen3tts_env\Scripts\python.exe"

# Basic info
& $py -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check flash_attn
& $py -c "import flash_attn; print('flash_attn:', flash_attn.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "flash_attn: NOT installed" }

# Check other acceleration packages
& $py -c "import bitsandbytes; print('bitsandbytes: OK')" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "bitsandbytes: NOT installed" }

& $py -c "import optimum; print('optimum: OK')" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "optimum: NOT installed" }

# Check installed packages related to acceleration
& $py -m pip list 2>$null | Select-String -Pattern "flash|triton|xformers|deepspeed|accelerate|bitsandbytes|optimum|vllm|quanto|gguf"

Write-Host ""
Write-Host "=== Transformers version ==="
& $py -c "import transformers; print(transformers.__version__)"
