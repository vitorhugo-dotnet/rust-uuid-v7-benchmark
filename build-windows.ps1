$ErrorActionPreference = 'Stop'

Write-Host 'Building Rust CLI...' -ForegroundColor Cyan
cargo build --release
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc -and $env:CUDA_PATH) {
    $candidate = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
    if (Test-Path $candidate) { $nvcc = @{ Source = $candidate } }
}
if (-not $nvcc) {
    Write-Warning 'CUDA Toolkit / nvcc not found. CPU CLI was built. GPU and hybrid require CUDA Toolkit 12.8+.'
    exit 0
}

Write-Host 'Building CUDA GPU executable for RTX 5060 (sm_120)...' -ForegroundColor Cyan
& $nvcc.Source -O3 -std=c++17 -arch=sm_120 cuda/uuid_gpu.cu -o target/release/uuid-gpu.exe
if ($LASTEXITCODE -ne 0) { throw 'nvcc compilation failed. Run from a Visual Studio x64 Developer PowerShell.' }
Write-Host 'Done: target/release/uuid-v7-bench.exe and uuid-gpu.exe' -ForegroundColor Green
