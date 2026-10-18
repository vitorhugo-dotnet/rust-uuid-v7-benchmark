$ErrorActionPreference = 'Stop'
$version = '12.8.0'
$baseUrl = 'https://developer.download.nvidia.com/compute/cuda/redist'
$manifest = Invoke-RestMethod -Uri "$baseUrl/redistrib_$version.json"
$target = Join-Path $env:RUNNER_TEMP 'cuda-toolkit'
New-Item -ItemType Directory -Force -Path $target | Out-Null

foreach ($package in @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl')) {
    $component = $manifest.$package.'windows-x86_64'
    if (-not $component) { throw "CUDA package not found: $package" }

    $file = Join-Path $env:RUNNER_TEMP "$package.zip"
    $uri = "$baseUrl/$($component.relative_path)"
    Write-Host "Downloading $uri"
    Invoke-WebRequest -Uri $uri -OutFile $file
    $actual = (Get-FileHash -Path $file -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $component.sha256.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $package"
    }

    $extract = Join-Path $env:RUNNER_TEMP "unzip-$package"
    Expand-Archive -Path $file -DestinationPath $extract -Force
    $root = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
    if (-not $root) { throw "Empty CUDA ZIP for $package" }
    Get-ChildItem -LiteralPath $root.FullName | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force
    }
}

$bin = Join-Path $target 'bin'
$nvcc = Join-Path $bin 'nvcc.exe'
if (-not (Test-Path $nvcc)) { throw "nvcc.exe missing from CUDA redistributables" }
"CUDA_PATH=$target" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
$bin | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
& $nvcc --version
