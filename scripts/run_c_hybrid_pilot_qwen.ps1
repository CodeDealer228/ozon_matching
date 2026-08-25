[CmdletBinding()]
param(
    [string]$KeyFile = (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.codex-secrets\dashscope_api_key.dpapi'),
    [string]$BaseUrl = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    [string]$PythonExe = 'C:\Users\fitai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$PyArrowPath = 'C:\Users\fitai\AppData\Local\Temp\codex_pyarrow_verify_24',
    [int]$Workers = 6,
    [string]$OutputDir,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$generatorPath = Join-Path $PSScriptRoot 'generate_c_hybrid_pilot_qwen.py'
$outputPath = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $repoRoot 'artifacts\synthetic_v2\c_hybrid_pilot_v1'
}
else {
    $OutputDir
}

foreach ($requiredFile in @($PythonExe, $generatorPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}
if (-not (Test-Path -LiteralPath $PyArrowPath -PathType Container)) {
    throw "PyArrow dependency directory not found: $PyArrowPath"
}

$previousPythonPath = $env:PYTHONPATH
$previousPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONPATH = $PyArrowPath
$env:PYTHONIOENCODING = 'utf-8'
$env:DASHSCOPE_BASE_URL = $BaseUrl
$keyPointer = [IntPtr]::Zero

try {
    $arguments = @($generatorPath, '--workers', $Workers, '--output-dir', $outputPath)
    if ($DryRun) {
        $arguments += '--dry-run'
    }
    else {
        if (-not (Test-Path -LiteralPath $KeyFile -PathType Leaf)) {
            throw "Encrypted DashScope key file not found: $KeyFile"
        }
        $encryptedKey = (Get-Content -LiteralPath $KeyFile -Raw).Trim()
        if (-not $encryptedKey) {
            throw "Encrypted DashScope key file is empty: $KeyFile"
        }
        $secureKey = ConvertTo-SecureString $encryptedKey
        $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        $env:DASHSCOPE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    }

    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Method C generator exited with code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }
    if ($null -eq $previousPythonIoEncoding) {
        Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONIOENCODING = $previousPythonIoEncoding
    }
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}
