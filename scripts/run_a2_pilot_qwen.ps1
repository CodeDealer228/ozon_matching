[CmdletBinding()]
param(
    [string]$KeyFile = (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.codex-secrets\dashscope_api_key.dpapi'),
    [string]$BaseUrl = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    [string]$PythonExe = 'C:\Users\fitai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$PyArrowPath = 'C:\Users\fitai\AppData\Local\Temp\codex_ozon_parquet\pydeps',
    [int]$PerCategory = 20,
    [int]$Workers = 6,
    [string]$OutputDir,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$generatorPath = Join-Path $PSScriptRoot 'generate_a2_pilot_qwen.py'
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $outputPath = Join-Path $repoRoot 'artifacts\synthetic_v2\a2_pilot_v2'
}
else {
    $outputPath = $OutputDir
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $generatorPath -PathType Leaf)) {
    throw "Generator not found: $generatorPath"
}
if (-not (Test-Path -LiteralPath $PyArrowPath -PathType Container)) {
    throw "PyArrow dependency directory not found: $PyArrowPath"
}

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $PyArrowPath
$env:DASHSCOPE_BASE_URL = $BaseUrl
$keyPointer = [IntPtr]::Zero

try {
    $arguments = @(
        $generatorPath,
        '--per-category', $PerCategory,
        '--workers', $Workers,
        '--output-dir', $outputPath
    )

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
    $generatorExitCode = $LASTEXITCODE
    if ($generatorExitCode -ne 0) {
        throw "A2 generator exited with code $generatorExitCode"
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
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}
