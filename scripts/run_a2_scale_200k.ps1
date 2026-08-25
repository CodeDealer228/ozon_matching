[CmdletBinding()]
param(
    [string]$KeyFile = (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.codex-secrets\dashscope_api_key.dpapi'),
    [string]$BaseUrl = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    [string]$PythonExe = 'C:\Users\fitai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$PythonDeps = 'C:\Users\fitai\AppData\Local\Temp\codex_a2_pydeps',
    [int]$Workers = 6,
    [int]$TargetPerCategory = 10000,
    [int]$MaxNewSourcesPerCategory = 0,
    [string]$OutputDir = '',
    [string[]]$ExcludeOutputDir = @(),
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$generator = Join-Path $PSScriptRoot 'generate_a2_scale_200k.py'
foreach ($required in @($PythonExe, $generator)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file not found: $required"
    }
}
if (-not (Test-Path -LiteralPath $PythonDeps -PathType Container)) {
    throw "Dependency directory not found: $PythonDeps"
}
$oldPythonPath = $env:PYTHONPATH
$oldEncoding = $env:PYTHONIOENCODING
$oldBaseUrl = $env:DASHSCOPE_BASE_URL
$hadApiKey = -not [string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)
$keyPointer = [IntPtr]::Zero

try {
    $env:PYTHONPATH = $PythonDeps
    $env:PYTHONIOENCODING = 'utf-8'
    $env:DASHSCOPE_BASE_URL = $BaseUrl
    if (-not $DryRun -and -not $hadApiKey) {
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

    $arguments = @(
        $generator,
        '--workers', $Workers,
        '--target-per-category', $TargetPerCategory
    )
    if ($MaxNewSourcesPerCategory -gt 0) {
        $arguments += @('--max-new-sources-per-category', $MaxNewSourcesPerCategory)
    }
    if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
        $arguments += @('--output-dir', $OutputDir)
    }
    foreach ($excludedDir in $ExcludeOutputDir) {
        if (-not [string]::IsNullOrWhiteSpace($excludedDir)) {
            $arguments += @('--exclude-output-dir', $excludedDir)
        }
    }
    if ($DryRun) {
        $arguments += '--dry-run'
    }
    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "A2 v4 generator exited with code $LASTEXITCODE"
    }
}
finally {
    if (-not $hadApiKey) {
        Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    }
    if ($null -eq $oldPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $oldPythonPath }
    if ($null -eq $oldEncoding) { Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue } else { $env:PYTHONIOENCODING = $oldEncoding }
    if ($null -eq $oldBaseUrl) { Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue } else { $env:DASHSCOPE_BASE_URL = $oldBaseUrl }
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}
