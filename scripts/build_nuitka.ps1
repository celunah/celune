$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $repoRoot "bin"
$buildPython = "3.13"
$pythonRuntimeSource = Join-Path ((& uv run --quiet --python $buildPython python -c "import sys; print(sys.base_prefix)").Trim()) "python313.dll"
$archivePath = Join-Path $outputDir "Celune-win-x64.zip"
$vcruntimeAsset = Join-Path $repoRoot "resources\vcruntime140.dll"
$templateExe = Join-Path $repoRoot "celune.exe"
$iconIco = Join-Path $repoRoot "resources\celune.ico"
$launcherSource = Join-Path $repoRoot "launcher.c"
$launcherRes = Join-Path $repoRoot "resources\celune.res"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$projectVersion = Select-String -Path (Join-Path $repoRoot "pyproject.toml") -Pattern '^version = "([^"]+)"' | Select-Object -First 1

$env:CL = "/O2 /GL /GS /guard:cf /DNDEBUG"
$env:_CL_ = "/link /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT"

if ($null -eq $projectVersion) {
    throw "Could not determine the project version from pyproject.toml."
}

$version = $projectVersion.Matches[0].Groups[1].Value
$windowsVersion = (($version -split '\+') | Select-Object -First 1)
if ($windowsVersion -notmatch '^\d+(?:\.\d+){0,3}$') {
    throw "The project version '$windowsVersion' is not a valid Windows version string."
}

if (-not (Test-Path (Join-Path $repoRoot "nuitka_main.py"))) {
    throw "nuitka_main.py was not found."
}

if (-not (Test-Path (Join-Path $repoRoot "resources\celune.res"))) {
    throw "resources\celune.res was not found."
}

if (-not (Test-Path $vcruntimeAsset)) {
    throw "resources\vcruntime140.dll was not found."
}

if (-not (Test-Path $pythonRuntimeSource)) {
    throw "The Python 3.13 runtime DLL was not found: $pythonRuntimeSource"
}

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$staleBuildArtifacts = @(
    $archivePath,
    (Join-Path $outputDir "default_config.yaml"),
    (Join-Path $outputDir "voices"),
    (Join-Path $outputDir "resources"),
    (Join-Path $outputDir "assets"),
    (Join-Path $outputDir "vcruntime140.dll"),
    (Join-Path $outputDir "python313.dll")
)
foreach ($stalePath in $staleBuildArtifacts) {
    if (Test-Path $stalePath) {
        Remove-Item -LiteralPath $stalePath -Recurse -Force
    }
}

$arguments = @(
    "run",
    "--python",
    $buildPython,
    "python",
    "-m",
    "nuitka",
    "--deployment",
    "--msvc=latest",
    "--follow-import-to=celune",
    "--include-package-data=celune",
    "--windows-console-mode=force",
    "--product-name=Celune",
    "--file-description=Celune",
    "--product-version=$windowsVersion",
    "--file-version=$windowsVersion",
    "--output-dir=$outputDir",
    "--output-filename=celune-bin.exe",
    "--lto=yes",
    "$repoRoot\nuitka_main.py"
)

if (Test-Path $iconIco) {
    $arguments += "--windows-icon-from-ico=$iconIco"
}
elseif (Test-Path $templateExe) {
    $arguments += "--windows-icon-from-exe=$templateExe"
}

& uv @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed with exit code $LASTEXITCODE."
}

Copy-Item -LiteralPath $vcruntimeAsset -Destination (Join-Path $outputDir "vcruntime140.dll") -Force
Copy-Item -LiteralPath $pythonRuntimeSource -Destination (Join-Path $outputDir "python313.dll") -Force

if (-not (Test-Path $launcherSource)) {
    throw "launcher.c was not found."
}

$launcherExe = Join-Path $outputDir "celune.exe"
$launcherObj = Join-Path $outputDir "launcher.obj"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe was not found."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    throw "Could not locate a Visual Studio installation with C++ build tools."
}

$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat was not found."
}

$env:CL = $null
$env:_CL_ = $null

$compileCmd = "call `"$vsDevCmd`" -arch=amd64 -host_arch=amd64 >nul && cl /nologo /O2 /GL /GS /guard:cf /W4 /DNDEBUG /Fe:`"$launcherExe`" /Fo:`"$launcherObj`" `"$launcherSource`" `"$launcherRes`" /link /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT"
& cmd /c $compileCmd
if ($LASTEXITCODE -ne 0) {
    throw "Failed to compile the Windows launcher."
}

$revision = (& git -C $repoRoot rev-parse HEAD).Trim()
if (-not $revision) {
    throw "Could not determine the Git revision for update metadata."
}

$manifest = [ordered]@{
    version = $windowsVersion
    revision = $revision
    artifact = "Celune-win-x64"
    files = [ordered]@{
        "celune.exe" = (Get-FileHash -Algorithm SHA256 $launcherExe).Hash.ToLowerInvariant()
        "celune-bin.exe" = (Get-FileHash -Algorithm SHA256 (Join-Path $outputDir "celune-bin.exe")).Hash.ToLowerInvariant()
        "vcruntime140.dll" = (Get-FileHash -Algorithm SHA256 (Join-Path $outputDir "vcruntime140.dll")).Hash.ToLowerInvariant()
        "python313.dll" = (Get-FileHash -Algorithm SHA256 (Join-Path $outputDir "python313.dll")).Hash.ToLowerInvariant()
    }
}

$manifestPath = Join-Path $outputDir "celune-update.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $manifestPath

Push-Location $outputDir
try {
    Compress-Archive -Path @(
        "celune.exe",
        "celune-bin.exe",
        "vcruntime140.dll",
        "python313.dll",
        "celune-update.json"
    ) -DestinationPath $archivePath -Force
}
finally {
    Pop-Location
}

$buildDir = Join-Path $outputDir "nuitka_main.build"
if (Test-Path $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}

if (Test-Path $launcherObj) {
    Remove-Item -LiteralPath $launcherObj -Force
}
