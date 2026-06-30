$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $repoRoot "bin"
$templateExe = Join-Path $repoRoot "celune.exe"
$iconIco = Join-Path $repoRoot "resources\celune.ico"
$launcherSource = Join-Path $repoRoot "launcher.c"
$launcherRes = Join-Path $repoRoot "resources\celune.res"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$projectVersion = Select-String -Path (Join-Path $repoRoot "pyproject.toml") -Pattern '^version = "([^"]+)"' | Select-Object -First 1
$existingProcesses = Get-Process -Name @("celune", "celune-bin") -ErrorAction SilentlyContinue

$env:CL = "/O2 /GL /GS /guard:cf /DNDEBUG"
$env:_CL_ = "/link /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT"

if ($null -ne $existingProcesses) {
    Write-Host "Celune is already running, terminating before proceeding with build."
    $existingProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
}

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

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$staleBuildArtifacts = @(
    (Join-Path $outputDir "default_config.yaml"),
    (Join-Path $outputDir "voices"),
    (Join-Path $outputDir "resources"),
    (Join-Path $outputDir "assets")
)
foreach ($stalePath in $staleBuildArtifacts) {
    if (Test-Path $stalePath) {
        Remove-Item -LiteralPath $stalePath -Recurse -Force
    }
}

$arguments = @(
    "run",
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
    }
}

$manifestPath = Join-Path $outputDir "celune-update.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $manifestPath

$buildDir = Join-Path $outputDir "nuitka_main.build"
if (Test-Path $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}

if (Test-Path $launcherObj) {
    Remove-Item -LiteralPath $launcherObj -Force
}
