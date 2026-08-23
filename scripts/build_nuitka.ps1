# SPDX-License-Identifier: Apache-2.0
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $repoRoot "bin"
$archivePath = Join-Path $outputDir "Celune-win-x64.zip"
$launcherDir = Join-Path $repoRoot "launcher"
$manifestScript = Join-Path $repoRoot "scripts\write_update_manifest.py"
$templateExe = Join-Path $repoRoot "celune.exe"
$iconIco = Join-Path $repoRoot "resources\celune.ico"
$launcherSources = @(
    (Join-Path $launcherDir "launcher.c"),
    (Join-Path $launcherDir "windows\runtime.c"),
    (Join-Path $launcherDir "windows\terminal.c")
)
$launcherRes = Join-Path $repoRoot "resources\celune.res"
$launcherCompatibilityScript = Join-Path $repoRoot "scripts\celune-bin.cmd"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$projectVersion = Select-String -Path (Join-Path $repoRoot "pyproject.toml") -Pattern '^version = "([^"]+)"' | Select-Object -First 1
$copyrightText = [char]0x00A9 + " celunah - Under MIT license."
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

if (-not (Test-Path $manifestScript)) {
    throw "The update manifest script was not found."
}

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

& uv run python (Join-Path $repoRoot "scripts\root.py")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update .celune-root."
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$staleBuildArtifacts = @(
    $archivePath,
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
    "--copyright=$copyrightText",
    "--company-name=https://github.com/celunah",
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

foreach ($launcherSource in $launcherSources) {
    if (-not (Test-Path $launcherSource)) {
        throw "Launcher source was not found: $launcherSource"
    }
}

if (-not (Test-Path $launcherCompatibilityScript)) {
    throw "The launcher compatibility script was not found."
}

$launcherExe = Join-Path $outputDir "celune.exe"
$launcherObjects = @(
    (Join-Path $outputDir "launcher_main.obj"),
    (Join-Path $outputDir "launcher_windows_runtime.obj"),
    (Join-Path $outputDir "launcher_windows_terminal.obj")
)
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

$compileCommands = @(
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /DNDEBUG /I`"$launcherDir`" /c /Fo:`"$($launcherObjects[0])`" `"$($launcherSources[0])`"",
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /DNDEBUG /I`"$launcherDir`" /c /Fo:`"$($launcherObjects[1])`" `"$($launcherSources[1])`"",
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /DNDEBUG /I`"$launcherDir`" /c /Fo:`"$($launcherObjects[2])`" `"$($launcherSources[2])`"",
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /DNDEBUG /Fe:`"$launcherExe`" `"$($launcherObjects[0])`" `"$($launcherObjects[1])`" `"$($launcherObjects[2])`" `"$launcherRes`" /link /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT"
)
$compileCmd = "call `"$vsDevCmd`" -arch=amd64 -host_arch=amd64 >nul && " + ($compileCommands -join " && ")
& cmd /c $compileCmd
if ($LASTEXITCODE -ne 0) {
    throw "Failed to compile the Windows launcher."
}

Copy-Item -LiteralPath $launcherCompatibilityScript -Destination (Join-Path $outputDir "celune-bin.cmd") -Force

$revision = (& git -C $repoRoot rev-parse HEAD).Trim()
if (-not $revision) {
    throw "Could not determine the Git revision for update metadata."
}

$manifestArguments = @(
    "run",
    "python",
    $manifestScript,
    "--output-dir",
    $outputDir,
    "--version",
    $windowsVersion,
    "--revision",
    $revision,
    "--artifact",
    "Celune-win-x64",
    "--file",
    "celune.exe",
    "--file",
    "celune-bin.exe"
)
& uv @manifestArguments
if ($LASTEXITCODE -ne 0) {
    throw "Failed to write the update manifest."
}

Push-Location $outputDir
try {
    Compress-Archive -Path @(
        "celune.exe",
        "celune-bin.cmd",
        "celune-bin.exe",
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

foreach ($launcherObj in $launcherObjects) {
    if (Test-Path $launcherObj) {
        Remove-Item -LiteralPath $launcherObj -Force
    }
}
