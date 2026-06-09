$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $repoRoot "build\nuitka"
$templateExe = Join-Path $repoRoot "celune.exe"
$projectVersion = Select-String -Path (Join-Path $repoRoot "pyproject.toml") -Pattern '^version = "([^"]+)"' | Select-Object -First 1

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

$arguments = @(
    "run",
    "python",
    "-m",
    "nuitka",
    "--deployment",
    "--msvc=latest",
    "--follow-import-to=celune",
    "--include-package-data=celune",
    "--include-data-files=$repoRoot\default_config.yaml=default_config.yaml",
    "--include-data-dir=$repoRoot\voices=voices",
    "--include-data-dir=$repoRoot\resources=resources",
    "--windows-console-mode=force",
    "--product-name=Celune",
    "--file-description=Celune",
    "--product-version=$windowsVersion",
    "--file-version=$windowsVersion",
    "--output-dir=$outputDir",
    "--output-filename=celune.exe",
    "$repoRoot\nuitka_main.py"
)

if (Test-Path $templateExe) {
    $arguments += "--windows-icon-from-exe=$templateExe"
}

& uv @arguments

New-Item -ItemType Directory -Force -Path (Join-Path $outputDir "celune") | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot "default_config.yaml") -Destination (Join-Path $outputDir "default_config.yaml") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "voices") -Destination (Join-Path $outputDir "voices") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "resources") -Destination (Join-Path $outputDir "resources") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "celune\assets") -Destination (Join-Path $outputDir "celune\assets") -Recurse -Force
