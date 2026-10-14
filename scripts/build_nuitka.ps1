# SPDX-License-Identifier: Apache-2.0
param(
    [Parameter(Position = 0)]
    [string]$Mode
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Mode)) {
    $Mode = "unsigned"
}
if ($Mode -cne "unsigned" -and $Mode -cne "signed") {
    throw "Build mode must be 'unsigned' or 'signed'."
}

$signingKeyFile = $env:CELUNE_RELEASE_SIGNING_KEY_FILE
$requireSignature = $env:CELUNE_REQUIRE_RELEASE_SIGNATURE -eq "1"
if ($Mode -eq "signed") {
    $requireSignature = $true
}
if ($requireSignature) {
    if (-not $signingKeyFile) {
        if ($env:LOCALAPPDATA) {
            $signingKeyFile = Join-Path $env:LOCALAPPDATA "Celune\keys\release-ed25519-private.pem"
        }
        else {
            $signingKeyFile = Join-Path $HOME ".celune\release-signing-key.pem"
        }
    }
}
if ($signingKeyFile -and -not (Test-Path -LiteralPath $signingKeyFile -PathType Leaf)) {
    throw "The release signing key file was not found: $signingKeyFile"
}
if ($requireSignature -and -not $signingKeyFile) {
    throw "A release signing key is required when signature enforcement is enabled."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $repoRoot "bin"
$buildPython = "3.13"
$archiveName = "Celune-win-x64.zip"
if ($Mode -eq "signed" -or $signingKeyFile) {
    $archiveName = "Celune-win-x64-signed.zip"
}
$archivePath = Join-Path $outputDir $archiveName
$launcherDir = Join-Path $repoRoot "launcher"
$manifestScript = Join-Path $repoRoot "scripts\write_update_manifest.py"
$templateExe = Join-Path $repoRoot "celune.exe"
$iconIco = Join-Path $repoRoot "resources\celune.ico"
$vcruntimeSource = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT\vcruntime140.dll"
$vcruntimeAsset = Join-Path $repoRoot "resources\vcruntime140.dll"
$vcpkgRoot = $env:VCPKG_ROOT
if (-not $vcpkgRoot) {
    $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
}
if (-not $vcpkgRoot) {
    $vcpkgRoot = "C:\vcpkg"
}
$sodiumTripletRoot = Join-Path $vcpkgRoot "installed\x64-windows-static-md"
$sodiumInclude = Join-Path $sodiumTripletRoot "include"
$sodiumLibrary = Join-Path $sodiumTripletRoot "lib\libsodium.lib"
$launcherSources = @(
    (Join-Path $launcherDir "launcher.c"),
    (Join-Path $launcherDir "windows\runtime.c"),
    (Join-Path $launcherDir "windows\terminal.c"),
    (Join-Path $launcherDir "signature.c")
)
$launcherRes = Join-Path $repoRoot "resources\celune.res"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$projectVersion = Select-String -Path (Join-Path $repoRoot "pyproject.toml") -Pattern '^version = "([^"]+)"' | Select-Object -First 1
$copyrightText = [char]0x00A9 + " celunah - Under Apache 2.0 license."
$processTerminationTimeoutSeconds = 30
$artifactRemovalAttempts = 20
$artifactRemovalDelayMilliseconds = 250

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
$revisionOutput = & python (Join-Path $repoRoot "scripts\root.py") --revision
$revisionExitCode = $LASTEXITCODE
if ($revisionExitCode -ne 0) {
    exit $revisionExitCode
}
$revision = ($revisionOutput -join [Environment]::NewLine).Trim()

$env:CL = "/O2 /GL /GS /guard:cf /DNDEBUG"
$env:_CL_ = "/link /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT"

function Wait-CeluneProcessesExit {
    param(
        [Parameter(Mandatory = $true)]
        [int[]]$ProcessIds
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($processTerminationTimeoutSeconds)
    while ($true) {
        $runningProcesses = @(
            $ProcessIds | ForEach-Object {
                Get-Process -Id $_ -ErrorAction SilentlyContinue
            }
        )
        if ($runningProcesses.Count -eq 0) {
            return
        }

        if ([DateTime]::UtcNow -ge $deadline) {
            $runningNames = @(
                $runningProcesses |
                    Select-Object -ExpandProperty ProcessName |
                    Sort-Object -Unique
            ) -join ", "
            throw "Celune processes did not exit within $processTerminationTimeoutSeconds seconds: $runningNames"
        }

        Start-Sleep -Milliseconds 100
    }
}

function Remove-BuildArtifact {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    for ($attempt = 1; $attempt -le $artifactRemovalAttempts; $attempt++) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            return
        }
        catch {
            $isTransientLock =
                $_.Exception -is [System.IO.IOException] -or
                $_.Exception -is [System.UnauthorizedAccessException]
            if (-not $isTransientLock -or $attempt -eq $artifactRemovalAttempts) {
                throw
            }

            Start-Sleep -Milliseconds $artifactRemovalDelayMilliseconds
        }
    }
}

$existingProcesses = @(
    Get-Process -Name @("celune", "celune-bin") -ErrorAction SilentlyContinue
)
if ($existingProcesses.Count -gt 0) {
    Write-Host "Celune is already running, terminating before proceeding with build."
    $existingProcessIds = @($existingProcesses | Select-Object -ExpandProperty Id)
    $existingProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
    Wait-CeluneProcessesExit -ProcessIds $existingProcessIds
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

if (Test-Path -LiteralPath $vcruntimeSource -PathType Leaf) {
    Copy-Item -LiteralPath $vcruntimeSource -Destination $vcruntimeAsset -Force
}
elseif (-not (Test-Path -LiteralPath $vcruntimeAsset -PathType Leaf)) {
    throw "The required Visual C++ runtime DLL was not found at the Visual Studio source or in the repository: $vcruntimeSource"
}

if (-not (Test-Path $manifestScript)) {
    throw "The update manifest script was not found."
}

if (-not (Test-Path (Join-Path $sodiumInclude "sodium.h")) -or
    -not (Test-Path $sodiumLibrary)) {
    throw "Install libsodium with vcpkg using the x64-windows-static-md triplet."
}

& python (Join-Path $repoRoot "scripts\root.py")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update .celune-root."
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$staleBuildArtifacts = @(
    $archivePath,
    (Join-Path $outputDir "default_config.yaml"),
    (Join-Path $outputDir "voices"),
    (Join-Path $outputDir "resources"),
    (Join-Path $outputDir "assets"),
    (Join-Path $outputDir "vcruntime140.dll"),
    (Join-Path $outputDir "celune-bin.cmd")
)
foreach ($stalePath in $staleBuildArtifacts) {
    Remove-BuildArtifact -Path $stalePath
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
    "--include-windows-runtime-dlls=yes",
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

Copy-Item -LiteralPath $vcruntimeAsset -Destination (Join-Path $outputDir "vcruntime140.dll") -Force

foreach ($launcherSource in $launcherSources) {
    if (-not (Test-Path $launcherSource)) {
        throw "Launcher source was not found: $launcherSource"
    }
}

$launcherExe = Join-Path $outputDir "celune.exe"
$launcherObjects = @(
    (Join-Path $outputDir "launcher_main.obj"),
    (Join-Path $outputDir "launcher_windows_runtime.obj"),
    (Join-Path $outputDir "launcher_windows_terminal.obj"),
    (Join-Path $outputDir "launcher_signature.obj")
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
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /MD /DNDEBUG /I`"$launcherDir`" /c /Fo:`"$($launcherObjects[0])`" `"$($launcherSources[0])`"",
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /MD /DNDEBUG /I`"$launcherDir`" /c /Fo:`"$($launcherObjects[1])`" `"$($launcherSources[1])`"",
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /MD /DNDEBUG /I`"$launcherDir`" /c /Fo:`"$($launcherObjects[2])`" `"$($launcherSources[2])`"",
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /MD /DNDEBUG /DSODIUM_STATIC /DSODIUM_EXPORT= /I`"$launcherDir`" /I`"$sodiumInclude`" /c /Fo:`"$($launcherObjects[3])`" `"$($launcherSources[3])`"",
    "cl /nologo /O2 /GL /GS /guard:cf /W4 /DNDEBUG /Fe:`"$launcherExe`" `"$($launcherObjects[0])`" `"$($launcherObjects[1])`" `"$($launcherObjects[2])`" `"$($launcherObjects[3])`" `"$launcherRes`" `"$sodiumLibrary`" bcrypt.lib /link /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT"
)
$compileCmd = "call `"$vsDevCmd`" -arch=amd64 -host_arch=amd64 >nul && " + ($compileCommands -join " && ")
& cmd /c $compileCmd
if ($LASTEXITCODE -ne 0) {
    throw "Failed to compile the Windows launcher."
}

$signatureTestExe = Join-Path $outputDir "launcher_signature_test.exe"
$signatureTestObjects = @(
    (Join-Path $outputDir "launcher_signature_test.obj"),
    (Join-Path $outputDir "launcher_signature_test_impl.obj")
)
$signatureTestSource = Join-Path $repoRoot "tests\launcher_signature_test.c"
$signatureTestCommands = @(
    "cl /nologo /O2 /W4 /MD /DSODIUM_STATIC /DSODIUM_EXPORT= /I`"$launcherDir`" /I`"$sodiumInclude`" /c /Fo:`"$($signatureTestObjects[0])`" `"$signatureTestSource`"",
    "cl /nologo /O2 /W4 /MD /DSODIUM_STATIC /DSODIUM_EXPORT= /I`"$launcherDir`" /I`"$sodiumInclude`" /c /Fo:`"$($signatureTestObjects[1])`" `"$($launcherSources[3])`"",
    "cl /nologo /Fe:`"$signatureTestExe`" `"$($signatureTestObjects[0])`" `"$($signatureTestObjects[1])`" `"$sodiumLibrary`" bcrypt.lib",
    "`"$signatureTestExe`""
)
$signatureTestCmd = "call `"$vsDevCmd`" -arch=amd64 -host_arch=amd64 >nul && " + ($signatureTestCommands -join " && ")
& cmd /c $signatureTestCmd
if ($LASTEXITCODE -ne 0) {
    throw "The native launcher signature tests failed."
}
foreach ($signatureTestArtifact in $signatureTestObjects + @($signatureTestExe)) {
    Remove-BuildArtifact -Path $signatureTestArtifact
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
    "celune-bin.exe",
    "--file",
    "vcruntime140.dll"
)
if ($signingKeyFile) {
    $manifestArguments += @("--signing-key", $signingKeyFile)
}
if ($requireSignature) {
    $manifestArguments += "--require-signature"
}
$previousPythonPath = $env:PYTHONPATH
if ($previousPythonPath) {
    $env:PYTHONPATH = "$repoRoot$([IO.Path]::PathSeparator)$previousPythonPath"
}
else {
    $env:PYTHONPATH = $repoRoot
}
try {
    & uv @manifestArguments
    $manifestExitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}
if ($manifestExitCode -ne 0) {
    throw "Failed to write the update manifest."
}

Push-Location $outputDir
try {
    $archiveFiles = @(
        "celune.exe",
        "celune-bin.exe",
        "vcruntime140.dll",
        "celune-update.json"
    )
    if (Test-Path (Join-Path $outputDir "celune-update.sig")) {
        $archiveFiles += "celune-update.sig"
    }
    Compress-Archive -Path $archiveFiles -DestinationPath $archivePath -Force
}
finally {
    Pop-Location
}

$buildDir = Join-Path $outputDir "nuitka_main.build"
Remove-BuildArtifact -Path $buildDir

foreach ($launcherObj in $launcherObjects) {
    Remove-BuildArtifact -Path $launcherObj
}
