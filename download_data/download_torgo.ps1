param(
    [string]$OutputDir = "dataset\torgo",
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$target = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $target | Out-Null

$baseUrl = "https://www.cs.toronto.edu/~complingweb/data/TORGO"
$files = @(
    "F.tar.bz2",
    "FC.tar.bz2",
    "M.tar.bz2",
    "MC.tar.bz2",
    "doc/ERRORS.xls",
    "doc/CoilLocations.pdf"
)

function Invoke-CurlDownload {
    param(
        [string]$Url,
        [string]$Destination
    )

    $destinationDir = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

    Write-Host "Downloading $Url"
    & curl.exe `
        --location `
        --fail `
        --continue-at - `
        --retry 8 `
        --retry-all-errors `
        --retry-delay 15 `
        --output $Destination `
        $Url

    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE for $Url"
    }
}

foreach ($file in $files) {
    $url = "$baseUrl/$file"
    $localPath = Join-Path $target ($file -replace "/", "\")
    Invoke-CurlDownload -Url $url -Destination $localPath
}

if (-not $SkipExtract) {
    $archives = @("F.tar.bz2", "FC.tar.bz2", "M.tar.bz2", "MC.tar.bz2")
    foreach ($archive in $archives) {
        $archivePath = Join-Path $target $archive
        Write-Host "Extracting $archivePath"
        & tar.exe -xjf $archivePath -C $target

        if ($LASTEXITCODE -ne 0) {
            throw "tar failed with exit code $LASTEXITCODE for $archivePath"
        }
    }
}

Write-Host "TORGO download finished at $target"
