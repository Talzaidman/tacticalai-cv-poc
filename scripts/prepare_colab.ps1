<#
Build the private Colab upload folder. Run from any working directory.
Original videos are copied without modification; nothing is uploaded.
#>
[CmdletBinding()]
param(
    [string[]]$Videos = @('data/raw/attack_clip.mp4', 'data/raw/attack_clip2.mp4'),
    [ValidatePattern('^[a-zA-Z0-9.-]+\.zip$')]
    [string]$ArchiveName = 'tacticalai-source.zip'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$archivePath = Join-Path $projectRoot $ArchiveName
$uploadRoot = Join-Path $projectRoot 'outputs/colab-upload/TacticalAI'
$codeDirectory = Join-Path $uploadRoot 'code'
$videoDirectory = Join-Path $uploadRoot 'data/raw'
$sourceDirectories = @('src', 'configs', 'notebooks', 'scripts', 'tests', 'docs')
$sourceFiles = @('pyproject.toml', 'requirements.txt', 'README.md', 'AGENTS.md', 'START_HERE.md', '.gitignore')

$packageFiles = @(
    foreach ($directory in $sourceDirectories) {
        Get-ChildItem -LiteralPath (Join-Path $projectRoot $directory) -File -Recurse |
            Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -ne '.pyc' }
    }
    foreach ($file in $sourceFiles) {
        Get-Item -LiteralPath (Join-Path $projectRoot $file)
    }
)

# Build in a fresh temporary archive before replacing a previous generated ZIP.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$temporaryArchive = Join-Path $projectRoot ('tmp/colab-source-' + [guid]::NewGuid().ToString('N') + '.zip')
New-Item -ItemType Directory -Path (Split-Path $temporaryArchive) -Force | Out-Null
$archive = [System.IO.Compression.ZipFile]::Open($temporaryArchive, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in $packageFiles) {
        $entryName = $file.FullName.Substring($projectRoot.Length + 1).Replace('\', '/')
        if ($file.Extension -eq '.ipynb') {
            # Executed notebooks can contain footage and runtime output inline.
            $notebook = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json -AsHashtable
            foreach ($cell in $notebook.cells) {
                if ($cell.cell_type -eq 'code') {
                    $cell.outputs = @()
                    $cell.execution_count = $null
                }
            }
            $notebook.metadata.Remove('widgets') | Out-Null
            $entry = $archive.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Optimal)
            $writer = [System.IO.StreamWriter]::new($entry.Open())
            try {
                $writer.Write(($notebook | ConvertTo-Json -Depth 100 -Compress))
            } finally {
                $writer.Dispose()
            }
        } else {
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $file.FullName, $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    }
} finally {
    $archive.Dispose()
}

$verification = [System.IO.Compression.ZipFile]::OpenRead($temporaryArchive)
try {
    $entries = @($verification.Entries | ForEach-Object FullName)
    foreach ($required in @('pyproject.toml', 'requirements.txt', 'src/tacticalai/runner.py', 'notebooks/colab_baseline.ipynb')) {
        if ($entries -notcontains $required) { throw "Missing archive entry: $required" }
    }
    if ($entries | Where-Object { $_ -match '\.(mp4|pt|pyc)$|(^|/)(\.env|__pycache__)(/|$)' }) {
        throw 'Unexpected video, model, cache or secret in source archive'
    }
} finally {
    $verification.Dispose()
}
Move-Item -LiteralPath $temporaryArchive -Destination $archivePath -Force
New-Item -ItemType Directory -Path $codeDirectory, $videoDirectory -Force | Out-Null
Copy-Item -LiteralPath $archivePath -Destination (Join-Path $codeDirectory $ArchiveName) -Force

$manifest = @(
    foreach ($relativePath in $Videos) {
        $original = Join-Path $projectRoot $relativePath
        if (-not (Test-Path -LiteralPath $original -PathType Leaf)) {
            throw "Missing requested video: $relativePath"
        }
        $name = Split-Path -Leaf $original
        $copy = Join-Path $videoDirectory $name
        Copy-Item -LiteralPath $original -Destination $copy -Force
        $originalHash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
        $copyHash = (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash
        if ($originalHash -ne $copyHash) { throw "Video copy verification failed: $name" }
        [pscustomobject]@{
            file = "data/raw/$name"
            source = $relativePath.Replace('\', '/')
            sha256 = $copyHash
            bytes = (Get-Item -LiteralPath $copy).Length
        }
    }
    [pscustomobject]@{
        file = "code/$ArchiveName"
        sha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
        bytes = (Get-Item -LiteralPath $archivePath).Length
    }
)
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $uploadRoot 'upload-manifest.json') -Encoding UTF8
Write-Output "Verified source archive: $archivePath"
Write-Output "Private upload folder: $uploadRoot"
Write-Output "Source entries: $($packageFiles.Count)"
$manifest | Format-Table -AutoSize
