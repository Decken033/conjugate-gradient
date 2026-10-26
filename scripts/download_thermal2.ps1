param(
    [string]$Destination = "data/suitesparse"
)

$ErrorActionPreference = "Stop"
$archive = Join-Path $Destination "thermal2.tar.gz"
$url = "https://www.cise.ufl.edu/research/sparse/MM/Schmid/thermal2.tar.gz"

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Invoke-WebRequest -Uri $url -OutFile $archive

tar -xzf $archive -C $Destination
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract $archive"
}

$matrix = Join-Path $Destination "thermal2/thermal2.mtx"
if (-not (Test-Path $matrix)) {
    throw "Expected Matrix Market file was not extracted: $matrix"
}

Write-Output "Downloaded and extracted: $matrix"
Get-FileHash -Algorithm SHA256 $archive
