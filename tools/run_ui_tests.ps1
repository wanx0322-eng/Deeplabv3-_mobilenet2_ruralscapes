param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Tests = "tests"
)

$ErrorActionPreference = "Stop"
$env:QT_QPA_PLATFORM = "offscreen"
$baseTemp = Join-Path ([System.IO.Path]::GetTempPath()) (
    "ruralscape-pytest-" + [System.Guid]::NewGuid().ToString("N")
)

& $Python -m pytest $Tests -q -p no:cacheprovider --basetemp $baseTemp
exit $LASTEXITCODE
