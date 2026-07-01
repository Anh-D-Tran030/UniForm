param(
    [switch]$KeepDocker
)

$ErrorActionPreference = "Continue"

$SubmissionRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RunDir = Join-Path $SubmissionRoot ".run"
$ComposeFile = Join-Path $SubmissionRoot "app\\infra\\docker-compose.yml"
$ServicePorts = @(8001, 8005, 8006, 8007, 8008, 8009)

function Stop-ProcessId {
    param(
        [int]$ProcessId,
        [string]$Reason
    )

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return
    }

    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped PID $ProcessId ($Reason)"
}

if (Test-Path $RunDir) {
    Get-ChildItem -Path $RunDir -Filter "*.pid" -File | ForEach-Object {
        $pidText = Get-Content -Path $_.FullName -ErrorAction SilentlyContinue | Select-Object -First 1
        $processId = 0
        if ([int]::TryParse($pidText, [ref]$processId)) {
            Stop-ProcessId -ProcessId $processId -Reason $_.BaseName
        }
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

Get-NetTCPConnection -ErrorAction SilentlyContinue |
    Where-Object { $ServicePorts -contains $_.LocalPort -and $_.State -eq "Listen" } |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object {
        Stop-ProcessId -ProcessId $_ -Reason "service port"
    }

if (-not $KeepDocker) {
    Write-Host "Stopping infrastructure containers..."
    docker compose -f $ComposeFile stop
}

Write-Host "Submission stack stopped."
