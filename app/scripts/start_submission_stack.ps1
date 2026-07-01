param(
    [switch]$RebuildNext
)

$ErrorActionPreference = "Stop"

$SubmissionRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RunDir = Join-Path $SubmissionRoot ".run"
$LogDir = Join-Path $SubmissionRoot "logs\\services"
$BackendDir = Join-Path $SubmissionRoot "app\\backend"
$GeoDir = Join-Path $SubmissionRoot "app\\geolayoutlm"
$NextDir = Join-Path $SubmissionRoot "app\\frontend\\odc-next-ui"
$ComposeFile = Join-Path $SubmissionRoot "app\\infra\\docker-compose.yml"

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null

function Get-PythonCommand {
    $preferred = "C:\\Users\\thanh\\anaconda3\\python.exe"
    if (Test-Path $preferred) {
        return $preferred
    }
    return "python"
}

function Test-PortListening {
    param([int]$Port)

    $listener = Get-NetTCPConnection -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq $Port -and $_.State -eq "Listen" } |
        Select-Object -First 1

    return $null -ne $listener
}

function Start-TrackedProcess {
    param(
        [string]$Name,
        [int]$Port,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )

    if (Test-PortListening -Port $Port) {
        Write-Host "$Name already listening on port $Port"
        return
    }

    $stdout = Join-Path $LogDir "$Name.log"
    $stderr = Join-Path $LogDir "$Name.err.log"
    $pidFile = Join-Path $RunDir "$Name.pid"

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -Path $pidFile -Value $process.Id
    Write-Host "Started $Name on port $Port (PID $($process.Id))"
}

$python = Get-PythonCommand

Write-Host "Starting infrastructure containers..."
docker compose -f $ComposeFile up -d

Write-Host "Applying Postgres schema..."
& $python (Join-Path $SubmissionRoot "app\\scripts\\init_backend_db.py")

$pythonServices = @(
    @{ Name = "odc_service_8005"; Port = 8005; Script = Join-Path $BackendDir "ODCService.py"; WorkingDirectory = $BackendDir },
    @{ Name = "geolayoutlm_service_8006"; Port = 8006; Script = Join-Path $GeoDir "GeoLayoutLMKVExtractService.py"; WorkingDirectory = $GeoDir },
    @{ Name = "minio_storage_service_8007"; Port = 8007; Script = Join-Path $BackendDir "MinIOStorageService.py"; WorkingDirectory = $BackendDir },
    @{ Name = "auth_service_8008"; Port = 8008; Script = Join-Path $BackendDir "AuthService.py"; WorkingDirectory = $BackendDir },
    @{ Name = "gold_tier_service_8009"; Port = 8009; Script = Join-Path $BackendDir "GoldTierService.py"; WorkingDirectory = $BackendDir }
)

foreach ($service in $pythonServices) {
    if (-not (Test-Path $service.Script)) {
        Write-Warning "$($service.Name) script not found: $($service.Script)"
        continue
    }

    Start-TrackedProcess `
        -Name $service.Name `
        -Port $service.Port `
        -FilePath $python `
        -ArgumentList @($service.Script) `
        -WorkingDirectory $service.WorkingDirectory
}

if (-not (Test-Path (Join-Path $NextDir "node_modules"))) {
    Write-Warning "Next.js dependencies are missing. Run 'npm install' in $NextDir before starting the UI."
}

if ($RebuildNext -or -not (Test-Path (Join-Path $NextDir ".next"))) {
    Write-Host "Building Next.js app..."
    Push-Location $NextDir
    try {
        npm run build
    }
    finally {
        Pop-Location
    }
}

Start-TrackedProcess `
    -Name "next_ui_8001" `
    -Port 8001 `
    -FilePath "cmd.exe" `
    -ArgumentList @("/c", "npm run start -- --hostname 0.0.0.0") `
    -WorkingDirectory $NextDir

Write-Host ""
Write-Host "Submission stack requested."
Write-Host "Next UI:          http://127.0.0.1:8001"
Write-Host "ODC:              http://127.0.0.1:8005"
Write-Host "GeoLayoutLM KVP:  http://127.0.0.1:8006"
Write-Host "Storage:          http://127.0.0.1:8007"
Write-Host "Auth:             http://127.0.0.1:8008"
Write-Host "Gold tier:        http://127.0.0.1:8009"
Write-Host "MinIO console:    http://127.0.0.1:9001"
Write-Host "Dremio console:   http://127.0.0.1:9047"
