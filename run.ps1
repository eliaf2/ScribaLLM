#!/usr/bin/env pwsh

# Compilation script for ScribaLLM repository
# Usage: .\run.ps1 [OPTIONS]
# Options:
#   -Dev                Run in development mode
#   -Detached           Run containers in detached mode
#   -Help               Show this help message
#   -NoCache            Build without using cache
#   -Verbose            Enable verbose output
#   -Rebuild            Force rebuild images
#   -Init               Reinitialize containers (fresh start)

[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$Detached,
    [switch]$Help,
    [switch]$NoCache,
    [switch]$Verbose,
    [switch]$Rebuild,
    [switch]$Init
)

# Script configuration
$SCRIPT_NAME = Split-Path -Leaf $MyInvocation.MyCommand.Name
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_NAME = "scriballm"
$COMPOSE_FILE = "docker-compose.yaml"
$COMPOSE_DEV_FILE = "docker-compose.dev.yaml"

# Colors for console output
$Colors = @{
    Red = "Red"
    Green = "Green" 
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Set error action preference
$ErrorActionPreference = "Stop"

# Print helpers
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

function Write-Debug {
    param([string]$Message)
    if ($Verbose) {
        Write-Host "[DEBUG] $Message" -ForegroundColor $Colors.Blue
    }
}

# Help function
function Show-Help {
    Write-Host @"
$SCRIPT_NAME - Production compilation script for $PROJECT_NAME

USAGE:
    .\$SCRIPT_NAME [OPTIONS]

OPTIONS:
    -Dev                Run in development mode using $COMPOSE_DEV_FILE
    -Detached           Run containers in detached mode
    -Help               Show this help message and exit
    -NoCache            Build without using Docker cache
    -Verbose            Enable verbose output and logging
    -Rebuild            Force rebuild of Docker images
    -Init               Reinitialize containers from images (remove and recreate)

EXAMPLES:
    .\$SCRIPT_NAME                  # Normal production start
    .\$SCRIPT_NAME -Dev             # Development start
    .\$SCRIPT_NAME -Rebuild         # Rebuild images before starting
    .\$SCRIPT_NAME -Init            # Recreate containers fresh
"@
}

# Show help if requested
if ($Help) {
    Show-Help
    exit 0
}

# Set rebuild logic
if ($Rebuild) {
    $Init = $true
}

# Dependency checks
function Test-Dependencies {
    $missing = @()
    
    try {
        $null = Get-Command docker -ErrorAction Stop
    }
    catch {
        $missing += "docker"
    }
    
    $hasDockerCompose = $false
    try {
        $null = Get-Command docker-compose -ErrorAction Stop
        $hasDockerCompose = $true
    }
    catch {
        try {
            $null = Invoke-Expression "docker compose version" -ErrorAction Stop
            $hasDockerCompose = $true
        }
        catch {
            # Neither docker-compose nor docker compose available
        }
    }
    
    if (-not $hasDockerCompose) {
        $missing += "docker-compose"
    }
    
    if ($missing.Count -gt 0) {
        Write-Error "Missing dependencies: $($missing -join ', ')"
        exit 1
    }
    
    Write-Debug "All dependencies are satisfied."
}

function Test-DockerDaemon {
    try {
        $null = Invoke-Expression "docker info" -ErrorAction Stop
        Write-Debug "Docker daemon is running."
    }
    catch {
        Write-Error "Docker daemon is not running."
        exit 1
    }
}

function Test-ComposeFiles {
    $file = $COMPOSE_FILE
    if ($Dev) {
        $file = $COMPOSE_DEV_FILE
    }
    
    if (-not (Test-Path $file)) {
        Write-Error "Missing compose file: $file"
        exit 1
    }
    
    Write-Debug "Validated compose file: $file"
}

function Get-ComposeCommand {
    try {
        $null = Get-Command docker-compose -ErrorAction Stop
        return "docker-compose"
    }
    catch {
        return "docker compose"
    }
}

function Test-ContainersRunning {
    param([string]$ComposeFile)
    
    $cmd = Get-ComposeCommand
    $baseArgs = @("-f", $ComposeFile, "-p", $PROJECT_NAME)
    
    try {
        $runningOutput = Invoke-Expression "$cmd $($baseArgs -join ' ') ps --services --filter `"status=running`""
        $running = ($runningOutput | Where-Object { $_.Trim() -ne "" }).Count
        
        $definedOutput = Invoke-Expression "$cmd $($baseArgs -join ' ') config --services"
        $defined = ($definedOutput | Where-Object { $_.Trim() -ne "" }).Count
        
        return ($running -eq $defined -and $defined -gt 0)
    }
    catch {
        return $false
    }
}

# Main compilation function
function Start-RepositoryCompilation {
    $file = $COMPOSE_FILE
    if ($Dev) {
        $file = $COMPOSE_DEV_FILE
    }
    
    $cmd = Get-ComposeCommand
    $baseArgs = @("-f", $file, "-p", $PROJECT_NAME)
    
    Write-Info "Using compose file: $file"
    
    # If -Init: tear down existing containers
    if ($Init) {
        Write-Info "Reinitializing containers..."
        try {
            Invoke-Expression "$cmd $($baseArgs -join ' ') down -v"
        }
        catch {
            # Ignore errors on down command
        }
    }
    
    # If -Rebuild: rebuild images explicitly
    if ($Rebuild) {
        Write-Info "Rebuilding images..."
        $buildArgs = $baseArgs + @("build")
        if ($NoCache) {
            $buildArgs += "--no-cache"
        }
        Invoke-Expression "$cmd $($buildArgs -join ' ')"
    }
    
    # If already running and not init/rebuild → skip
    if (-not $Init -and -not $Rebuild -and (Test-ContainersRunning $file)) {
        Write-Info "Containers already running. Skipping startup."
        return
    }
    
    # Start containers
    $upArgs = $baseArgs + @("up")
    if ($Detached) {
        $upArgs += "--detach"
    }
    if ($Rebuild) {
        $upArgs += "--build"
    }
    if ($NoCache) {
        $upArgs += "--no-cache"
    }
    
    Write-Info "Starting containers..."
    Invoke-Expression "$cmd $($upArgs -join ' ')"
}

# Main execution
function Main {
    try {
        Test-Dependencies
        Test-DockerDaemon
        Test-ComposeFiles
        Start-RepositoryCompilation
        Write-Info "Done!"
    }
    catch {
        Write-Error "Script execution failed: $($_.Exception.Message)"
        exit 1
    }
}

# Execute main function
Main