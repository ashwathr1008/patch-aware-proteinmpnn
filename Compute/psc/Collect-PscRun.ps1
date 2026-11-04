[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9._-]*$')][string]$RunName,
    [Parameter(Mandatory)][ValidatePattern('^[0-9]+$')][string]$JobId,
    [switch]$Wait,
    [string]$Deadline = '2026-11-05T20:00:00-05:00'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$resultRoot = Join-Path $projectRoot 'Results/PSC'
$archiveRoot = Join-Path $resultRoot 'archives'
$remoteRoot = '/ocean/projects/cis260266p/raghuram/patch_aware_project'
$remoteRun = "$remoteRoot/runs/$RunName"
$archiveName = "$RunName-job$JobId.tar.gz"
$remoteArchive = "$remoteRoot/exports/$archiveName"
$localArchive = Join-Path $archiveRoot $archiveName
$destination = Join-Path $resultRoot "$RunName-job$JobId"
$deadlineInstant = [DateTimeOffset]::Parse($Deadline)
$sshOptions = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2')
$terminal = @('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
              'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED')

function Invoke-Psc([string]$Command) {
    $output = & ssh @sshOptions bridges2 $Command
    if ($LASTEXITCODE -ne 0) { throw "PSC SSH failed (exit $LASTEXITCODE)." }
    return $output
}

New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
do {
    if ([DateTimeOffset]::UtcNow -ge $deadlineInstant) {
        throw 'Collection deadline reached without a verified download. Check the job and transfer manually.'
    }
    try {
        $accounting = @(Invoke-Psc "sacct -n -X -j $JobId --format=JobIDRaw,State,ExitCode --parsable2")
    } catch {
        if (-not $Wait) { throw }
        Write-Warning "Status check failed; retrying in 60 seconds. $_"
        Start-Sleep -Seconds 60
        continue
    }
    $record = $accounting | Where-Object { $_ -match "^$JobId\|" } | Select-Object -First 1
    $state = if ($record) { (($record -split '\|')[1] -split '[ +]')[0] } else { 'UNKNOWN' }
    Write-Output ("{0} job {1}: {2}" -f [DateTimeOffset]::Now.ToString('o'), $JobId, $state)
    if ($terminal -contains $state) { break }
    if (-not $Wait) { throw 'Job is not terminal. Retry later or add -Wait.' }
    Start-Sleep -Seconds 60
} while ($true)

$jobDetail = @(Invoke-Psc "sacct -n -X -j $JobId --format=WorkDir%1024 --parsable2")
if (-not ($jobDetail | Where-Object { $_.Trim().TrimEnd('|') -eq $remoteRun })) {
    throw 'The Slurm job working directory does not match the requested run.'
}
$remotePython = '/ocean/projects/cis260266p/raghuram/envs/progen/bin/python'
Invoke-Psc "if test -f $remoteArchive; then $remotePython $remoteRun/code/bundle.py verify --archive $remoteArchive; else $remotePython $remoteRun/code/bundle.py pack --source $remoteRun --output $remoteArchive; fi"

foreach ($suffix in @('', '.sha256')) {
    $target = "$localArchive$suffix"
    & scp @sshOptions "b2-data:$remoteArchive$suffix" "$target.partial"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $archiveName$suffix" }
    if (Test-Path -LiteralPath $target) {
        $existing = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        $downloaded = (Get-FileHash -LiteralPath "$target.partial" -Algorithm SHA256).Hash
        if ($existing -ne $downloaded) { throw 'Existing local archive differs; refusing to overwrite.' }
        Remove-Item -LiteralPath "$target.partial"
    } else {
        Move-Item -LiteralPath "$target.partial" -Destination $target
    }
}
& python (Join-Path $PSScriptRoot 'bundle.py') verify --archive $localArchive
if ($LASTEXITCODE -ne 0) { throw 'Archive integrity verification failed.' }
& python (Join-Path $PSScriptRoot 'bundle.py') extract --archive $localArchive --destination $destination
if ($LASTEXITCODE -ne 0) { throw 'Verified extraction failed.' }
$receipt = [ordered]@{
    job_id = $JobId; state = $state; scheduler_record = $record
    remote_source = $remoteRun; local_directory = $destination
    archive = $localArchive; archive_sha256 = (Get-FileHash -LiteralPath $localArchive).Hash
    verified_at = [DateTimeOffset]::Now.ToString('o')
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resultRoot "$RunName-job$JobId.receipt.json") -Encoding utf8
Write-Output "VERIFIED LOCAL COPY: $destination"
if ($state -ne 'COMPLETED') { Write-Warning "Files were recovered, but Slurm state is $state; these are not successful experimental results." }
