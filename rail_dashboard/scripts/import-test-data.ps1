param(
    [string]$DataDirectory = (Join-Path $PSScriptRoot "..\public\data"),
    [string]$ProjectId = "your-firebase-project",
    [string]$Bucket = "your-firebase-project.firebasestorage.app",
    [switch]$SkipStorageUpload
)

$ErrorActionPreference = "Stop"

function ConvertTo-FirestoreValue {
    param([AllowNull()][object]$Value)

    if ($null -eq $Value) {
        return @{ nullValue = $null }
    }
    if ($Value -is [bool]) {
        return @{ booleanValue = $Value }
    }
    if ($Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]) {
        return @{ integerValue = [string]$Value }
    }
    if ($Value -is [single] -or $Value -is [double] -or $Value -is [decimal]) {
        return @{ doubleValue = [double]$Value }
    }
    if ($Value -is [DateTime]) {
        return @{ stringValue = $Value.ToUniversalTime().ToString("o") }
    }
    if ($Value -is [DateTimeOffset]) {
        return @{ stringValue = $Value.UtcDateTime.ToString("o") }
    }
    if ($Value -is [string]) {
        return @{ stringValue = $Value }
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $fields = [ordered]@{}
        foreach ($key in $Value.Keys) {
            $fields[[string]$key] = ConvertTo-FirestoreValue $Value[$key]
        }
        return @{ mapValue = @{ fields = $fields } }
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $values = @()
        foreach ($item in $Value) {
            $values += ,(ConvertTo-FirestoreValue $item)
        }
        return @{ arrayValue = @{ values = $values } }
    }

    throw "지원하지 않는 JSON 값 형식입니다: $($Value.GetType().FullName)"
}

function Test-SafeStoragePath {
    param([string]$Path)
    return $Path.StartsWith("dashboard/media/") -and
        -not $Path.Contains("..") -and
        -not $Path.StartsWith("/")
}

$resolvedDataDirectory = (Resolve-Path -LiteralPath $DataDirectory).Path
$jsonPath = Join-Path $resolvedDataDirectory "inspection-export.json"
if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    throw "inspection-export.json 파일을 찾지 못했습니다: $jsonPath"
}

$credentialPath = Join-Path $env:USERPROFILE ".config\configstore\firebase-tools.json"
if (-not (Test-Path -LiteralPath $credentialPath -PathType Leaf)) {
    throw "Firebase CLI 로그인이 없습니다. firebase login --reauth를 먼저 실행하세요."
}

$credential = Get-Content -LiteralPath $credentialPath -Raw | ConvertFrom-Json
$expiresAt = [int64]$credential.tokens.expires_at
$now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
if (-not $credential.tokens.access_token -or $expiresAt -le ($now + 60000)) {
    throw "Firebase CLI 로그인이 만료됐습니다. firebase login --reauth를 실행한 뒤 다시 시도하세요."
}

$source = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json -AsHashtable -Depth 100
$requiredArrays = @("runs", "model_provenance", "captures", "analyses", "artifacts", "run_summaries")
if ($source.schema_version -ne 1) {
    throw "지원하지 않는 schema_version입니다."
}
foreach ($name in $requiredArrays) {
    if ($null -eq $source.$name) {
        throw "필수 배열이 없습니다: $name"
    }
}

$filesByName = @{}
Get-ChildItem -LiteralPath $resolvedDataDirectory -File -Recurse | ForEach-Object {
    $filesByName[$_.Name] = $_.FullName
}

$previewArtifacts = @($source.artifacts | Where-Object {
    ($_.artifact_type -eq "rust_preview" -or $_.artifact_type -eq "crack_preview") -and
    $_.public_url -and
    (Test-SafeStoragePath $_.object_key)
})

$missing = @($previewArtifacts | Where-Object {
    -not $filesByName.ContainsKey([System.IO.Path]::GetFileName($_.object_key))
})
if ($missing.Count -gt 0) {
    throw "마스크 이미지 $($missing.Count)개를 찾지 못했습니다."
}

$headers = @{ Authorization = "Bearer $($credential.tokens.access_token)" }
if (-not $SkipStorageUpload) {
    for ($index = 0; $index -lt $previewArtifacts.Count; $index++) {
        $artifact = $previewArtifacts[$index]
        $fileName = [System.IO.Path]::GetFileName($artifact.object_key)
        $filePath = $filesByName[$fileName]
        $encodedObjectName = [System.Uri]::EscapeDataString([string]$artifact.object_key)
        $uploadUri = "https://storage.googleapis.com/upload/storage/v1/b/$Bucket/o?uploadType=media&name=$encodedObjectName"
        $contentType = if ($artifact.media_type) { [string]$artifact.media_type } else { "image/png" }

        Write-Progress -Activity "Firebase Storage 업로드" -Status "$($index + 1)/$($previewArtifacts.Count) $fileName" -PercentComplete ((($index + 1) / $previewArtifacts.Count) * 100)
        Invoke-RestMethod -Method Post -Uri $uploadUri -Headers $headers -ContentType $contentType -InFile $filePath | Out-Null
    }
    Write-Progress -Activity "Firebase Storage 업로드" -Completed
}

foreach ($run in @($source.runs)) {
    $runId = [string]$run.run_id
    $captures = @($source.captures | Where-Object { $_.run_id -eq $runId })
    $captureIds = @{}
    foreach ($capture in $captures) {
        $captureIds[[string]$capture.capture_id] = $true
    }

    $artifacts = @($source.artifacts | Where-Object {
        $captureIds.ContainsKey([string]$_.capture_id)
    } | ForEach-Object {
        $copy = $_ | ConvertTo-Json -Depth 100 | ConvertFrom-Json -AsHashtable -Depth 100
        $copy["public_url"] = $null
        $copy
    })

    $payload = [ordered]@{
        schema_version = [int]$source.schema_version
        exported_at_utc = [string]$source.exported_at_utc
        runs = @($source.runs | Where-Object { $_.run_id -eq $runId })
        model_provenance = @($source.model_provenance | Where-Object { $_.run_id -eq $runId })
        captures = $captures
        analyses = @($source.analyses | Where-Object { $captureIds.ContainsKey([string]$_.capture_id) })
        artifacts = $artifacts
        run_summaries = @($source.run_summaries | Where-Object { $_.run_id -eq $runId })
    }

    $document = @{
        fields = @{
            payload = ConvertTo-FirestoreValue $payload
            source = @{ stringValue = "legacy-test-data" }
            imported_at = @{ timestampValue = [DateTime]::UtcNow.ToString("o") }
        }
    }
    $encodedRunId = [System.Uri]::EscapeDataString($runId)
    $documentUri = "https://firestore.googleapis.com/v1/projects/$ProjectId/databases/(default)/documents/inspection_exports/$encodedRunId"
    $body = $document | ConvertTo-Json -Depth 100 -Compress
    Invoke-RestMethod -Method Patch -Uri $documentUri -Headers $headers -ContentType "application/json; charset=utf-8" -Body $body | Out-Null
}

[pscustomobject]@{
    project = $ProjectId
    runs = @($source.runs).Count
    captures = @($source.captures).Count
    analyses = @($source.analyses).Count
    artifacts = @($source.artifacts).Count
    summaries = @($source.run_summaries).Count
    preview_images = $previewArtifacts.Count
} | ConvertTo-Json -Compress
