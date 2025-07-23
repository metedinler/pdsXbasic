$ErrorActionPreference = 'Stop'
$pythonDir = 'C:\\Users\\mete\\AppData\\Local\\Programs\\Python\\Python310'
$venvDir = '.pdsx_isolated_env\Scripts'
$oldPath = [System.Environment]::GetEnvironmentVariable('Path', [System.EnvironmentVariableTarget]::User)
if ($oldPath -notlike "*$pythonDir*") {
    $newPath = "$oldPath;$pythonDir;$venvDir"
    [System.Environment]::SetEnvironmentVariable('Path', $newPath, [System.EnvironmentVariableTarget]::User)
    Write-Host 'PATH güncellendi.'
} else {
    Write-Host 'PATH zaten güncel.'
}
