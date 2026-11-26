# DEFECTX AI - Mobile Camera & Laptop Production Inspection Launcher (PowerShell)
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  DEFECTX AI - PRODUCTION INSPECTION SERVER (LAPTOP & MOBILE)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$php = "C:\xampp\php\php.exe"

if (-not (Test-Path $php)) {
    $php = "php"
}

# 1. Check for Android USB ADB connection
if (Test-Path $adb) {
    Write-Host "[1/3] Checking connected Android devices via ADB..." -ForegroundColor Yellow
    & $adb devices
    Write-Host "[2/3] Configuring ADB reverse tunnel (tcp:8080 -> tcp:8080)..." -ForegroundColor Yellow
    & $adb reverse tcp:8080 tcp:8080
    Write-Host "[SUCCESS] USB reverse tunnel established! http://localhost:8080 is treated as a Secure Context by Android Chrome." -ForegroundColor Green
} else {
    Write-Host "[NOTE] ADB not found in default Android SDK path. (For wireless phone access, use secure tunnel/ngrok or USB port reverse)." -ForegroundColor Yellow
}

# Find local IPv4 address
$ip = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi*', 'Ethernet*' -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike '169.254*' -and $_.IPAddress -notlike '127.*' } | Select-Object -First 1).IPAddress
if (-not $ip) { $ip = "127.0.0.1" }

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  INSPECTION SERVER LISTENING ON PORT 8080" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  [Option 1: Laptop Browser (Edge / Chrome)]" -ForegroundColor White
Write-Host "     http://localhost:8080/frontend/inspection.html" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [Option 2: Android Phone via USB (Reverse Port Forwarded - Highly Recommended)]" -ForegroundColor White
Write-Host "     1. Connect phone with USB Debugging enabled." -ForegroundColor Gray
Write-Host "     2. Open Google Chrome on your phone and browse to:" -ForegroundColor Gray
Write-Host "        http://localhost:8080/frontend/inspection.html" -ForegroundColor Cyan
Write-Host "        (Android Chrome treats localhost as a Secure Context, enabling the rear camera!)" -ForegroundColor Gray
Write-Host ""
Write-Host "  [Option 3: Android Phone via Wireless LAN / Secure Tunnel]" -ForegroundColor White
Write-Host "     Run: npx localtunnel --port 8080" -ForegroundColor Yellow
Write-Host "     Or:  ngrok http 8080" -ForegroundColor Yellow
Write-Host "     (Gives a valid https:// URL so Chrome allows rear camera wirelessly)" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop server." -ForegroundColor Gray
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""

& $php -S 0.0.0.0:8080
