@echo off
title DEFECTX AI - Mobile Camera Server & ADB Reverse
color 0b

echo ============================================================================
echo   DEFECTX AI - MOBILE CAMERA INSPECTION RUNNER (CLOCKIT 2026)
echo ============================================================================
echo.

set ADB_PATH=C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe
set PHP_PATH=C:\xampp\php\php.exe

if not exist "%PHP_PATH%" (
    set PHP_PATH=php
)

echo [1/3] Checking USB ADB Connection to Android Phone...
if exist "%ADB_PATH%" (
    "%ADB_PATH%" devices
    echo.
    echo [2/3] Setting up USB port forwarding: tcp:8080 -^> tcp:8080 ...
    "%ADB_PATH%" reverse tcp:8080 tcp:8080
    if %errorlevel% neq 0 (
        echo [WARNING] USB device not detected or unauthorized.
        echo Make sure your phone is connected with USB Debugging enabled.
    ) else (
        echo [SUCCESS] USB reverse tunnel established!
    )
) else (
    echo [NOTE] ADB not found in default Android SDK path. If using USB, enable port reverse manually.
)

echo.
echo ============================================================================
echo   SERVER STARTING ON PORT 8080
echo ============================================================================
echo.
echo   * On your Android phone (via USB ADB), open Chrome and go to:
echo.
echo         http://localhost:8080/frontend/inspection.html
echo.
echo   * Or on this laptop, open:
echo.
echo         http://localhost:8080/frontend/inspection.html
echo.
echo   * For wireless HTTPS on Android, launch a secure tunnel:
echo         npx localtunnel --port 8080
echo.
echo   Press Ctrl+C to stop the server when finished.
echo ============================================================================
echo.

"%PHP_PATH%" -S 0.0.0.0:8080
pause
