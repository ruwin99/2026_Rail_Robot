@echo off
setlocal
cd /d "%~dp0"
if /i "%~1"=="build" (call npm run firebase:build & exit /b)
if /i "%~1"=="serve" (call npm run firebase:preview & exit /b)
if /i "%~1"=="dev" (call npm run firebase:dev & exit /b)
echo Usage: firebase-dashboard.cmd build ^| serve ^| dev
exit /b 2
