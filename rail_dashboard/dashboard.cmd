@echo off
setlocal

set "NODE_EXE="
for /f "delims=" %%I in ('where node.exe 2^>nul') do if not defined NODE_EXE set "NODE_EXE=%%I"


if not exist "%NODE_EXE%" (
  echo [ERROR] Node.js was not found.
  echo Install 64-bit Node.js 22.13 or later, then run this command again.
  exit /b 1
)

if /i "%~1"=="sync" (
  "%NODE_EXE%" "%~dp0scripts\sync-dashboard-data.mjs" %2 %3 %4 %5 %6 %7 %8 %9
  exit /b %ERRORLEVEL%
)

if /i "%~1"=="start" (
  if not exist "%~dp0node_modules\vinext\dist\cli.js" (
    echo [ERROR] Dashboard dependencies were not found in node_modules.
    echo Install Node.js with npm and run: npm install
    exit /b 1
  )
  "%NODE_EXE%" "%~dp0node_modules\vinext\dist\cli.js" dev
  exit /b %ERRORLEVEL%
)

echo Usage:
echo   dashboard.cmd sync
echo   dashboard.cmd sync --source "D:\copied_outputs\dashboard"
echo   dashboard.cmd start
exit /b 2
