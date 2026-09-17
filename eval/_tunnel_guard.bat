@echo off
:loop
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | find /I "cloudflared.exe" >nul
if errorlevel 1 (
  echo %date% %time% cloudflared dead, restarting >> C:\project3\eval\_tunnel_watchdog.log
  start "" /min "C:\Users\33423\Downloads\cloudflared.exe" tunnel --url http://192.168.88.138:3000 --protocol http2 --no-autoupdate >> C:\project3\eval\_tunnel.log 2>&1
)
timeout /t 15 /nobreak >nul
goto loop
