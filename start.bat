@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "Semi-Agent" powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0启动配置\run_agent.ps1"