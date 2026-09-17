@echo off
chcp 65001 >nul
set TEMP=C:\m
set TMP=C:\m
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
C:\project3\.venv_mineru\Scripts\python.exe -u C:\project3\RAG\pdf\clean.py
pause