@echo off
cd /d C:\project3
set PYTHONUNBUFFERED=1
".venv_mineru\Scripts\python.exe" -u "_tos_upload.py" --workers 16 >> _tos_upload.log 2>&1
echo TOS_UPLOAD_DONE >> _tos_upload.log
