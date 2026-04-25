@echo off
setlocal

REM Install dependencies into the project virtual environment.
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
