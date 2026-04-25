@echo off
setlocal

REM Run the app through the project virtual environment so it does not
REM accidentally use a global Python installation.
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run: py -3.11 -m venv .venv
    exit /b 1
)

".venv\Scripts\python.exe" src\main.py
