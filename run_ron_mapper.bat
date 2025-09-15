@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === Change to the folder this script lives in ===
cd /d "%~dp0"

REM === Find Python (prefer py launcher) ===
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set PYTHON=py
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    set PYTHON=python
  ) else (
    echo ERROR: Python not found on PATH. Please install Python 3.8+ from https://www.python.org and try again.
    pause
    exit /b 1
  )
)

REM === Create a virtual environment if it doesn't exist ===
if not exist ".venv" (
  echo Creating virtual environment...
  %PYTHON% -m venv .venv
  if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
  )
)

REM === Activate the venv ===
call .venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Failed to activate virtual environment.
  pause
  exit /b 1
)

REM === Upgrade pip (quiet) ===
python -m pip install --upgrade pip --quiet

REM === Install dependencies ===
if exist "requirements.txt" (
  echo Installing requirements from requirements.txt ...
  pip install -r requirements.txt
) else (
  echo requirements.txt not found; installing core packages...
  pip install streamlit pandas numpy matplotlib plotly openpyxl XlsxWriter
)

REM === Run the app ===
echo Launching Streamlit app...
streamlit run ron_streamlit_v1_0.py

REM === Keep window open after Streamlit exits ===
echo.
echo Streamlit process exited. Press any key to close this window.
pause >nul
