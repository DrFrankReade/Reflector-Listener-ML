@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "APP_PYTHON_ARGS="

:: Prefer this application's isolated runtime when it has been installed.
if exist "%~dp0.venv\Scripts\python.exe" (
    set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
    goto run_app
)

:: Try 'python', fall back to 'py -3' if python is not in PATH.
python --version >nul 2>&1
if not errorlevel 1 (
    set "APP_PYTHON=python"
    goto run_app
)

py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "APP_PYTHON=py"
    set "APP_PYTHON_ARGS=-3"
    goto run_app
)

echo [ERROR] Python was not found on your system.
echo Please install 64-bit Python 3.11 or newer from https://www.python.org/
echo and ensure "Add Python to PATH" is checked during installation.
echo.
pause
exit /b 1

:run_app
"%APP_PYTHON%" %APP_PYTHON_ARGS% -B "%~dp0app.py" %*
set "APP_EXIT_CODE=%errorlevel%"
if not "%APP_EXIT_CODE%"=="0" pause
exit /b %APP_EXIT_CODE%
