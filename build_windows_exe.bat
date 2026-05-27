@echo off
echo ========================================================
echo Auto_OPDx - Windows Standalone Executable Compiler
echo ========================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your Windows PATH.
    echo Please install Python 3.11 or higher from python.org and try again.
    pause
    exit /b
)

:: 2. Check if Git is installed (needed for the GitHub dependency)
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not added to your Windows PATH.
    echo Git is required to fetch the OPDx_read dependency.
    echo Please install Git from git-scm.com and try again.
    pause
    exit /b
)

echo [1/4] Creating temporary Python virtual environment...
python -m venv .venv_build
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

echo [2/4] Activating virtual environment and installing dependencies...
call .venv_build\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing pre-compiled binary wheels...
python -m pip install numpy pandas matplotlib opencv-python scipy openpyxl pyqt5 pyinstaller
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install pre-compiled binary packages.
    pause
    exit /b
)

echo Installing repository dependencies...
python -m pip install git+https://github.com/opruvd/OPDx_read.git
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install git dependency.
    pause
    exit /b
)

echo Installing Auto_OPDx...
python -m pip install --no-deps .
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b
)

echo [3/4] Compiling Auto_OPDx to a Windows Executable...
pyinstaller --noconfirm --onefile --windowed --name="Auto_OPDx" entry.py
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller compilation failed.
    pause
    exit /b
)

echo [4/4] Cleaning up temporary build artifacts...
deactivate
rmdir /s /q .venv_build
rmdir /s /q build
del /q Auto_OPDx.spec

echo.
echo ========================================================
echo [SUCCESS] Standalone Windows executable successfully built!
echo Location: dist\Auto_OPDx.exe
echo ========================================================
echo.
pause
