@echo off
setlocal
title AI Face Door Lock

echo.
echo ========================================
echo       AI FACE DOOR LOCK
echo ========================================
echo.

REM ----------------------------------------
REM Check Python
REM ----------------------------------------
echo [1/5] Checking Python...

py -3.12 --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo Python 3.12 was not found.
    echo.
    echo This project requires Python 3.12.
    echo Please install Python 3.12 from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Python 3.12 found.
echo.

REM ----------------------------------------
REM Create virtual environment
REM ----------------------------------------
echo [2/5] Setting up virtual environment...

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Virtual environment ready.
echo.

REM ----------------------------------------
REM Install dependencies
REM ----------------------------------------
echo [3/5] Checking Python packages...

".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check >nul

".venv\Scripts\python.exe" -m pip install -r requirements.txt --disable-pip-version-check

if errorlevel 1 (
    echo.
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Dependencies ready.
echo.

REM ----------------------------------------
REM Create required folders
REM ----------------------------------------
echo [4/5] Checking project folders...

if not exist "known_faces" mkdir known_faces
if not exist "logs" mkdir logs

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo.
    echo ERROR: YuNet model is missing.
    echo.
    pause
    exit /b 1
)

if not exist "models\face_recognition_sface_2021dec.onnx" (
    echo.
    echo ERROR: SFace model is missing.
    echo.
    pause
    exit /b 1
)

if not exist "models\face_landmarker.task" (
    echo.
    echo ERROR: Face Landmarker model is missing.
    echo.
    pause
    exit /b 1
)

echo Models found.
echo.

REM ----------------------------------------
REM Start application menu
REM ----------------------------------------
echo [5/5] Starting application...
echo.

:MENU

echo ========================================
echo             MAIN MENU
echo ========================================
echo.
echo 1. Enroll New User
echo 2. Start Door Lock
echo 3. Exit
echo.
set /p choice="Enter your choice: "

if "%choice%"=="1" goto ENROLL
if "%choice%"=="2" goto START
if "%choice%"=="3" goto END

echo.
echo Invalid choice.
echo.
goto MENU

:ENROLL

echo.
echo ========================================
echo          USER ENROLLMENT
echo ========================================
echo.

".venv\Scripts\python.exe" enroll_user.py

echo.
echo Enrollment finished.
echo.
pause
goto MENU

:START

echo.
echo ========================================
echo          STARTING DOOR LOCK
echo ========================================
echo.

".venv\Scripts\python.exe" door_lock.py

echo.
echo Door lock application closed.
echo.
pause
goto MENU

:END

echo.
echo Goodbye!
echo.
endlocal