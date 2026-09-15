@echo off
setlocal EnableExtensions
title AI Face Door Lock

cd /d "%~dp0"

echo.
echo ========================================
echo        AI FACE DOOR LOCK
echo        Automatic Setup
echo ========================================
echo.

REM ==================================================
REM 1. CHECK / INSTALL PYTHON 3.12
REM ==================================================

echo [1/6] Checking Python 3.12...

py -3.12 --version >nul 2>&1

if errorlevel 1 (
    echo Python 3.12 is not installed.
    echo.

    where winget >nul 2>&1

    if errorlevel 1 (
        echo ERROR: winget is not available.
        echo.
        echo Please install Python 3.12 manually from:
        echo https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )

    echo Installing Python 3.12 automatically...
    echo This may require administrator permission.
    echo.

    winget install --id Python.Python.3.12 ^
        --exact ^
        --accept-source-agreements ^
        --accept-package-agreements

    if errorlevel 1 (
        echo.
        echo ERROR: Python 3.12 installation failed.
        echo.
        pause
        exit /b 1
    )

    echo.
    echo Python installation completed.
    echo Please restart this script once if Python is not detected.
    echo.
)

py -3.12 --version

if errorlevel 1 (
    echo.
    echo ERROR: Python 3.12 could not be detected.
    echo Please close this window and run run.bat again.
    echo.
    pause
    exit /b 1
)

echo.


REM ==================================================
REM 2. CREATE VIRTUAL ENVIRONMENT
REM ==================================================

echo [2/6] Setting up virtual environment...

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...

    py -3.12 -m venv .venv

    if errorlevel 1 (
        echo.
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
)

echo Virtual environment ready.
echo.


REM ==================================================
REM 3. INSTALL DEPENDENCIES
REM ==================================================

echo [3/6] Checking dependencies...

".venv\Scripts\python.exe" -c "import cv2, mediapipe, numpy" >nul 2>&1

if errorlevel 1 (
    echo Installing required packages...
    echo.

    ".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check

    ".venv\Scripts\python.exe" -m pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo ERROR: Package installation failed.
        pause
        exit /b 1
    )
) else (
    echo Dependencies already installed.
)

echo.


REM ==================================================
REM 4. CHECK AI MODELS
REM ==================================================

echo [4/6] Checking AI models...

set MODEL_ERROR=0

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo [MISSING] YuNet face detection model
    set MODEL_ERROR=1
)

if not exist "models\face_recognition_sface_2021dec.onnx" (
    echo [MISSING] SFace face recognition model
    set MODEL_ERROR=1
)

if not exist "models\face_landmarker.task" (
    echo [MISSING] MediaPipe Face Landmarker model
    set MODEL_ERROR=1
)

if not exist "models\blaze_face_short_range.tflite" (
    echo [WARNING] BlazeFace model not found
)

if "%MODEL_ERROR%"=="1" (
    echo.
    echo ERROR: One or more required AI models are missing.
    echo Please make sure the models folder was downloaded.
    echo.
    pause
    exit /b 1
)

echo AI models ready.
echo.


REM ==================================================
REM 5. CREATE PROJECT FOLDERS
REM ==================================================

echo [5/6] Preparing project folders...

if not exist "known_faces" mkdir known_faces
if not exist "logs" mkdir logs

echo Project folders ready.
echo.


REM ==================================================
REM 6. APPLICATION MENU
REM ==================================================

echo [6/6] Setup complete!
echo.

:MENU

echo ========================================
echo             AI FACE DOOR LOCK
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
echo Invalid choice. Please enter 1, 2 or 3.
echo.
goto MENU


:ENROLL

echo.
echo ========================================
echo             USER ENROLLMENT
echo ========================================
echo.

".venv\Scripts\python.exe" enroll_user.py

echo.
echo Enrollment process finished.
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
echo Door lock application stopped.
echo.
pause
goto MENU


:END

echo.
echo Thank you for using AI Face Door Lock.
echo.

endlocal
exit /b 0