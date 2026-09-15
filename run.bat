@echo off
setlocal EnableExtensions EnableDelayedExpansion

title AI Face Door Lock

cd /d "%~dp0"

echo.
echo ========================================
echo        AI FACE DOOR LOCK
echo ========================================
echo.

REM ============================================================
REM STEP 1 - CHECK GIT CLONE DIRECTORY
REM ============================================================

if not exist "door_lock.py" (
    echo [ERROR] Project files are missing.
    echo.
    echo Make sure you are running this file from the
    echo AI-face-doorlock project folder.
    echo.
    pause
    exit /b 1
)

REM ============================================================
REM STEP 2 - CHECK PYTHON 3.12
REM ============================================================

echo [1/7] Checking Python 3.12...
echo.

set "PYTHON_CMD="

py -3.12 --version >nul 2>&1

if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3.12"
    echo Python 3.12 found.
    py -3.12 --version
    goto PYTHON_READY
)

echo Python 3.12 was not found.
echo.

REM ============================================================
REM STEP 3 - TRY TO INSTALL PYTHON 3.12
REM ============================================================

echo [2/7] Installing Python 3.12...
echo.

where winget >nul 2>&1

if %errorlevel% neq 0 (
    echo [ERROR] Windows Package Manager ^(winget^) was not found.
    echo.
    echo Please install Python 3.12 manually from:
    echo https://www.python.org/downloads/
    echo.
    echo After installing Python 3.12, run run.bat again.
    echo.
    pause
    exit /b 1
)

echo Trying winget...
echo.

winget install --id Python.Python.3.12 --exact --source winget --accept-source-agreements --accept-package-agreements

echo.
echo Checking whether Python 3.12 was actually installed...
echo.

REM Give Windows a moment to update PATH/launcher information
timeout /t 3 /nobreak >nul

py -3.12 --version >nul 2>&1

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python 3.12 installation could not be verified.
    echo.
    echo Possible reasons:
    echo - Internet connection problem
    echo - winget installation failed
    echo - Python installer requires user interaction
    echo.
    echo Please install Python 3.12 manually from:
    echo https://www.python.org/downloads/
    echo.
    echo Then run run.bat again.
    echo.
    pause
    exit /b 1
)

set "PYTHON_CMD=py -3.12"

echo Python 3.12 successfully installed.
%PYTHON_CMD% --version

:PYTHON_READY

echo.
echo ========================================
echo Python environment ready.
echo ========================================
echo.

REM ============================================================
REM STEP 4 - CREATE VIRTUAL ENVIRONMENT
REM ============================================================

echo [3/7] Checking virtual environment...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv

    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )

    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

set "VENV_PYTHON=.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo.
    echo [ERROR] Virtual environment Python was not found.
    echo.
    pause
    exit /b 1
)

echo.

REM ============================================================
REM STEP 5 - CHECK / INSTALL PYTHON PACKAGES
REM ============================================================

echo [4/7] Checking required Python packages...
echo.

"%VENV_PYTHON%" -c "import cv2, mediapipe, numpy" >nul 2>&1

if %errorlevel% equ 0 (
    echo Required packages are already installed.
    goto PACKAGES_READY
)

echo Required packages are missing.
echo Installing packages...
echo.

if not exist "requirements.txt" (
    echo.
    echo [ERROR] requirements.txt was not found.
    echo.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m pip install --upgrade pip

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] pip upgrade failed.
    echo Check your internet connection.
    echo.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Required packages could not be installed.
    echo.
    echo Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo Packages installed successfully.

:PACKAGES_READY

echo.
echo ========================================
echo Python packages ready.
echo ========================================
echo.

REM ============================================================
REM STEP 6 - CHECK AI MODELS
REM ============================================================

echo [5/7] Checking AI models...
echo.

set "MODEL_ERROR=0"

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo [MISSING] face_detection_yunet_2023mar.onnx
    set "MODEL_ERROR=1"
)

if not exist "models\face_recognition_sface_2021dec.onnx" (
    echo [MISSING] face_recognition_sface_2021dec.onnx
    set "MODEL_ERROR=1"
)

if not exist "models\face_landmarker.task" (
    echo [MISSING] face_landmarker.task
    set "MODEL_ERROR=1"
)

if not exist "models\blaze_face_short_range.tflite" (
    echo [MISSING] blaze_face_short_range.tflite
    set "MODEL_ERROR=1"
)

if "!MODEL_ERROR!"=="1" (
    echo.
    echo [ERROR] One or more AI model files are missing.
    echo.
    echo Make sure the complete GitHub repository was cloned.
    echo.
    pause
    exit /b 1
)

echo All AI models are present.
echo.

REM ============================================================
REM STEP 7 - CREATE REQUIRED DIRECTORIES
REM ============================================================

echo [6/7] Preparing project folders...
echo.

if not exist "known_faces" mkdir "known_faces"

if not exist "logs" mkdir "logs"

echo Project folders ready.
echo.

REM ============================================================
REM FINAL CHECK
REM ============================================================

echo [7/7] Performing final system check...
echo.

"%VENV_PYTHON%" -c "import cv2; import mediapipe; import numpy; print('OpenCV:', cv2.__version__); print('MediaPipe:', mediapipe.__version__); print('NumPy:', numpy.__version__)"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Final Python environment check failed.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo        SETUP COMPLETE
echo ========================================
echo.

:MENU

echo ========================================
echo          AI FACE DOOR LOCK
echo ========================================
echo.
echo  1. Enroll New User
echo  2. Start Door Lock
echo  3. Exit
echo.
set /p "CHOICE=Select an option: "

if "%CHOICE%"=="1" goto ENROLL

if "%CHOICE%"=="2" goto START

if "%CHOICE%"=="3" goto EXIT

echo.
echo Invalid option.
echo.
goto MENU


:ENROLL

echo.
echo ========================================
echo          USER ENROLLMENT
echo ========================================
echo.

"%VENV_PYTHON%" enroll_user.py

echo.
echo ========================================
echo Enrollment process finished.
echo ========================================
echo.

pause
cls
goto MENU


:START

echo.
echo ========================================
echo          STARTING DOOR LOCK
echo ========================================
echo.

if not exist "known_faces\*.npy" (
    echo [WARNING] No authorized users are enrolled.
    echo.
    echo Please select option 1 first and enroll a user.
    echo.
    pause
    cls
    goto MENU
)

"%VENV_PYTHON%" door_lock.py

echo.
echo ========================================
echo Door lock program stopped.
echo ========================================
echo.

pause
cls
goto MENU


:EXIT

echo.
echo ========================================
echo       Thank you for using the
echo          AI Face Door Lock
echo ========================================
echo.

timeout /t 2 /nobreak >nul

endlocal
exit /b 0