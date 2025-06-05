@echo off
REM autobuild.bat  – Run from auto_tester\ directory

setlocal ENABLEDELAYEDEXPANSION
set VENV_DIR=.venv

:: ------------------------------------------------------------------
:: 1. Python venv
:: ------------------------------------------------------------------
if not exist %VENV_DIR% (
    python -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate

:: ------------------------------------------------------------------
:: 2. Python deps
:: ------------------------------------------------------------------
python -m pip install --upgrade pip
python -m pip install "opencv-python-headless>=4.9.0" numpy requests

:: ------------------------------------------------------------------
:: 3. FFmpeg (static build)
:: ------------------------------------------------------------------
set FFMPEG_DIR=tools\ffmpeg
set FFMPEG_BIN=%FFMPEG_DIR%\bin\ffmpeg.exe

if not exist %FFMPEG_BIN% (
    echo Downloading static FFmpeg…
    if not exist tools mkdir tools
    powershell -Command ^
      "Invoke-WebRequest -Uri https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile tools\ffmpeg.zip"
    powershell -Command ^
      "Expand-Archive tools\ffmpeg.zip -DestinationPath tools"
    for /D %%d in (tools\ffmpeg-*) do (
        move %%d %FFMPEG_DIR%
    )
    del tools\ffmpeg.zip
)

:: ------------------------------------------------------------------
:: 4. Add to user PATH if missing
:: ------------------------------------------------------------------
for %%p in ("%FFMPEG_DIR%\bin") do (
    echo !PATH! | find /I "%%~p" >nul
    if errorlevel 1 (
        setx PATH "%%~p;!PATH!" >nul
        echo Added %%~p to user PATH – you may need to restart the shell.
    )
)

echo.
echo === Autobuild complete ===
echo Activate with:  call %VENV_DIR%\Scripts\activate
endlocal
