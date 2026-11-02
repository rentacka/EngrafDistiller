@echo off
chcp 65001 > nul
title [37_Studio] Download Google Gemma-4 E4B (4B Full Model)

echo ===============================================================================
echo [37_Studio] Download Google Gemma-4 E4B (Official 4B Model)
echo Target : google/gemma-4-E4B (safetensors ~14.9 GB)
echo SaveTo : H:\Aria_FreeTokenModels\gemma-4-E4B
echo ===============================================================================
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

"H:\DiffusionQwen_env\Scripts\python.exe" scripts\download_gemma4_e4b.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Download exited with error code: %ERRORLEVEL%
    pause
) else (
    echo.
    echo [SUCCESS] Gemma-4 E4B is ready!
    pause
)

