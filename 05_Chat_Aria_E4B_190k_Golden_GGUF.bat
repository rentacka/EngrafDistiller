@echo off
chcp 65001 > nul
title [Aria Studio] Chat with Aria E4B-it 190k Golden GGUF (llama.cpp)

echo ===============================================================================
echo [Aria Studio] Interactive Chat: Gemma-4 E4B-it 190k Golden GGUF
echo Model : H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it-190k-golden-Q4_K_M.gguf
echo Engine: llama-cli.exe (CUDA 12.4 Native)
echo ===============================================================================
echo.

set LLAMA_CLI=H:\llama.cpp\llama-b11719-bin-win-cuda-12.4-x64\llama-cli.exe
set MODEL=H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it-190k-golden-Q4_K_M.gguf

if not exist "%MODEL%" (
    echo [ERROR] Model not found: %MODEL%
    echo Please run 04_Pipeline_Bake_E4B_190k_Golden_GGUF.bat first!
    pause
    exit /b 1
)

"%LLAMA_CLI%" -m "%MODEL%" -ngl 99 -c 4096 -n 512 -cnv --chat-template gemma2

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Chat session ended with code: %ERRORLEVEL%
    pause
)

