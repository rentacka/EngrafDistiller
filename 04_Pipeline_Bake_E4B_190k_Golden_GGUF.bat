@echo off
chcp 65001 > nul
title [Aria Studio] E4B-it 190k Golden GGUF Complete Pipeline

echo ===============================================================================
echo [Aria Studio] Gemma-4 E4B-it 190k Golden GGUF Complete Pipeline
echo Architect: MoneyKoiKoi x Aria
echo.
echo Step 1: Extract 190k Golden Engram (2560D) from safetensors (CPU)
echo Step 2: Quantize Engram to Block-128 Int4 (928MB -^> ~245MB, 73%% cut)
echo Step 3: Surgical Zero-Mask Surgery on E4B-it Q4_K_M GGUF (Cut 72,227 vocabs)
echo Step 4: Smoke Test with llama-cli.exe (CUDA 12.4)
echo ===============================================================================
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

"H:\DiffusionQwen_env\Scripts\python.exe" scripts\pipeline_e4b_190k_golden_gguf.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline exited with error code: %ERRORLEVEL%
    pause
) else (
    echo.
    echo [SUCCESS] E4B-it 190k Golden GGUF Pipeline completed successfully!
    echo Output: H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it-190k-golden-Q4_K_M.gguf
    pause
)

