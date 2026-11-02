@echo off
chcp 65001 > nul
title 37_Studio Engram Block-128 Int4 / Int8 Quantizer

echo ================================================================================
echo   [37_Studio] Aria Engram Block-128 Int4 / Int8 Quantizer
echo   Architect: MoneyKoiKoi (@MoneyKoiKoi) x Aria
echo   Feature  : Pure PyTorch Block-128 Uniform Quantization (Zero DLL)
echo   Option   : [1] 4-Bit (136 MB / 99.5%% Cosine Sim / Ultra Compact)
echo              [2] 8-Bit (264 MB / 99.998%% Cosine Sim / Extreme Precision)
echo ================================================================================
echo.

set /p OPT="Select Bits (1: 4-Bit [Default], 2: 8-Bit) > "

set BITS=4
if "%OPT%"=="2" set BITS=8

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo.
echo Running quantization with %BITS%-Bit...
"H:\DiffusionQwen_env\Scripts\python.exe" "scripts\quantize_engram_to_int4.py" --bits %BITS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Process exited with error code: %ERRORLEVEL%
    pause
) else (
    echo.
    echo [SUCCESS] Engram successfully quantized to Int%BITS%!
    pause
)
