@echo off
cd /d "%~dp0"
set "MEDIRELY_ARTIFACT_ROOT=%cd%\artifacts_local"
set "MEDIRELY_IMAGE_ROOT=%cd%\artifacts_local\bank_images"
set "MEDIRELY_MS_DTYPE=float32"
set "MEDIRELY_MODEL_ROOT="
if exist "%cd%\medsiglip-448\config.json" set "MEDIRELY_MODEL_ROOT=%cd%"
if not defined MEDIRELY_MODEL_ROOT echo [!] medsiglip-448 not found - put it in this folder or edit MEDIRELY_MODEL_ROOT in this file. & pause & exit /b 1
echo Using MedSigLIP from: %MEDIRELY_MODEL_ROOT%\medsiglip-448
python -m gtc_demo.app.app --local-gpu
pause
