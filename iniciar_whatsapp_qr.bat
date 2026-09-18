@echo off
REM Bot Facultad — WhatsApp QR solo para +54 9 3815397265
REM Inicia local_app + bot QR

echo ============================================
echo  Bot Facultad — WhatsApp QR (solo vos)
echo  Numero permitido: +54 9 3815397265 (5493815397265)
echo ============================================
echo.

if exist "env\Scripts\activate.bat" call env\Scripts\activate.bat

echo [1/2] Verificando local_app en http://localhost:8000 ...
powershell -command "try { Invoke-WebRequest -Uri http://127.0.0.1:8000/api/materias -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel% neq 0 (
    echo   local_app no esta corriendo — iniciandolo en segundo plano...
    start "" /MIN python -u -m uvicorn local_app:app --host 0.0.0.0 --port 8000
    echo   esperando 4s...
    timeout /t 4 /nobreak >nul
)

echo [2/2] Iniciando bot WhatsApp QR...
echo   Cuando aparezca el QR, escanealo con:
echo   WhatsApp ^> Ajustes ^> Dispositivos vinculados ^> Vincular dispositivo
echo   Solo respondera a tu numero: 5493815397265
echo.

cd wa_qr_bot
set ALLOWED_NUMBERS=5493815397265
set RAG_URL=http://localhost:8000
set IGNORE_GROUP=true
node bot.js

pause
