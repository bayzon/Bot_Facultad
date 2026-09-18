@echo off
REM Bot Facultad - start webhook server (Windows)
REM Activates env if present, starts uvicorn webhook_server:app, shows banner. Errors are visible.

setlocal
set PORT=8000
if not "%1"=="" set PORT=%1

echo ========================================
echo  BOT FACULTAD
echo  Webhook http://localhost:%PORT%/webhook
echo  Health  http://localhost:%PORT%/health
echo  Esperando mensajes...
echo ========================================
echo.

if exist "env\Scripts\activate.bat" (
  echo [INFO] Activando env...
  call env\Scripts\activate.bat
) else (
  echo [INFO] env no encontrado, usando python global.
)

echo [INFO] Iniciando uvicorn webhook_server:app en puerto %PORT%...
python -m uvicorn webhook_server:app --host 0.0.0.0 --port %PORT%
set ERR=%ERRORLEVEL%
echo.
echo [INFO] Servidor terminado con codigo %ERR%.
pause
endlocal
