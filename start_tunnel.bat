@echo off
REM Bot Facultad - start Cloudflare tunnel for Meta webhook (Windows)
REM Usage: start_tunnel.bat [PORT, default 8000]
REM Does not assume a fixed public URL; configure the printed URL in Meta Dashboard.

setlocal
set PORT=8000
if not "%1"=="" set PORT=%1

echo ========================================
echo  BOT FACULTAD - TUNNEL
echo  Local: http://localhost:%PORT%
echo ========================================
echo.
echo [INFO] Iniciando: cloudflared tunnel --url http://localhost:%PORT%
echo [INFO] Copia la URL https generada y configurala en Meta Dashboard:
echo        Callback URL: https://XXXX.trycloudflare.com/webhook
echo        Verify Token: el de tu .env (VERIFY_TOKEN)
echo.

cloudflared tunnel --url http://localhost:%PORT%
set ERR=%ERRORLEVEL%
echo.
echo [INFO] Tunnel terminado con codigo %ERR%.
pause
endlocal
