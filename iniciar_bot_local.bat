@echo off
REM Bot Facultad — Iniciador LOCAL (sin WhatsApp)
REM Doble click para abrir el chat en el navegador

echo ========================================
echo  Bot Facultad - Modo LOCAL
echo  http://localhost:8000
echo ========================================
echo.

if exist "env\Scripts\activate.bat" call env\Scripts\activate.bat

if not exist "chroma_db" (
    echo [AVISO] No se encontro chroma_db/ - se creara al primer uso
    echo.
)

echo Iniciando servidor...
echo  - Tarda 2-3 segundos en aparecer
echo  - Primera pregunta tarda 5-8s (carga el indice), luego instantaneo
echo  - Abri manualmente: http://localhost:8000
echo  - Para salir: Ctrl+C
echo.

python local_app.py

pause
