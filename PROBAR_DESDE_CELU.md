# Probar desde el celu (solo 3 pasos manuales)
1. Meta > App > WhatsApp > API Setup: agregar tu +54 como destinatario de prueba.
2. Meta > Configuration > Webhook: Callback URL nueva + VERIFY_TOKEN > Verify and Save > suscribir campo `messages`.
3. En el server: poner JSON real de Service Account + GOOGLE_DRIVE_FOLDER_ID real en `.env` y reiniciar (`start_bot.bat`).
Verificar: enviar `Redes de Datos: ¿Qué es una VLAN?` → responde `✅ Pregunta guardada y respondida`.

## Bot 24/7 gratis (Render + UptimeRobot, solo dashboard, sin comandos remotos)
> Resultado: el bot responde sin tu PC encendida. El túnel local queda solo para desarrollo.

### Render (plan free)
1. Subir este repo a GitHub (sin `.env` ni `*-sa.json`, ya están en `.gitignore`).
2. Render > New > Web Service > conectar el repo (usa `render.yaml`: `uvicorn webhook_server:app`, health check `/health`).
3. Environment: cargar cada variable de `.env.example` con su valor real (VERIFY_TOKEN, WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_APP_SECRET, GEMINI_API_KEY u OPENAI_API_KEY, GOOGLE_SERVICE_ACCOUNT_JSON, GOOGLE_DRIVE_FOLDER_ID).
4. Disk: crear disco de 1 GB montado en `chroma_db` (sin disco, cada redeploy borra el índice y hay que reindexar).
5. Deploy > copiar la URL pública `https://<app>.onrender.com`.

### Meta (apunta al Render, jubila el túnel)
1. Meta > Configuration > Webhook: Callback URL `https://<app>.onrender.com/webhook` + VERIFY_TOKEN > Verify and Save.
2. Suscribir el campo `messages`.
3. Probar desde el celu: `Gestión de Calidad: ¿qué es calidad?` → responde `✅ Pregunta guardada y respondida`.

### UptimeRobot (evita que el free se duerma)
1. New Monitor > tipo HTTP(s), URL `https://<app>/health`, intervalo 5 min.
2. Verificar: el monitor muestra UP y `/health` responde `200`.

### Límites a tener en cuenta
- [ ] Plan free se suspende sin tráfico: UptimeRobot lo mantiene despierto, pero puede responder lento el primer mensaje.
- [ ] Sin disco, cada redeploy borra `chroma_db/` → reindexar con `python main.py --reindex` en local y volver a subir, o reindexar contra el disco.
- [ ] `materias_docs.json` e `historial_local.json` son caché local: sin disco persistente se recrean vacíos en cada deploy.
- [ ] El QR (`run_qr.py`) no se puede hostear en Render: el QR se escanea una sola vez desde tu PC.
