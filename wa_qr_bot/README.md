# Bot Facultad — WhatsApp QR (uso personal)

Alternativa **sin quilombo** a la API oficial de Meta. Ideal si **solo vos** vas a usar el bot.

- No necesitas `VERIFY_TOKEN`, `WHATSAPP_TOKEN`, `ngrok` ni Deploy.
- Escaneás un QR una vez con tu WhatsApp y queda vinculado como WhatsApp Web.
- El bot lee tus mensajes y responde usando el mismo RAG de `local_app.py`.

## Requisitos

- `local_app.py` corriendo en `http://localhost:8000` (ya lo tenés)
- Node 18+ (tenés v22)
- Tu celu y la PC con WhatsApp

## Instalación (una vez)

```bash
cd wa_qr_bot
npm install
```

La primera vez descarga Chromium (~150MB) para `whatsapp-web.js`.

## Uso

1. Asegurate que el bot local esté corriendo:
   ```bash
   # en otra terminal
   python local_app.py
   # o doble click iniciar_bot_local.bat
   ```

2. Iniciá el bot QR:
   ```bash
   cd wa_qr_bot
   npm start
   ```

3. Aparece un QR en la consola → en tu celu: **WhatsApp > Ajustes > Dispositivos vinculados > Vincular dispositivo** → escaneá.

4. Probá mandándote un mensaje a vos mismo:
   ```
   Gestion de Calidad: que dice la iso 9001 en general
   Analisis Numerico: que es el metodo de biseccion?
   ```

   Si mandás sin `MATERIA:` te responde con ayuda y lista de materias con PDFs.

## Solo vos

Por defecto responde a todos los que le escriban al número vinculado. Si querés que solo te responda a vos:

```bash
# en Windows PowerShell
$env:ALLOWED_NUMBERS="5491123456789"
npm start

# o en .env
ALLOWED_NUMBERS=54911XXXXXXXX
```

Ignora grupos por defecto. Para responder en grupos: `IGNORE_GROUP=false npm start`.

## Dónde se guarda la sesión

En `wa_qr_bot/.wwebjs_auth/` — no borrar si no querés volver a escanear. Si el QR no anda, borrá esa carpeta y `npm start` de nuevo.

## Ventajas vs Meta Cloud API

|  | QR (este) | Cloud API (webhook_server.py) |
|---|---|---|
| Config | QR 1 vez | Meta App + tokens + webhook público + ngrok |
| Tokens | ninguno | expiran |
| Para 1 usuario | perfecto | overkill |
| Para producción masiva | no recomendado (riesgo ban) | sí, oficial |
| PC debe estar encendida | sí | sí (o deploy a Render) |

Para entregar a toda la facultad, usar Cloud API. Para vos solo, este es 10x más simple.

## Troubleshooting

- `No se pudo contactar local_app`: iniciá `python local_app.py` antes.
- `Failed to launch chrome`: reinstalá con `npm install --force` o instalá Chrome.
- Mensajes no llegan: revisá que el número no esté bloqueado y que `ALLOWED_NUMBERS` no filtre.
