#!/usr/bin/env node
/**
 * wa_qr_bot/bot.js — Bot Facultad WhatsApp via QR (alternativa a Meta Cloud API)
 *
 * Para USO PERSONAL SOLO VOS. No necesitas Meta Dashboard, ni VERIFY_TOKEN,
 * ni WHATSAPP_TOKEN, ni ngrok, ni Render.
 *
 * Cómo funciona:
 *   1. Escaneás un QR con tu WhatsApp (como WhatsApp Web)
 *   2. Este script queda conectado como si fuera un navegador
 *   3. Cuando te escribís a VOS MISMO (o te escribe alguien si usás número secundario),
 *      el bot intercepta el mensaje, lo manda a tu local_app (http://localhost:8000/api/chat)
 *      y te responde con la respuesta del RAG + fuentes.
 *
 * Requisitos:
 *   - Node 18+ (tenés v22)
 *   - local_app.py corriendo en http://localhost:8000 (ya lo tenés)
 *   - Primer arranque descarga Chromium (~150MB) para whatsapp-web.js
 *
 * Uso:
 *   cd wa_qr_bot
 *   npm install
 *   npm start
 *   -> aparece QR en consola -> escanear con WhatsApp > Dispositivos vinculados > Vincular
 *   -> luego mandá mensaje: "Gestion de Calidad: que dice iso 9001?"
 *
 * Ventajas vs Meta Cloud API:
 *   ✅ No tokens que expiran, no verificación, no webhook público
 *   ✅ 5 minutos vs 2 horas de config
 *   ✅ Ideal para 1 usuario (vos)
 *   ⚠️  Necesita PC encendida (igual que local_app)
 *   ⚠️  WhatsApp no permite automatización oficial en números personales para spam masivo,
 *       pero para uso académico personal de bajo volumen no hay problema. Para producción
 *       masiva, usar Meta Cloud API (webhook_server.py).
 */

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');

const RAG_URL = process.env.RAG_URL || 'http://localhost:8000';
const API_CHAT = `${RAG_URL}/api/chat`;
const API_MATERIAS = `${RAG_URL}/api/materias`;

// Solo responde a estos números si están definidos (vacío = responde a todos)
// Para uso solo-vos: poné tu número con país sin + ni espacios, ej "5491123456789"
const ALLOWED_NUMBERS = (process.env.ALLOWED_NUMBERS || '').split(',').map(s => s.trim()).filter(Boolean);
const IGNORE_GROUP = process.env.IGNORE_GROUP !== 'false'; // ignora grupos por defecto

console.log('='.repeat(72));
console.log(' Bot Facultad — WhatsApp QR (uso personal)');
console.log(` RAG: ${API_CHAT}`);
console.log(` Permitidos: ${ALLOWED_NUMBERS.length ? ALLOWED_NUMBERS.join(', ') : 'TODOS (solo vos deberías escribirle)'}`);
console.log(` Grupos: ${IGNORE_GROUP ? 'ignora' : 'responde'}`);
console.log('='.repeat(72));

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: './.wwebjs_auth' }),
  puppeteer: {
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  },
});

client.on('qr', (qr) => {
  console.log('\n[QR] Escaneá este código con WhatsApp > Dispositivos vinculados > Vincular dispositivo:\n');
  qrcode.generate(qr, { small: true });
  console.log('\n[QR] Si no ves el QR bien, copiá este string a https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=' + encodeURIComponent(qr));
  console.log('[QR] Tenés 60 segundos antes que expire y genere otro.\n');
});

client.on('authenticated', () => {
  console.log('[OK] Autenticado — sesión guardada en .wwebjs_auth/');
});

client.on('auth_failure', (msg) => {
  console.error('[FAIL] Auth failure:', msg);
});

client.on('ready', async () => {
  console.log('\n[READY] WhatsApp conectado! ✅');
  console.log(' Mandate un mensaje a este número (vos mismo) tipo:');
  console.log('   Gestion de Calidad: que dice la iso 9001 en general');
  console.log('   Analisis Numerico: que es el metodo de Newton?\n');
  try {
    const r = await axios.get(API_MATERIAS, { timeout: 5000 });
    console.log('[INFO] Materias disponibles (desde datos/):', (r.data.materias || []).join(', '));
  } catch (e) {
    console.log('[WARN] No se pudo contactar local_app en', RAG_URL, '- asegurate que python local_app.py esté corriendo');
  }
  console.log('\n Para salir: Ctrl+C\n');
});

client.on('message', async (msg) => {
  try {
    // Filtros
    if (msg.fromMe && !process.env.RESPOND_TO_SELF) {
      // por defecto NO responde a mensajes que vos te mandás a vos mismo desde el mismo dispositivo?
      // whatsapp-web.js marca fromMe=true incluso si te escribís a vos mismo desde otro lado.
      // Para uso personal queremos que SÍ responda cuando te escribís a vos mismo (notas).
      // Si fromMe y el chat es tu propio número, igual procesar.
    }
    const chat = await msg.getChat();
    if (chat.isGroup && IGNORE_GROUP) return;

    const from = msg.from; // ej 54911...@c.us
    const fromNumber = from.split('@')[0];
    if (ALLOWED_NUMBERS.length && !ALLOWED_NUMBERS.includes(fromNumber)) {
      // silencioso para no spamear desconocidos; descomenta para responder con ayuda
      // await msg.reply('No autorizado.');
      return;
    }

    const body = (msg.body || '').trim();
    if (!body) return;
    if (body.startsWith('/')) return; // ignora comandos

    // Evita loop si el bot se menciona a sí mismo
    if (msg.body.length > 2000) {
      await msg.reply('Mensaje muy largo, acórtalo porfa.');
      return;
    }

    // Parseo MATERIA: pregunta — si no hay ":", asumimos que es pregunta sin materia y damos ayuda
    let materia = null;
    let pregunta = null;
    const colonIdx = body.indexOf(':');
    if (colonIdx > 0) {
      materia = body.slice(0, colonIdx).trim();
      pregunta = body.slice(colonIdx + 1).trim();
    }

    if (!materia || !pregunta) {
      // Sin formato, intentamos ayudar listando materias
      try {
        const r = await axios.get(API_MATERIAS, { timeout: 4000 });
        const materias = r.data.materias || [];
        await msg.reply(
          `Formato: *MATERIA: pregunta*\n` +
          `Materias con PDFs en datos/: ${materias.join(', ')}\n` +
          `Ejemplo: Gestion de Calidad: que dice la iso 9001?`
        );
      } catch {
        await msg.reply('Formato: MATERIA: pregunta  (ej: Gestion de Calidad: que dice iso 9001?)');
      }
      return;
    }

    console.log(`[MSG] ${fromNumber}: ${materia}: ${pregunta.slice(0, 80)}`);

    // Indicador de escribiendo
    try { await chat.sendStateTyping(); } catch {}

    // Llamar a local_app
    let ragResp;
    try {
      const r = await axios.post(API_CHAT, { materia, pregunta }, { timeout: 60000 });
      ragResp = r.data;
    } catch (e) {
      const detail = e.response?.data?.detail || e.message || 'Error RAG';
      console.log('[RAG FAIL]', detail);
      // Si es materia no reconocida, reenvía el help
      if (e.response?.status === 400 && detail.includes('Materias disponibles')) {
        await msg.reply(detail);
        return;
      }
      await msg.reply(`Error consultando RAG: ${detail}\nAsegurate que local_app.py esté corriendo en ${RAG_URL}`);
      return;
    }

    const isFallback = ragResp.is_fallback;
    const respuesta = ragResp.respuesta || '';
    const fuentes = ragResp.fuentes || [];
    const canon = ragResp.materia || materia;

    // Formato corto para WhatsApp (igual que webhook_server: header + respuesta corta + fuente)
    let reply;
    if (isFallback) {
      reply = `✅ Pregunta anotada en ${canon}\n\n${respuesta}\n\nGuardado en historial: ${canon} - Bot Facultad`;
    } else {
      // Truncar a ~350 chars si es muy larga
      let short = respuesta.replace(/\s+/g, ' ').trim();
      if (short.length > 380) {
        let t = short.slice(0, 380);
        const sp = t.lastIndexOf(' ');
        if (sp > 260) t = t.slice(0, sp);
        short = t + '…';
      }
      const fuenteLine = fuentes.length ? `Fuente: ${fuentes[0].file} p. ${fuentes[0].page}` : 'Fuente: no disponible';
      reply = `✅ Pregunta anotada en ${canon}\n\n${short}\n\n${fuenteLine}\n→ Detalle completo en historial_local.json`;
    }

    await client.sendMessage(msg.from, reply);
    console.log(`[REPLY] a ${fromNumber} (${isFallback ? 'fallback' : fuentes.length + ' fuentes'})`);
    try { await chat.clearState(); } catch {}
  } catch (err) {
    console.error('[ERR] message handler', err.message || err);
    try { await msg.reply('Error interno del bot, revisá la consola.'); } catch {}
  }
});

client.on('disconnected', (reason) => {
  console.log('[DISC] Desconectado:', reason, '- reiniciá con npm start');
});

client.initialize().catch(err => {
  console.error('[FAIL] No se pudo iniciar:', err);
  console.log('Tip: borrá la carpeta .wwebjs_auth/ y probá de nuevo si el QR no anda.');
});
