#!/usr/bin/env node
/**
 * pairing.js — Genera código de 8 dígitos para vincular sin QR
 * Uso: node pairing.js 5493815397265
 */
const { Client, LocalAuth } = require('whatsapp-web.js');

const phone = process.argv[2] || process.env.PAIRING_NUMBER || '5493815397265';
const cleanPhone = phone.replace(/[^0-9]/g, '');

console.log('='.repeat(72));
console.log(' Bot Facultad — Código de vinculación (8 dígitos)');
console.log(` Número: ${cleanPhone}`);
console.log('='.repeat(72));

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: './.wwebjs_auth_pairing' }),
  puppeteer: { headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] },
});

let pairingRequested = false;

client.on('qr', (qr) => {
  console.log('[INFO] QR generado pero usaremos código de vinculación en su lugar...');
});

client.initialize().then(async () => {
  // Esperar un poco a que el cliente esté listo para pedir código
  setTimeout(async () => {
    if (pairingRequested) return;
    pairingRequested = true;
    try {
      console.log(`[PAIRING] Solicitando código para ${cleanPhone}...`);
      const code = await client.requestPairingCode(cleanPhone);
      console.log('\n' + '='.repeat(72));
      console.log(`  TU CÓDIGO: ${code}`);
      console.log('='.repeat(72));
      console.log('\nEn tu celu: WhatsApp > Ajustes > Dispositivos vinculados >');
      console.log('  Vincular dispositivo > Vincular con número de teléfono');
      console.log(`  Ingresá: ${code}`);
      console.log('\nTenés 60 segundos. Dejando el proceso vivo 90s...');
      console.log('Si expira, corré de nuevo: node pairing.js\n');
      // Mantener vivo 90s para que puedan ingresar el código
      setTimeout(() => {
        console.log('[TIMEOUT] Cerrando...');
        process.exit(0);
      }, 90000);
    } catch (err) {
      console.error('[FAIL] No se pudo generar código:', err.message || err);
      console.log('Tip: borrá .wwebjs_auth_pairing/ y probá de nuevo, o usá QR con npm start');
      process.exit(1);
    }
  }, 3000);
}).catch(err => {
  console.error('Init fail', err);
});

client.on('authenticated', () => console.log('[OK] Autenticado!'));
client.on('ready', () => console.log('[READY] Conectado! Ya podés usar el bot.'));
client.on('auth_failure', m => console.error('[FAIL] Auth', m));
