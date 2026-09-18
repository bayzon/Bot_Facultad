import subprocess, os, time, sys, re, json
from pathlib import Path

env = os.environ.copy()
env["ALLOWED_NUMBERS"] = "5493815397265"
env["RAG_URL"] = "http://localhost:8000"
env["IGNORE_GROUP"] = "true"

# kill previous node
try:
    subprocess.run(["taskkill","/F","/IM","node.exe"], capture_output=True)
    time.sleep(1)
except: pass

proc = subprocess.Popen(
    ["node","bot.js"],
    cwd=r"C:\Users\Juan Ignacio Bizone\Desktop\Bot_Facultad\wa_qr_bot",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

qr_data = None
start = time.time()
output_lines = []

print("Esperando QR (max 25s)...")
try:
    while time.time() - start < 25:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                print("proceso terminado", proc.poll())
                break
            time.sleep(0.2)
            continue
        output_lines.append(line)
        print(line, end='')
        # QR string is after "[QR]" and the qrcode-terminal prints ascii, but we need raw qr data
        # bot.js logs "QR" and also logs URL with data=...
        if "data=" in line:
            # extract encoded qr
            m = re.search(r"data=([^ ]+)", line)
            if m:
                import urllib.parse
                qr_data = urllib.parse.unquote(m.group(1))
                print(f"\n[CAPTURADO] QR raw len {len(qr_data)}")
                break
        if "qrcode" in line.lower() or "Escaneá" in line:
            # keep reading a bit more for data= line
            pass
        # check if ready (already authenticated)
        if "READY" in line or "Autenticado" in line:
            print("YA ESTA AUTENTICADO - no necesita QR")
            break
    # if not captured via data=, try to get from output that contains QR ascii - fallback use last lines
    if not qr_data and proc.poll() is None:
        print("No se capturó QR raw, esperando 5s más...")
        time.sleep(5)
        # try to read more
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=3)
            output_lines.append(out)
            print(out[-2000:])
            m = re.search(r"data=([^ \n]+)", out)
            if m:
                import urllib.parse
                qr_data = urllib.parse.unquote(m.group(1))
        except: pass
    else:
        # keep process running for user to scan, don't kill yet
        print(f"\n[PROCESO SIGUE VIVO PID {proc.pid}] - QR listo para escanear, dejando 60s")
        # save qr to file for display
        if qr_data:
            Path(r"C:\Users\Juan Ignacio Bizone\Desktop\Bot_Facultad\wa_qr_bot\last_qr.txt").write_text(qr_data, encoding="utf-8")
            # also save output
            Path(r"C:\Users\Juan Ignacio Bizone\Desktop\Bot_Facultad\wa_qr_bot\qr_output.txt").write_text("".join(output_lines), encoding="utf-8")
            print(f"QR guardado en last_qr.txt ({len(qr_data)} chars)")
        # leave running for 60s so user can scan
        time.sleep(60)
        # check if authenticated in that time
        proc.terminate()
        try:
            out,_ = proc.communicate(timeout=3)
            print(out[-2000:])
        except: proc.kill()
except Exception as e:
    print("error", e)
    import traceback; traceback.print_exc()
    try: proc.kill()
    except: pass
