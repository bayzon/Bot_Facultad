import subprocess, time
proc = subprocess.Popen(
    ["node","pairing.js","5493815397265"],
    cwd=r"C:\Users\Juan Ignacio Bizone\Desktop\Bot_Facultad\wa_qr_bot",
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
print("Iniciando pairing, esperando código (max 20s)...")
start=time.time()
out=""
while time.time()-start < 20:
    line = proc.stdout.readline()
    if not line:
        if proc.poll() is not None:
            break
        time.sleep(0.2)
        continue
    print(line, end='')
    out+=line
    if "TU CÓDIGO" in line or "CÓDIGO:" in line:
        # capture next lines
        for _ in range(5):
            l=proc.stdout.readline()
            if l: print(l,end=''); out+=l
        break
    if "FAIL" in line:
        break
# keep alive a bit
print("\n[Dejando vivo 80s para que ingreses el código...]")
time.sleep(80)
proc.terminate()
try:
    remaining,_=proc.communicate(timeout=3)
    print(remaining[-2000:])
except: proc.kill()
