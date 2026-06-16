from machine import Pin
import network
import socket
import utime
import json

# ===== CONFIGURAÇÕES =====
SSID = "LabRedes"
SENHA = "iNFO@g203"

# ===== LEDS =====
led_fechado = Pin(22, Pin.OUT)  # GPIO22 - VERMELHO (FECHADO)
led_aberto = Pin(23, Pin.OUT)   # GPIO23 - VERDE (ABERTO)

# ===== ESTADO =====
estado = "FECHADO"

def atualizar_leds():
    if estado == "FECHADO":
        led_fechado.value(1)
        led_aberto.value(0)
        print("🔒 FECHADO")
    else:
        led_fechado.value(0)
        led_aberto.value(1)
        print("🔓 ABERTO")

atualizar_leds()

# ===== CONECTAR WI-FI =====
def conectar_wifi():
    print(f"\n📡 Conectando ao Wi-Fi: {SSID}")
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        wlan.disconnect()
        utime.sleep(1)
    
    wlan.connect(SSID, SENHA)
    
    tentativas = 0
    while not wlan.isconnected() and tentativas < 60:
        print(".", end="")
        utime.sleep(0.5)
        tentativas += 1
    
    print("")
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"✅ Conectado! IP: {ip}")
        return ip
    else:
        print(f"❌ Falha ao conectar!")
        return None

# ===== PÁGINA HTML =====
HTML_PAGINA = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HubGuard - ESP32</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 400px;
            width: 100%;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #667eea; }
        .status { font-size: 24px; margin: 20px 0; padding: 15px; border-radius: 10px; }
        .fechado { background: #f8d7da; color: #721c24; }
        .aberto { background: #d4edda; color: #155724; }
        .btn {
            display: inline-block;
            padding: 12px 30px;
            margin: 5px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            color: white;
            text-decoration: none;
        }
        .btn-abrir { background: #28a745; }
        .btn-fechar { background: #dc3545; }
        .btn:hover { opacity: 0.8; }
        .ip { margin-top: 20px; font-size: 12px; color: #999; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🔐 HubGuard</h1>
        <p>ESP32 - Fechadura</p>
        <div class="status STATUS_CLASS">🔒 STATUS_TEXTO</div>
        <div>
            <a href="/control?acao=abrir" class="btn btn-abrir">🚪 ABRIR</a>
            <a href="/control?acao=fechar" class="btn btn-fechar">🔒 FECHAR</a>
        </div>
        <div class="ip">📡 IP: IP_ATUAL</div>
    </div>
</body>
</html>
"""

# ===== SERVIR PÁGINA =====
def servir_pagina():
    status_texto = "FECHADO 🔒" if estado == "FECHADO" else "ABERTO 🔓"
    status_class = "fechado" if estado == "FECHADO" else "aberto"
    
    html = HTML_PAGINA
    html = html.replace("STATUS_TEXTO", status_texto)
    html = html.replace("STATUS_CLASS", status_class)
    html = html.replace("IP_ATUAL", ip)
    return html

# ===== PROCESSAR REQUISIÇÕES =====
def processar(request):
    global estado
    
    # Controle
    if '/control' in request:
        if 'acao=abrir' in request:
            estado = "ABERTO"
            atualizar_leds()
            return '{"status":"aberto","mensagem":"Porta destrancada!"}'
        
        elif 'acao=fechar' in request:
            estado = "FECHADO"
            atualizar_leds()
            return '{"status":"fechado","mensagem":"Porta trancada!"}'
        
        else:
            return '{"status":"erro","mensagem":"Comando inválido"}'
    
    # Status
    elif '/status' in request:
        return json.dumps({
            "status": "aberto" if estado == "ABERTO" else "fechado",
            "ip": ip,
            "estado": estado
        })
    
    # Página inicial
    else:
        return servir_pagina()

# ===== INICIAR SERVIDOR =====
def iniciar():
    global ip
    
    ip = conectar_wifi()
    if ip is None:
        print("❌ Sem Wi-Fi. Criando AP...")
        ap = network.WLAN(network.AP_IF)
        ap.active(True)
        ap.config(essid="HubGuard_ESP", password="12345678")
        ip = ap.ifconfig()[0]
        print(f"📡 AP criado! IP: {ip}")
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', 80))
    s.listen(5)
    
    print("\n" + "="*50)
    print("✅ SERVIDOR INICIADO!")
    print(f"🌐 http://{ip}/")
    print("="*50 + "\n")
    
    while True:
        try:
            conn, addr = s.accept()
            request = conn.recv(1024).decode()
            
            resposta = processar(request)
            
            # Determinar o tipo de resposta
            if resposta.startswith('{'):
                conn.send('HTTP/1.1 200 OK\r\n')
                conn.send('Content-Type: application/json\r\n')
                conn.send('Access-Control-Allow-Origin: *\r\n')
            elif resposta.startswith('<!DOCTYPE'):
                conn.send('HTTP/1.1 200 OK\r\n')
                conn.send('Content-Type: text/html\r\n')
            else:
                conn.send('HTTP/1.1 200 OK\r\n')
                conn.send('Content-Type: text/plain\r\n')
            
            conn.send('Content-Length: ' + str(len(resposta)) + '\r\n')
            conn.send('\r\n')
            conn.send(resposta)
            conn.close()
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            try:
                conn.close()
            except:
                pass

# ===== INICIAR =====
print("\n🚀 INICIANDO ESP32 - HUBGUARD")
iniciar()