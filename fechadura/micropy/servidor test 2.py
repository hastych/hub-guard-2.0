from machine import Pin
import network
import socket
import utime

# Configuração dos LEDs
led1 = Pin(22, Pin.OUT)  # LED FECHADO
led2 = Pin(23, Pin.OUT)  # LED ABERTO

# Estado inicial: FECHADO
estado_fechadura = "FECHADO"

def atualizar_leds():
    if estado_fechadura == "FECHADO":
        led1.value(1)
        led2.value(0)
        print("🔒 Fechadura FECHADA - LED22 ligado")
    else:
        led1.value(0)
        led2.value(1)
        print("🔓 Fechadura ABERTA - LED23 ligado")

atualizar_leds()

# Configuração do Wi-Fi
SSID = "Fechadura_ESP32"
SENHA = "12345678"

# Página HTML simplificada
HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fechadura Virtual</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            display: inline-block;
        }
        .icone {
            font-size: 80px;
            margin: 20px;
        }
        .status {
            font-size: 24px;
            margin: 20px;
            padding: 10px;
            border-radius: 10px;
        }
        button {
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 20px;
            border-radius: 50px;
            cursor: pointer;
            margin: 10px;
        }
        button.abrir {
            background: #51cf66;
        }
        .info {
            font-size: 12px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icone" id="icone">🔒</div>
        <h1>Fechadura Virtual</h1>
        <div class="status" id="status">
            Status: <strong>FECHADO</strong>
        </div>
        <button id="botao" onclick="alternar()">🔓 ABRIR</button>
        <div class="info">
            Conectado ao ESP32<br>
            Clique para abrir/fechar
        </div>
    </div>
    
    <script>
        function alternar() {
            // Faz a requisição para o ESP32
            fetch('/alternar')
                .then(response => response.text())
                .then(data => {
                    // Atualiza a interface baseado na resposta
                    if (data === "ABERTO") {
                        document.getElementById('status').innerHTML = 'Status: <strong>ABERTO 🔓</strong>';
                        document.getElementById('status').style.color = '#51cf66';
                        document.getElementById('botao').innerHTML = '🔒 FECHAR';
                        document.getElementById('botao').style.background = '#ff6b6b';
                        document.getElementById('icone').innerHTML = '🔓';
                    } else {
                        document.getElementById('status').innerHTML = 'Status: <strong>FECHADO 🔒</strong>';
                        document.getElementById('status').style.color = '#ff6b6b';
                        document.getElementById('botao').innerHTML = '🔓 ABRIR';
                        document.getElementById('botao').style.background = '#51cf66';
                        document.getElementById('icone').innerHTML = '🔒';
                    }
                })
                .catch(error => {
                    console.log('Erro:', error);
                });
        }
    </script>
</body>
</html>
"""

def criar_servidor():
    # Configurar Access Point
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=SSID, password=SENHA, authmode=network.AUTH_WPA_WPA2_PSK)
    
    while not ap.active():
        utime.sleep(0.1)
    
    ip = ap.ifconfig()[0]
    print("\n" + "="*50)
    print("✅ SERVIDOR INICIADO!")
    print("="*50)
    print("📡 Rede Wi-Fi: " + SSID)
    print("🔑 Senha: " + SENHA)
    print("🌐 Acesse no navegador: http://" + ip)
    print("="*50 + "\n")
    
    # Criar socket do servidor
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', 80))
    s.listen(5)
    
    while True:
        try:
            conn, addr = s.accept()
            print("📱 Cliente conectado: " + str(addr))
            
            request = conn.recv(1024).decode()
            
            # Verificar se é requisição para alternar
            if 'GET /alternar' in request:
                global estado_fechadura
                if estado_fechadura == "FECHADO":
                    estado_fechadura = "ABERTO"
                else:
                    estado_fechadura = "FECHADO"
                
                atualizar_leds()
                
                # Enviar resposta
                response = estado_fechadura
                conn.send('HTTP/1.1 200 OK\r\n')
                conn.send('Content-Type: text/plain\r\n')
                conn.send('Access-Control-Allow-Origin: *\r\n')
                conn.send('Content-Length: ' + str(len(response)) + '\r\n')
                conn.send('\r\n')
                conn.send(response)
            
            else:
                # Enviar página HTML
                conn.send('HTTP/1.1 200 OK\r\n')
                conn.send('Content-Type: text/html\r\n')
                conn.send('Content-Length: ' + str(len(HTML)) + '\r\n')
                conn.send('\r\n')
                conn.send(HTML)
            
            conn.close()
            
        except Exception as e:
            print("❌ Erro: " + str(e))
            try:
                conn.close()
            except:
                pass

# Iniciar o programa
print("\n🚀 Iniciando Fechadura Virtual...")
print("Conectando LEDs...")
atualizar_leds()

criar_servidor()