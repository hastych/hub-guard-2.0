from machine import Pin, Timer
import network
import socket
import utime

# Configuração dos LEDs
led1 = Pin(22, Pin.OUT)  # LED FECHADO (D22)
led2 = Pin(23, Pin.OUT)  # LED ABERTO (D23)

# Estado inicial: FECHADO
estado_fechadura = "FECHADO"

def atualizar_leds():
    if estado_fechadura == "FECHADO":
        led1.value(1)
        led2.value(0)
    else:
        led1.value(0)
        led2.value(1)

atualizar_leds()

# Configuração do Wi-Fi (Access Point)
SSID = "Fechadura_ESP32"
SENHA = "12345678"

# Página HTML (sem formatação complicada)
HTML_INICIO = """<!DOCTYPE html>
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
            min-height: 100vh;
        }
        .container {
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            display: inline-block;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        h1 {
            margin-bottom: 30px;
        }
        .status {
            font-size: 1.5em;
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            background: rgba(0,0,0,0.3);
        }
        .fechado {
            color: #ff6b6b;
        }
        .aberto {
            color: #51cf66;
        }
        button {
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 20px 40px;
            font-size: 1.5em;
            border-radius: 50px;
            cursor: pointer;
            margin: 20px;
            font-weight: bold;
        }
        button.abrir {
            background: #51cf66;
        }
        button.fechar {
            background: #ff6b6b;
        }
        .icone {
            font-size: 4em;
            margin: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icone" id="icone">🔒</div>
        <h1>Fechadura Virtual</h1>
        <div class="status">
            Status: <strong id="statusTexto">FECHADO</strong>
        </div>
        <button id="botao" onclick="alternar()">🔓 ABRIR</button>
        <p>Clique no botão para abrir/fechar</p>
    </div>
    
    <script>
        function alternar() {
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    var resposta = JSON.parse(xhr.responseText);
                    var statusTexto = document.getElementById("statusTexto");
                    var botao = document.getElementById("botao");
                    var icone = document.getElementById("icone");
                    
                    if (resposta.estado == "ABERTO") {
                        statusTexto.innerHTML = "ABERTO";
                        statusTexto.className = "aberto";
                        botao.innerHTML = "🔒 FECHAR";
                        botao.className = "fechar";
                        icone.innerHTML = "🔓";
                        document.querySelector(".status").style.color = "#51cf66";
                    } else {
                        statusTexto.innerHTML = "FECHADO";
                        statusTexto.className = "fechado";
                        botao.innerHTML = "🔓 ABRIR";
                        botao.className = "abrir";
                        icone.innerHTML = "🔒";
                        document.querySelector(".status").style.color = "#ff6b6b";
                    }
                }
            };
            xhr.open("GET", "/alternar", true);
            xhr.send();
        }
    </script>
</body>
</html>
"""

def conectar_wifi():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=SSID, password=SENHA, authmode=network.AUTH_WPA_WPA2_PSK)
    
    while not ap.active():
        utime.sleep(0.1)
    
    print("✅ Rede Wi-Fi criada!")
    print("📡 SSID: " + SSID)
    print("🔑 Senha: " + SENHA)
    print("🌐 IP do ESP32: " + ap.ifconfig()[0])
    print("=" * 40)
    print("Conecte seu celular/PC à rede Wi-Fi")
    print("Depois acesse: http://" + ap.ifconfig()[0])
    print("=" * 40)
    
    return ap.ifconfig()[0]

def iniciar_servidor(ip):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', 80))
    server_socket.listen(5)
    print("✅ Servidor rodando em http://" + ip)
    
    while True:
        try:
            client_socket, addr = server_socket.accept()
            print("📱 Cliente conectado: " + str(addr))
            
            request = client_socket.recv(1024).decode()
            
            if "GET /alternar" in request:
                global estado_fechadura
                if estado_fechadura == "FECHADO":
                    estado_fechadura = "ABERTO"
                    print("🔓 Fechadura ABERTA!")
                else:
                    estado_fechadura = "FECHADO"
                    print("🔒 Fechadura FECHADA!")
                
                atualizar_leds()
                
                response = '{"estado": "' + estado_fechadura + '"}'
                client_socket.send("HTTP/1.1 200 OK\r\n")
                client_socket.send("Content-Type: application/json\r\n")
                client_socket.send("Content-Length: " + str(len(response)) + "\r\n")
                client_socket.send("\r\n")
                client_socket.send(response)
                
            else:
                response = HTML_INICIO
                client_socket.send("HTTP/1.1 200 OK\r\n")
                client_socket.send("Content-Type: text/html\r\n")
                client_socket.send("Content-Length: " + str(len(response)) + "\r\n")
                client_socket.send("\r\n")
                client_socket.send(response)
            
            client_socket.close()
            
        except Exception as e:
            print("❌ Erro: " + str(e))
            try:
                client_socket.close()
            except:
                pass

# Execução principal
print("=" * 40)
print("🚀 Iniciando Fechadura Virtual")
print("=" * 40)

atualizar_leds()
print("✅ LEDs configurados")

ip = conectar_wifi()
iniciar_servidor(ip)