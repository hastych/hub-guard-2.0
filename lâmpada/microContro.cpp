#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "SEU_SSID";
const char* password = "SUA_SENHA";

WebServer server(80);

const int relePin = 5; // Substitua pelo pino ao qual o relé está conectado

void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>Controle de Relé</title></head><body>";
  html += "<h1>Controle de Relé</h1>";
  html += "<form action=\"/liga\" method=\"POST\"><button type=\"submit\">Ligar</button></form>";
  html += "<form action=\"/desliga\" method=\"POST\"><button type=\"submit\">Desligar</button></form>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void ligaRele() {
  digitalWrite(relePin, LOW); // Liga o relé
  server.sendHeader("Location", "/");
  server.send(303);
}

void desligaRele() {
  digitalWrite(relePin, HIGH); // Desliga o relé
  server.sendHeader("Location", "/");
  server.send(303);
}

void setup() {
  Serial.begin(115200);
  pinMode(relePin, OUTPUT);
  digitalWrite(relePin, HIGH); // Inicialmente desligar o relé

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Conectando ao WiFi...");
  }

  Serial.println("Conectado ao WiFi");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/liga", HTTP_POST, ligaRele);
  server.on("/desliga", HTTP_POST, desligaRele);

  server.begin();
  Serial.println("Servidor iniciado");
}

void loop() {
  server.handleClient();
}