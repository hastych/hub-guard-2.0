from machine import Pin, Timer
import utime

# Configuração dos pinos dos LEDs
led1 = Pin(22, Pin.OUT)  # LED no GPIO 22
led2 = Pin(23, Pin.OUT)  # LED no GPIO 23

# Estado atual: False = LED1 ligado, True = LED2 ligado
led_ativo = False

# Inicializa com LED1 ligado e LED2 desligado
led1.value(1)  # Liga LED1
led2.value(0)  # Desliga LED2

def alternar_leds(timer):
    """Função chamada pelo timer para alternar os LEDs"""
    global led_ativo
    
    if led_ativo:
        # Se LED2 estava ativo, muda para LED1
        led1.value(1)
        led2.value(0)
        led_ativo = False
        print("LED1 ligado")
    else:
        # Se LED1 estava ativo, muda para LED2
        led1.value(0)
        led2.value(1)
        led_ativo = True
        print("LED2 ligado")

# Configura o timer para executar a cada 15 segundos (15000 ms)
timer = Timer(1)
timer.init(period=15000, mode=Timer.PERIODIC, callback=alternar_leds)

print("Programa iniciado - LEDs alternam a cada 15 segundos")
print("LED1 está ligado inicialmente")

# Loop principal - mantém o programa rodando
try:
    while True:
        utime.sleep(1)  # Sleep para não sobrecarregar a CPU
        
except KeyboardInterrupt:
    # Limpeza ao interromper o programa
    timer.deinit()
    led1.value(0)
    led2.value(0)
    print("\nPrograma encerrado - LEDs desligados")