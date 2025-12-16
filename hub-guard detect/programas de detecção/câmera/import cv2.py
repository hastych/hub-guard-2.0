import cv2

# Inicialize a captura de vídeo
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Erro ao abrir a webcam")
else:
    while True:
        validacao, frame = webcam.read()
        if not validacao:
            print("Erro ao capturar o frame da webcam")
            break

        # Mostre o frame na janela
        cv2.imshow("Webcam", frame)

        # Se a tecla 'Esc' (código 27) for pressionada, o loop é interrompido
        if cv2.waitKey(5) == 27:
            break

    # Libere a webcam e feche as janelas
    webcam.release()
    cv2.destroyAllWindows()