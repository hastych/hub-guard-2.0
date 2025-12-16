import cv2
import mediapipe as mp

# Inicializando a captura de vídeo
webcam = cv2.VideoCapture(0)
# Inicializando os módulos do MediaPipe para detecção de rosto e desenho
reconhecimento_rosto = mp.solutions.face_detection
desenho = mp.solutions.drawing_utils
reconhecedor_rosto = reconhecimento_rosto.FaceDetection()

# Loop para capturar os frames da webcam
while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break

    # Convertendo a imagem de BGR para RGB, conforme esperado pelo MediaPipe
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processando a imagem para detectar rostos
    lista_rostos = reconhecedor_rosto.process(imagem_rgb)

    # Se algum rosto for detectado, desenhá-lo na imagem original (BGR)
    if lista_rostos and lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    # Exibindo a imagem com os rostos detectados
    cv2.imshow("Rostos na sua webcam", frame)

    # Se a tecla 'Esc' (código 27) for pressionada, o loop é interrompido
    if cv2.waitKey(5) == 27:
        break

# Liberando a webcam e fechando as janelas
webcam.release()
cv2.destroyAllWindows()