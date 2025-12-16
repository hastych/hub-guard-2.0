import cv2
import os
import numpy as np
from datetime import datetime

# Caminhos
path_known_faces = r"C:\Users\gustavo\Desktop\test\imagens_conhecidas"
path_unknown_faces = r"C:\Users\gustavo\Desktop\test\rostos_desconhecidos"

# Inicializar o reconhecedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Listas de rostos conhecidos
known_face_names = []

# Função para carregar rostos conhecidos e treinar o modelo
def load_and_train_known_faces():
    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    for filename in os.listdir(path_known_faces):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(path_known_faces, filename)
            known_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if known_image is None:
                print(f"Erro ao carregar imagem: {filename}")
                continue

            # Redimensionar a imagem para um tamanho padrão
            known_image_resized = cv2.resize(known_image, (150, 150))

            # Adicionar a imagem e o rótulo
            name = os.path.splitext(filename)[0]
            if name not in label_ids:
                label_ids[name] = current_id
                current_id += 1

            faces.append(known_image_resized)
            labels.append(label_ids[name])
            known_face_names.append(name)

    # Treinar o modelo
    recognizer.train(faces, np.array(labels))

# Carregar e treinar o modelo
load_and_train_known_faces()
if not known_face_names:
    print("Nenhuma face conhecida foi carregada. Verifique a pasta de imagens.")
    exit()

# Inicializar vídeo e detector Haar Cascade
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Iniciando reconhecimento facial...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.09, minNeighbors=3, minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_image = gray_frame[y:y+h, x:x+w]
        face_image_resized = cv2.resize(face_image, (150, 150))

        # Reconhecer a face usando LBPH
        label, confidence = recognizer.predict(face_image_resized)
        if confidence < 60:  
            name = f"{known_face_names[label]} ({100 - confidence:.2f}%)"
        else:
            name = "Desconhecido"
            if not os.path.exists(path_unknown_faces):
                os.makedirs(path_unknown_faces)
            img_name = f"{path_unknown_faces}/desconhecido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(img_name, frame[y:y+h, x:x+w])

        # Exibir informações no quadro
        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)

    # Exibe o quadro
    cv2.imshow("Video", frame)

    # Para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
video_capture.release()
cv2.destroyAllWindows()