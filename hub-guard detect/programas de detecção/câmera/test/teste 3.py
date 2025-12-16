import cv2
import os
import numpy as np
from datetime import datetime

# Caminhos das pastas
path_known_faces = r"C:\Users\gustavo\Desktop\test\imagens_conhecidas"
path_unknown_faces = r"C:\Users\gustavo\Desktop\test\rostos_desconhecidos"

# Inicialização
known_face_encodings = []
known_face_names = []

# Função para carregar e processar imagens conhecidas
def load_known_faces():
    for filename in os.listdir(path_known_faces):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(path_known_faces, filename)
            known_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if known_image is None:
                print(f"Erro ao carregar imagem: {filename}")
                continue

            # Normaliza e redimensiona
            known_image_resized = cv2.resize(known_image, (150, 150))
            known_face_encodings.append(known_image_resized)
            known_face_names.append(os.path.splitext(filename)[0])

# Função para comparar rostos
def compare_faces(known_encodings, face_image, threshold=80):
    similarities = []
    face_image_resized = cv2.resize(face_image, (150, 150))

    for known_encoding in known_encodings:
        diff = cv2.absdiff(known_encoding, face_image_resized)
        similarity = np.mean(diff)
        similarities.append(similarity)

    if similarities:
        min_similarity = min(similarities)
        match_index = similarities.index(min_similarity)
        return min_similarity < threshold, match_index
    return False, None

# Carregar rostos conhecidos
load_known_faces()
if not known_face_encodings:
    print("Nenhuma face conhecida foi carregada. Verifique a pasta de imagens.")
    exit()

# Captura de vídeo
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
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_image = gray_frame[y:y+h, x:x+w]

        # Comparar rosto detectado
        match, match_index = compare_faces(known_face_encodings, face_image)

        if match:
            name = known_face_names[match_index]
        else:
            name = "Desconhecido"
            if not os.path.exists(path_unknown_faces):
                os.makedirs(path_unknown_faces)
            img_name = f"{path_unknown_faces}/desconhecido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(img_name, frame[y:y+h, x:x+w])

        # Desenhar retângulo e nome
        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)

    # Mostrar vídeo
    cv2.imshow("Video", frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
