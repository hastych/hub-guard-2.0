import cv2
import os
import numpy as np
from datetime import datetime

# Caminho para imagens conhecidas e onde salvar rostos desconhecidos
path_known_faces = r"C:\Users\gustavo\Desktop\test\imagens_conhecidas"
path_unknown_faces = r"C:\Users\gustavo\Desktop\test\rostos_desconhecidos"

# Inicializar arrays para codificações e nomes conhecidos
known_face_encodings = []
known_face_names = []

# Carregar as imagens conhecidas e aplicar variações de iluminação
for filename in os.listdir(path_known_faces):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(path_known_faces, filename)
        known_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Escala de cinza
        known_image = cv2.resize(known_image, (150, 150))  # Redimensiona para 150x150

        # Adicionar variações de iluminação
        bright_image = cv2.convertScaleAbs(known_image, alpha=1.2, beta=50)  # Clareada
        dark_image = cv2.convertScaleAbs(known_image, alpha=0.8, beta=-30)  # Escurecida

        known_face_encodings.extend([known_image, bright_image, dark_image])
        known_face_names.extend([os.path.splitext(filename)[0]] * 3)

# Função para comparar rostos usando LBPH
def compare_faces(known_encodings, face_image, threshold=50):
    face_image_resized = cv2.resize(face_image, (150, 150))

    # Inicializar o LBPH face recognizer
    lbph = cv2.face.LBPHFaceRecognizer_create()
    similarities = []

    for idx, known_encoding in enumerate(known_encodings):
        lbph.train([known_encoding], np.array([idx]))
        _, similarity = lbph.predict(face_image_resized)
        similarities.append(similarity)

    min_similarity = min(similarities)
    match_index = similarities.index(min_similarity)

    return min_similarity < threshold, match_index

# Iniciar a captura de vídeo
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Converter para escala de cinza e normalizar iluminação
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)

    # Detecção de rostos
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_image = gray_frame[y:y+h, x:x+w]  # Extrair rosto detectado

        # Comparar o rosto detectado com os conhecidos
        match, match_index = compare_faces(known_face_encodings, face_image)

        if match:
            name = known_face_names[match_index]
        else:
            name = "Desconhecido"

        # Desenhar retângulo e escrever nome
        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)

        # Salvar rostos desconhecidos
        if name == "Desconhecido":
            if not os.path.exists(path_unknown_faces):
                os.makedirs(path_unknown_faces)
            img_name = f"{path_unknown_faces}/desconhecido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(img_name, face_image)

    # Mostrar o vídeo com os resultados
    cv2.imshow('Video', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o vídeo e fechar as janelas
video_capture.release()
cv2.destroyAllWindows()
