import os
import cv2
import dlib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Função para carregar as imagens e labels
def load_images_and_labels(data_dir):
    images = []
    labels = []
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    # Carregar a imagem em escala de cinza
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        images.append(image)
                        labels.append(person_name)
                    else:
                        print(f"[ERRO] Não foi possível carregar a imagem: {image_path}")
                except Exception as e:
                    print(f"[ERRO] Erro ao processar a imagem {image_path}: {e}")
    return images, labels

# Função para detectar e alinhar faces
def detect_and_align_faces(images, face_detector, target_size=(150, 150)):
    aligned_faces = []
    valid_indices = []  # Índices das imagens que contêm faces detectadas
    for idx, image in enumerate(images):
        # Detectar faces na imagem
        faces = face_detector(image)
        if len(faces) > 0:
            # Pegar a primeira face detectada
            x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
            face_roi = image[y:y+h, x:x+w]
            # Redimensionar a face para o tamanho desejado
            face_roi = cv2.resize(face_roi, target_size)
            aligned_faces.append(face_roi)
            valid_indices.append(idx)
        else:
            print(f"[AVISO] Nenhuma face detectada na imagem {idx}.")
    return aligned_faces, valid_indices

# Função para aplicar equalização de histograma e suavização
def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Aplicar equalização de histograma
        equalized_image = cv2.equalizeHist(image)
        # Aplicar suavização (blur) para reduzir ruído
        blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
        preprocessed_images.append(blurred_image)
    return preprocessed_images

# Função para treinar o modelo LBPH
def train_lbph_recognizer(images, labels):
    if len(images) == 0:
        raise ValueError("Nenhuma imagem válida foi carregada. Verifique os dados de entrada.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Requer opencv-contrib-python
    recognizer.train(images, np.array(labels))
    return recognizer

# Função para reconhecer faces
def recognize_faces(recognizer, face_detector, label_encoder, frame, confidence_threshold=70):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = gray[y:y+h, x:x+w]
        # Redimensionar e pré-processar a face detectada
        face_roi = cv2.resize(face_roi, (150, 150))
        face_roi = cv2.equalizeHist(face_roi)
        face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
        # Reconhecer a face
        label, confidence = recognizer.predict(face_roi)
        # Se a confiança for menor que o limite, reconhece a pessoa
        if confidence < confidence_threshold:
            person_name = label_encoder.inverse_transform([label])[0]
        else:
            person_name = "Desconhecido"
        # Desenhar retângulo e texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Diretório onde as imagens estão armazenadas
data_dir = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"

# Carregar imagens e labels
images, labels = load_images_and_labels(data_dir)

# Verificar se há imagens carregadas
if len(images) == 0:
    print("Nenhuma imagem válida foi carregada. Verifique o diretório e os arquivos.")
    exit()

# Inicializar o detector de faces do dlib
face_detector = dlib.get_frontal_face_detector()

# Detectar e alinhar faces
aligned_faces, valid_indices = detect_and_align_faces(images, face_detector)

# Filtrar os rótulos para incluir apenas os correspondentes às imagens com faces detectadas
labels = [labels[idx] for idx in valid_indices]

# Verificar se há faces detectadas
if len(aligned_faces) == 0:
    print("Nenhuma face foi detectada nas imagens carregadas. Verifique as imagens.")
    exit()

# Pré-processar as imagens (equalização de histograma e suavização)
aligned_faces = preprocess_images(aligned_faces)

# Codificar labels para números
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Treinar o reconhecedor LBPH
try:
    recognizer = train_lbph_recognizer(aligned_faces, labels_encoded)
    print(f"Modelo treinado com {len(aligned_faces)} imagens.")
except ValueError as e:
    print(e)
    exit()

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reconhecer faces no frame
    frame = recognize_faces(recognizer, face_detector, label_encoder, frame)

    # Mostrar o frame
    cv2.imshow("Reconhecimento Facial", frame)

    # Parar o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()