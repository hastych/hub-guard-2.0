import os
import cv2
import dlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

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
def detect_and_align_faces(images, face_detector, target_size=(250, 250)):
    aligned_faces = []
    valid_indices = []
    for idx, image in enumerate(images):
        faces = face_detector(image)
        if len(faces) > 0:
            x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
            face_roi = image[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, target_size)
            aligned_faces.append(face_roi)
            valid_indices.append(idx)
        else:
            print(f"[AVISO] Nenhuma face detectada na imagem {idx}.")
    return aligned_faces, valid_indices

# Função para pré-processamento
def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        equalized_image = cv2.equalizeHist(image)
        blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
        preprocessed_images.append(blurred_image)
    return preprocessed_images

# Função para treinar o modelo LBPH
def train_lbph_recognizer(images, labels):
    if len(images) == 0:
        raise ValueError("Nenhuma imagem válida foi carregada. Verifique os dados de entrada.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer

# Função para detectar fogo/chamas
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Intervalos de cor para fogo (ajuste conforme necessário)
    lower_fire = np.array([0, 120, 70])
    upper_fire = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fire_rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:  # Área mínima para considerar como fogo
            x, y, w, h = cv2.boundingRect(cnt)
            fire_rects.append((x, y, w, h))
    
    return fire_rects

# Função principal que reconhece faces e detecta fogo
def recognize_faces_and_fire(recognizer, face_detector, label_encoder, frame, confidence_threshold=70):
    # Detecção de faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (150, 150))
        face_roi = cv2.equalizeHist(face_roi)
        face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
        
        label, confidence = recognizer.predict(face_roi)
        person_name = label_encoder.inverse_transform([label])[0] if confidence < confidence_threshold else "Desconhecido"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detecção de fogo
    fire_rects = detect_fire(frame)
    for (x, y, w, h) in fire_rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(frame, "PERIGO! FOGO DETECTADO", (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return frame

# Configurações principais
data_dir = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"
fire_dir = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\perigo\fogo"

# Verificar/Criar diretório de fogo
os.makedirs(fire_dir, exist_ok=True)

# Inicializar sistema de reconhecimento facial
print("Carregando imagens de treinamento facial...")
images, labels = load_images_and_labels(data_dir)
if len(images) == 0:
    print("Nenhuma imagem válida encontrada. Verifique o diretório:", data_dir)
    exit()

print("Inicializando detector de faces...")
face_detector = dlib.get_frontal_face_detector()

print("Preparando imagens para treinamento...")
aligned_faces, valid_indices = detect_and_align_faces(images, face_detector)
labels = [labels[idx] for idx in valid_indices]

if len(aligned_faces) == 0:
    print("Nenhuma face detectada nas imagens de treino. Verifique a qualidade das imagens.")
    exit()

print("Pré-processando imagens...")
aligned_faces = preprocess_images(aligned_faces)

print("Codificando labels...")
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

print("Treinando modelo LBPH...")
try:
    recognizer = train_lbph_recognizer(aligned_faces, labels_encoded)
    print(f"Modelo treinado com {len(aligned_faces)} imagens de faces.")
except ValueError as e:
    print("Erro no treinamento:", e)
    exit()

# Verificar imagens de fogo de referência
fire_images = [f for f in os.listdir(fire_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Encontradas {len(fire_images)} imagens de fogo para referência em {fire_dir}")

# Inicializar webcam
print("Iniciando câmera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Sistema pronto. Pressione 'q' para sair ou 's' para salvar detecções de fogo.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    frame = cv2.resize(frame, (800, 600))
    frame = recognize_faces_and_fire(recognizer, face_detector, label_encoder, frame)
    
    cv2.imshow("Sistema de Segurança - Detecção Facial e de Incêndio", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(fire_dir, f"fogo_detectado_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Imagem salva: {filename}")

cap.release()
cv2.destroyAllWindows()
print("Sistema encerrado.")