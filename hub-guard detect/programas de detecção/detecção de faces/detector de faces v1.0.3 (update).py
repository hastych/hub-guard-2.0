import os
import cv2
import dlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Para barra de progresso

# Configurações globais
TARGET_SIZE = (160, 160)  # Tamanho compatível com a maioria das CNNs
DATA_DIR = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"
DETECTOR_CNN = "mmod_human_face_detector.dat"  # Modelo CNN do dlib

# 1. Paralelização do carregamento de imagens
def load_person_images(person_dir, person_name):
    person_images = []
    person_labels = []
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                person_images.append(image)
                person_labels.append(person_name)
        except Exception as e:
            print(f"[ERRO] Erro ao processar {image_path}: {e}")
    return person_images, person_labels

def load_images_and_labels(data_dir):
    images = []
    labels = []
    dirs = [(os.path.join(data_dir, p), p) for p in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, p))]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_person_images, pd, pn) for pd, pn in dirs]
        for future in tqdm(futures, desc="Carregando imagens"):
            person_images, person_labels = future.result()
            images.extend(person_images)
            labels.extend(person_labels)
    
    return images, labels

# 2. Detecção de faces mais robusta com CNN do dlib
def initialize_face_detector():
    if not os.path.exists(DETECTOR_CNN):
        raise FileNotFoundError(f"Modelo CNN do dlib não encontrado em {DETECTOR_CNN}")
    return dlib.cnn_face_detection_model_v1(DETECTOR_CNN)

# 3. Pré-processamento aprimorado
def preprocess_image(image):
    # Equalização de histograma adaptativa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image)
    
    # Redução de ruído não-local means
    denoised = cv2.fastNlMeansDenoising(equalized, None, h=10, 
                                       templateWindowSize=7, searchWindowSize=21)
    
    # Normalização
    normalized = cv2.normalize(denoised, None, alpha=0, beta=255, 
                             norm_type=cv2.NORM_MINMAX)
    
    return normalized

# 4. Data Augmentation
def apply_augmentation(images, labels):
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(images, labels):
        # Original
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Flip horizontal
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        augmented_labels.append(label)
        
        # Pequenas rotações
        for angle in [-15, 15]:
            center = (image.shape[1]//2, image.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                                   borderMode=cv2.BORDER_REPLICATE)
            augmented_images.append(rotated)
            augmented_labels.append(label)
        
        # Brilho ajustado
        for beta in [-30, 30]:
            adjusted = cv2.add(image, beta)
            augmented_images.append(adjusted)
            augmented_labels.append(label)
    
    return augmented_images, augmented_labels

def detect_and_align_faces(images, face_detector):
    aligned_faces = []
    valid_indices = []
    
    for idx, image in enumerate(tqdm(images, desc="Detectando faces")):
        # Converter para colorido (necessário para o detector CNN)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Detectar faces com CNN
        faces = face_detector(rgb_image)
        
        if len(faces) > 0:
            # Pegar a maior face detectada
            face = max(faces, key=lambda f: f.rect.width() * f.rect.height())
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            
            # Extrair ROI e redimensionar
            face_roi = image[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, TARGET_SIZE)
            
            aligned_faces.append(face_roi)
            valid_indices.append(idx)
    
    return aligned_faces, valid_indices

def train_lbph_recognizer(images, labels):
    if len(images) == 0:
        raise ValueError("Nenhuma imagem válida para treinamento.")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=16, grid_x=8, grid_y=8, threshold=100)
    
    recognizer.train(images, np.array(labels))
    return recognizer

def main():
    # 1. Carregar imagens com paralelização
    print("Carregando dataset...")
    images, labels = load_images_and_labels(DATA_DIR)
    
    if not images:
        print("Nenhuma imagem carregada. Verifique o diretório.")
        return
    
    # 2. Inicializar detector CNN
    print("\nInicializando detector de faces CNN...")
    face_detector = initialize_face_detector()
    
    # 3. Detectar e alinhar faces
    aligned_faces, valid_indices = detect_and_align_faces(images, face_detector)
    labels = [labels[i] for i in valid_indices]
    
    if not aligned_faces:
        print("Nenhuma face detectada. Verifique as imagens.")
        return
    
    # 4. Aplicar data augmentation
    print("\nAplicando data augmentation...")
    augmented_faces, augmented_labels = apply_augmentation(aligned_faces, labels)
    
    # 5. Pré-processamento aprimorado
    print("\nPré-processando imagens...")
    preprocessed_faces = [preprocess_image(face) for face in tqdm(augmented_faces)]
    
    # 6. Codificar labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(augmented_labels)
    
    # 7. Treinar modelo
    print("\nTreinando reconhecedor LBPH...")
    recognizer = train_lbph_recognizer(preprocessed_faces, encoded_labels)
    print(f"Modelo treinado com {len(preprocessed_faces)} faces de {len(label_encoder.classes_)} pessoas.")
    
    # 8. Inicializar webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Reconhecimento em tempo real
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar faces com CNN
        faces = face_detector(rgb_frame)
        
        for face in faces:
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            face_roi = gray[y:y+h, x:x+w]
            
            # Pré-processamento igual ao treinamento
            face_roi = cv2.resize(face_roi, TARGET_SIZE)
            face_roi = preprocess_image(face_roi)
            
            # Reconhecer
            label, confidence = recognizer.predict(face_roi)
            
            if confidence < 70:  # Threshold de confiança
                person = label_encoder.inverse_transform([label])[0]
            else:
                person = "Desconhecido"
            
            # Desenhar resultados
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{person} ({confidence:.2f})", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Reconhecimento Facial Otimizado", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()