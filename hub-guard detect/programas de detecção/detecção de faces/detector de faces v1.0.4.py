import os
import cv2
import dlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configurações globais
TARGET_SIZE = (160, 160)
DATA_DIR = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"

# 1. Paralelização do carregamento de imagens
def load_person_images(person_dir, person_name):
    person_images = []
    person_labels = []
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        try:
            image = cv2.imread(image_path)
            if image is not None:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                person_images.append(gray_image)
                person_labels.append(person_name)
        except Exception as e:
            print(f"[ERRO] Erro ao processar {image_path}: {e}")
    return person_images, person_labels

def load_images_and_labels(data_dir):
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Diretório {data_dir} não encontrado!")
    
    dirs = [(os.path.join(data_dir, p), p) for p in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, p))]
    
    if not dirs:
        raise ValueError(f"Nenhum subdiretório encontrado em {data_dir}")
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_person_images, pd, pn) for pd, pn in dirs]
        for future in tqdm(futures, desc="Carregando imagens"):
            try:
                person_images, person_labels = future.result()
                images.extend(person_images)
                labels.extend(person_labels)
            except Exception as e:
                print(f"Erro ao carregar imagens: {e}")
    
    return images, labels

# 2. Detecção de faces com fallback - CORREÇÃO PRINCIPAL
def initialize_face_detector():
    # Tentar carregar diferentes detectores em ordem de preferência
    detectors_to_try = [
        # 1. Tentar CNN do dlib se o arquivo existir
        ("cnn", "mmod_human_face_detector.dat", 
         lambda path: dlib.cnn_face_detection_model_v1(path)),
        
        # 2. Tentar Haar Cascade do OpenCV (pré-instalado)
        ("haar", cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
         lambda path: cv2.CascadeClassifier(path)),
        
        # 3. Usar detector HOG do dlib (sempre disponível)
        ("hog", None, lambda path: dlib.get_frontal_face_detector())
    ]
    
    for detector_name, detector_path, detector_loader in detectors_to_try:
        try:
            if detector_path is None or os.path.exists(detector_path):
                print(f"Tentando inicializar detector {detector_name.upper()}...")
                detector = detector_loader(detector_path)
                print(f"Detector {detector_name.upper()} inicializado com sucesso!")
                return detector, detector_name
        except Exception as e:
            print(f"Erro ao carregar detector {detector_name}: {e}")
            continue
    
    # Fallback final - usar detector simples baseado em características
    print("Usando detector básico como fallback...")
    return "basic", "basic"

# 3. Pré-processamento aprimorado
def preprocess_image(image):
    if image is None:
        raise ValueError("Imagem inválida para pré-processamento")
    
    # Equalização de histograma adaptativa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image)
    
    # Redução de ruído
    if image.shape[0] > 50 and image.shape[1] > 50:
        denoised = cv2.fastNlMeansDenoising(equalized, None, h=10, 
                                           templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = equalized
    
    # Normalização
    normalized = cv2.normalize(denoised, None, alpha=0, beta=255, 
                             norm_type=cv2.NORM_MINMAX)
    
    return normalized

# 4. Detecção de faces universal - CORRIGIDO
def detect_faces_universal(image, detector, detector_type):
    faces = []
    
    try:
        if detector_type == "cnn":
            # Detector CNN do dlib
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            detected_faces = detector(rgb_image)
            for face in detected_faces:
                x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
                faces.append((x, y, w, h))
                
        elif detector_type == "haar":
            # Detector Haar Cascade
            detected_faces = detector.detectMultiScale(
                image, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            for (x, y, w, h) in detected_faces:
                faces.append((x, y, w, h))
                
        elif detector_type == "hog":
            # Detector HOG do dlib
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            detected_faces = detector(rgb_image, 1)
            for face in detected_faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                faces.append((x, y, w, h))
                
        else:
            # Detector básico - fallback
            # Aplicar blur e threshold para encontrar regiões com características de rosto
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Área mínima
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filtro de proporção aproximada de rosto
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 1.5:  # Proporção aproximada de rosto
                        faces.append((x, y, w, h))
    
    except Exception as e:
        print(f"Erro na detecção de faces: {e}")
    
    return faces

# 5. Detecção e alinhamento de faces
def detect_and_align_faces(images, detector, detector_type):
    aligned_faces = []
    valid_indices = []
    
    for idx, image in enumerate(tqdm(images, desc="Detectando faces")):
        if image is None:
            continue
            
        try:
            # Detectar faces
            faces = detect_faces_universal(image, detector, detector_type)
            
            if len(faces) > 0:
                # Pegar a maior face detectada
                face = max(faces, key=lambda f: f[2] * f[3])  # Maior área (w * h)
                x, y, w, h = face
                
                # Garantir coordenadas válidas
                x, y = max(0, x), max(0, y)
                w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
                
                if w > 20 and h > 20:  # Mínimo tamanho para face
                    face_roi = image[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, TARGET_SIZE)
                    
                    aligned_faces.append(face_roi)
                    valid_indices.append(idx)
                    
        except Exception as e:
            print(f"Erro na detecção da face {idx}: {e}")
            continue
    
    return aligned_faces, valid_indices

# 6. Data Augmentation
def apply_augmentation(images, labels):
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(images, labels):
        if image is None:
            continue
            
        # Original
        augmented_images.append(image)
        augmented_labels.append(label)
        
        try:
            # Flip horizontal
            flipped = cv2.flip(image, 1)
            augmented_images.append(flipped)
            augmented_labels.append(label)
            
        except Exception as e:
            print(f"Erro na augmentação: {e}")
            continue
    
    return augmented_images, augmented_labels

# 7. Treinamento do reconhecedor
def train_lbph_recognizer(images, labels):
    if len(images) == 0:
        raise ValueError("Nenhuma imagem válida para treinamento.")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100)
    
    recognizer.train(images, np.array(labels))
    return recognizer

def main():
    try:
        # 1. Carregar imagens
        print("Carregando dataset...")
        images, labels = load_images_and_labels(DATA_DIR)
        
        if not images:
            print("Nenhuma imagem carregada. Verifique o diretório.")
            return
        
        print(f"Carregadas {len(images)} imagens de {len(set(labels))} pessoas")
        
        # 2. Inicializar detector com fallback
        print("\nInicializando detector de faces...")
        face_detector, detector_type = initialize_face_detector()
        
        # 3. Detectar e alinhar faces
        aligned_faces, valid_indices = detect_and_align_faces(images, face_detector, detector_type)
        labels = [labels[i] for i in valid_indices]
        
        if not aligned_faces:
            print("Nenhuma face detectada. Verifique as imagens.")
            return
        
        print(f"Detectadas {len(aligned_faces)} faces válidas")
        
        # 4. Aplicar data augmentation simples
        print("\nAplicando data augmentation...")
        augmented_faces, augmented_labels = apply_augmentation(aligned_faces, labels)
        
        # 5. Pré-processamento
        print("\nPré-processando imagens...")
        preprocessed_faces = []
        for face in tqdm(augmented_faces):
            try:
                processed = preprocess_image(face)
                preprocessed_faces.append(processed)
            except Exception as e:
                print(f"Erro no pré-processamento: {e}")
                continue
        
        if not preprocessed_faces:
            print("Nenhuma face pré-processada válida.")
            return
        
        # 6. Codificar labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(augmented_labels)
        
        # 7. Treinar modelo
        print("\nTreinando reconhecedor LBPH...")
        recognizer = train_lbph_recognizer(preprocessed_faces, encoded_labels)
        print(f"Modelo treinado com {len(preprocessed_faces)} faces de {len(label_encoder.classes_)} pessoas.")
        
        # 8. Inicializar webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro ao acessar webcam")
            return
        
        print("Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Reconhecimento em tempo real
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar faces
            faces = detect_faces_universal(gray, face_detector, detector_type)
            
            for (x, y, w, h) in faces:
                try:
                    # Garantir coordenadas válidas
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, gray.shape[1] - x), min(h, gray.shape[0] - y)
                    
                    if w > 20 and h > 20:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, TARGET_SIZE)
                        face_roi = preprocess_image(face_roi)
                        
                        # Reconhecer
                        label, confidence = recognizer.predict(face_roi)
                        
                        if confidence < 70:
                            person = label_encoder.inverse_transform([label])[0]
                            color = (0, 255, 0)  # Verde para conhecido
                        else:
                            person = "Desconhecido"
                            color = (0, 0, 255)  # Vermelho para desconhecido
                        
                        # Desenhar resultados
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{person} ({confidence:.2f})", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                except Exception as e:
                    print(f"Erro no reconhecimento: {e}")
                    continue
            
            cv2.imshow("Reconhecimento Facial", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Erro no programa: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()