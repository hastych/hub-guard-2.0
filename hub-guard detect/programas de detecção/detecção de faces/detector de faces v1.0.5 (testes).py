import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Configurações globais
TARGET_SIZE = (160, 160)
DATA_DIR = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"

print(f"✅ OpenCV version: {cv2.__version__}")
print(f"✅ NumPy version: {np.__version__}")

class FaceRecognizer:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def extract_lbp_features(self, image):
        """Extrai características LBP da face"""
        try:
            if image is None or image.size == 0:
                return np.zeros(256, dtype=np.float32)
                
            # Garantir escala de cinza
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Redimensionar
            image = cv2.resize(image, TARGET_SIZE)
            
            # Aplicar LBP
            lbp = np.zeros_like(image, dtype=np.uint8)
            height, width = image.shape
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = image[i, j]
                    code = 0
                    code |= (image[i-1, j-1] >= center) << 7
                    code |= (image[i-1, j] >= center) << 6
                    code |= (image[i-1, j+1] >= center) << 5
                    code |= (image[i, j+1] >= center) << 4
                    code |= (image[i+1, j+1] >= center) << 3
                    code |= (image[i+1, j] >= center) << 2
                    code |= (image[i+1, j-1] >= center) << 1
                    code |= (image[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            # Histograma
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
            return hist
            
        except Exception as e:
            print(f"Erro na extração LBP: {e}")
            return np.zeros(256, dtype=np.float32)
    
    def train(self, images, labels):
        """Treina o modelo de reconhecimento"""
        print("📊 Extraindo características das imagens...")
        
        features = []
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        for i, image in enumerate(images):
            feature = self.extract_lbp_features(image)
            features.append(feature)
            
            if (i + 1) % 50 == 0:
                print(f"   Processadas {i + 1}/{len(images)} imagens")
        
        features = np.array(features)
        
        print("🤖 Treinando modelo KNN...")
        self.model = KNeighborsClassifier(
            n_neighbors=3, 
            weights='distance', 
            metric='euclidean'
        )
        self.model.fit(features, encoded_labels)
        self.is_trained = True
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"👥 Pessoas cadastradas: {len(self.label_encoder.classes_)}")
        print(f"📈 Amostras de treino: {len(features)}")
    
    def predict(self, image):
        """Reconhece uma face"""
        if not self.is_trained:
            return "Modelo não treinado", 0
            
        try:
            features = self.extract_lbp_features(image)
            if np.all(features == 0):
                return "Face inválida", 0
                
            # Fazer predição
            distances, indices = self.model.kneighbors([features])
            confidence = max(0, 100 - distances[0][0] * 3)  # Converter distância para confiança
            predicted_label = self.model.predict([features])[0]
            person = self.label_encoder.inverse_transform([predicted_label])[0]
            
            return person, confidence
            
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            return "Erro", 0

class FaceDetector:
    def __init__(self):
        self.detector = self._initialize_detector()
    
    def _initialize_detector(self):
        """Inicializa o detector de faces Haar Cascade"""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                detector = cv2.CascadeClassifier(cascade_path)
                if not detector.empty():
                    print("✅ Detector Haar Cascade carregado com sucesso!")
                    return detector
        except Exception as e:
            print(f"❌ Erro ao carregar Haar Cascade: {e}")
        
        print("⚠️  Usando detecção básica por contornos")
        return None
    
    def detect_faces(self, image):
        """Detecta faces na imagem"""
        try:
            # Converter para escala de cinza se necessário
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Usar Haar Cascade se disponível
            if self.detector is not None:
                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                return faces
            else:
                # Fallback: detecção básica
                return self._basic_face_detection(gray)
                
        except Exception as e:
            print(f"❌ Erro na detecção: {e}")
            return []
    
    def _basic_face_detection(self, gray_image):
        """Detecção básica baseada em contornos"""
        try:
            # Equalizar histograma para melhor contraste
            equalized = cv2.equalizeHist(gray_image)
            
            # Suavizar e binarizar
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # Filtrar por área (evitar ruídos pequenos e objetos muito grandes)
                if 2000 < area < 30000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    # Filtrar por proporção de rosto
                    if 0.7 < aspect_ratio < 1.4:
                        faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            print(f"❌ Erro na detecção básica: {e}")
            return []

def load_dataset(data_dir):
    """Carrega o dataset de imagens"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Diretório não encontrado: {data_dir}")
    
    images = []
    labels = []
    
    # Listar subdiretórios (cada um é uma pessoa)
    persons = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if not persons:
        raise ValueError(f"❌ Nenhuma pasta de pessoa encontrada em {data_dir}")
    
    print(f"👥 Pessoas encontradas no dataset: {persons}")
    
    for person in persons:
        person_dir = os.path.join(data_dir, person)
        image_count = 0
        
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Converter para escala de cinza
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        images.append(gray)
                        labels.append(person)
                        image_count += 1
                except Exception as e:
                    print(f"⚠️  Erro ao carregar {image_path}: {e}")
        
        print(f"   📁 {person}: {image_count} imagens")
    
    return images, labels

def preprocess_face(face_image):
    """Pré-processa uma imagem de face"""
    try:
        if face_image is None or face_image.size == 0:
            return None
            
        # Redimensionar para tamanho padrão
        face_image = cv2.resize(face_image, TARGET_SIZE)
        
        # Equalização de histograma adaptativa (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(face_image)
        
        # Redução de ruído
        denoised = cv2.fastNlMeansDenoising(equalized, None, h=10)
        
        return denoised
        
    except Exception as e:
        print(f"⚠️  Erro no pré-processamento: {e}")
        return face_image

def main():
    try:
        print("=" * 60)
        print("           SISTEMA DE RECONHECIMENTO FACIAL")
        print("=" * 60)
        
        # 1. Carregar dataset
        print("\n📁 ETAPA 1: CARREGANDO DATASET...")
        images, labels = load_dataset(DATA_DIR)
        
        if not images:
            print("❌ Nenhuma imagem foi carregada. Verifique:")
            print(f"   - O diretório existe: {DATA_DIR}")
            print(f"   - Há subpastas com imagens no diretório")
            print(f"   - As imagens são .jpg, .jpeg ou .png")
            return
        
        print(f"✅ Total de imagens carregadas: {len(images)}")
        print(f"✅ Número de pessoas: {len(set(labels))}")
        
        # 2. Inicializar detector de faces
        print("\n🔍 ETAPA 2: INICIALIZANDO DETECTOR DE FACES...")
        face_detector = FaceDetector()
        
        # 3. Detectar e extrair faces
        print("\n🎯 ETAPA 3: DETECTANDO FACES NAS IMAGENS...")
        face_images = []
        face_labels = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            faces = face_detector.detect_faces(image)
            
            if faces:
                # Usar a maior face detectada
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Garantir coordenadas válidas
                x, y = max(0, x), max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w > 40 and h > 40:  # Tamanho mínimo
                    face_roi = image[y:y+h, x:x+w]
                    processed_face = preprocess_face(face_roi)
                    
                    if processed_face is not None:
                        face_images.append(processed_face)
                        face_labels.append(label)
            
            # Progresso
            if (i + 1) % 20 == 0:
                print(f"   Processadas {i + 1}/{len(images)} imagens")
        
        if not face_images:
            print("❌ Nenhuma face foi detectada nas imagens.")
            print("   Verifique se as imagens contêm rostos visíveis.")
            return
        
        print(f"✅ Faces detectadas e extraídas: {len(face_images)}")
        
        # 4. Data augmentation
        print("\n🔄 ETAPA 4: APLICANDO DATA AUGMENTATION...")
        augmented_faces = []
        augmented_labels = []
        
        for face, label in zip(face_images, face_labels):
            # Imagem original
            augmented_faces.append(face)
            augmented_labels.append(label)
            
            # Flip horizontal (espelhamento)
            flipped = cv2.flip(face, 1)
            augmented_faces.append(flipped)
            augmented_labels.append(label)
        
        print(f"✅ Dataset aumentado: {len(augmented_faces)} imagens")
        
        # 5. Treinar modelo
        print("\n🧠 ETAPA 5: TREINANDO MODELO...")
        recognizer = FaceRecognizer()
        recognizer.train(augmented_faces, augmented_labels)
        
        # 6. Testar com webcam
        print("\n📷 ETAPA 6: INICIALIZANDO WEBCAM...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Não foi possível acessar a webcam")
            return
        
        print("✅ Webcam inicializada com sucesso!")
        print("\n🎮 CONTROLES:")
        print("   Q - Sair do programa")
        print("   S - Salvar frame atual")
        print("   ESPAÇO - Pausar/continuar")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erro ao capturar frame da webcam")
                    break
                
                # Espelhar o frame para comportamento mais natural
                frame = cv2.flip(frame, 1)
                
                # Detectar faces no frame
                faces = face_detector.detect_faces(frame)
                
                for (x, y, w, h) in faces:
                    try:
                        # Extrair região da face
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Converter para escala de cinza e pré-processar
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        processed_face = preprocess_face(gray_face)
                        
                        if processed_face is not None:
                            # Reconhecer a face
                            person, confidence = recognizer.predict(processed_face)
                            
                            # Definir cor e texto baseado na confiança
                            if confidence > 65:
                                color = (0, 255, 0)  # Verde - reconhecido com alta confiança
                                status = f"{person} ({confidence:.1f}%)"
                            elif confidence > 40:
                                color = (0, 255, 255)  # Amarelo - confiança média
                                status = f"{person}? ({confidence:.1f}%)"
                            else:
                                color = (0, 0, 255)  # Vermelho - baixa confiança/desconhecido
                                status = f"Desconhecido ({confidence:.1f}%)"
                            
                            # Desenhar retângulo e texto
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, status, (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                    except Exception as e:
                        print(f"⚠️  Erro no reconhecimento: {e}")
                        continue
            
            # Mostrar frame
            cv2.imshow('Reconhecimento Facial - Pressione Q para sair', frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('captura_frame.png', frame)
                print("📸 Frame salvo como 'captura_frame.png'")
            elif key == ord(' '):
                paused = not paused
                print("⏸️  Pausado" if paused else "▶️  Continuando")
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Programa finalizado com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no programa: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()