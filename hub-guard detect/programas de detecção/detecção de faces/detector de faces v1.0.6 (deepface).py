import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import warnings

# Configurações para melhor performance
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurações globais
TARGET_SIZE = (160, 160)
DATA_DIR = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"

print("✅ DeepFace e TensorFlow carregados com sucesso!")

class FaceRecognizer:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.embeddings = []
        self.labels = []
        
    def extract_embeddings(self, image):
        """Extrai embeddings faciais usando DeepFace"""
        try:
            if image is None or image.size == 0:
                return None
                
            # Converter para RGB
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Usar Facenet (bom equilíbrio entre velocidade e precisão)
            result = DeepFace.represent(
                img_path=rgb_image,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )
            
            if result:
                embedding = result[0]['embedding']
                return np.array(embedding, dtype=np.float32)
            return None
                
        except Exception as e:
            print(f"❌ Erro na extração DeepFace: {e}")
            return None
    
    def train(self, images, labels):
        """Treina o modelo de reconhecimento"""
        print("📊 Extraindo embeddings com DeepFace...")
        
        self.embeddings = []
        self.labels = []
        
        successful = 0
        for i, (image, label) in enumerate(zip(images, labels)):
            embedding = self.extract_embeddings(image)
            
            if embedding is not None:
                self.embeddings.append(embedding)
                self.labels.append(label)
                successful += 1
            
            if (i + 1) % 10 == 0:
                print(f"   ✅ {i + 1}/{len(images)} imagens processadas ({successful} embeddings)")
        
        if not self.embeddings:
            raise ValueError("❌ Nenhum embedding válido extraído")
        
        # Codificar labels e treinar
        encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.embeddings = np.array(self.embeddings)
        
        print("🤖 Treinando classificador KNN...")
        self.model = KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            metric='cosine'  # Melhor para embeddings
        )
        self.model.fit(self.embeddings, encoded_labels)
        self.is_trained = True
        
        print(f"🎯 Modelo treinado com sucesso!")
        print(f"👥 Pessoas cadastradas: {list(self.label_encoder.classes_)}")
        print(f"📈 Total de embeddings: {len(self.embeddings)}")
        print(f"🔢 Dimensão dos embeddings: {self.embeddings[0].shape}")
    
    def predict(self, image):
        """Reconhece uma face"""
        if not self.is_trained:
            return "Modelo não treinado", 0
            
        try:
            embedding = self.extract_embeddings(image)
            if embedding is None:
                return "Face não detectada", 0
            
            # Calcular distância e confiança
            distances, indices = self.model.kneighbors([embedding])
            confidence = max(0, 100 - distances[0][0] * 100)
            
            # Fazer predição
            predicted_label = self.model.predict([embedding])[0]
            person = self.label_encoder.inverse_transform([predicted_label])[0]
            
            return person, min(confidence, 95)  # Limitar a 95% máximo
            
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            return "Erro", 0

def load_dataset(data_dir):
    """Carrega e valida o dataset"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Diretório não encontrado: {data_dir}")
    
    images = []
    labels = []
    
    persons = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if not persons:
        raise ValueError("❌ Nenhuma pasta de pessoa encontrada")
    
    print(f"👥 Pessoas encontradas: {persons}")
    
    for person in persons:
        person_dir = os.path.join(data_dir, person)
        image_count = 0
        
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is not None and image.size > 0:
                        # Redimensionar para consistência
                        image = cv2.resize(image, (640, 480))
                        images.append(image)
                        labels.append(person)
                        image_count += 1
                except Exception:
                    continue  # Ignorar imagens com erro
        
        print(f"   📁 {person}: {image_count} imagens")
    
    if not images:
        raise ValueError("❌ Nenhuma imagem válida carregada")
    
    return images, labels

def detect_faces(image):
    """Detecta faces usando OpenCV com múltiplos classificadores"""
    try:
        # Tentar diferentes classificadores para melhor detecção
        cascades = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        ]
        
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        all_faces = []
        
        for cascade_path in cascades:
            try:
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if face_cascade.empty():
                    continue
                
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )
                
                if len(faces) > 0:
                    all_faces.extend(faces)
            except Exception:
                continue
        
        # Retornar a maior face detectada
        if len(all_faces) > 0:
            return [max(all_faces, key=lambda f: f[2] * f[3])]
        
        return []
        
    except Exception as e:
        print(f"❌ Erro na detecção: {e}")
        return []

def enhance_face(face_image):
    """Melhora a qualidade da imagem da face"""
    try:
        if face_image is None or face_image.size == 0:
            return None
            
        # Redimensionar para tamanho padrão
        face_image = cv2.resize(face_image, TARGET_SIZE)
        
        # Melhorar contraste com CLAHE
        if len(face_image.shape) > 2:
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_image = clahe.apply(face_image)
        
        # Reduzir ruído suavemente
        face_image = cv2.medianBlur(face_image, 3)
        
        return face_image
        
    except Exception as e:
        print(f"⚠️  Erro no enhancement: {e}")
        return face_image

def main():
    try:
        print("=" * 60)
        print("           SISTEMA DE RECONHECIMENTO FACIAL COM DEEPFACE")
        print("=" * 60)
        
        # 1. Carregar dataset
        print("\n📁 ETAPA 1: CARREGANDO DATASET...")
        images, labels = load_dataset(DATA_DIR)
        
        print(f"✅ Total de imagens: {len(images)}")
        print(f"✅ Pessoas únicas: {len(set(labels))}")
        
        # 2. Extrair faces do dataset
        print("\n🎯 ETAPA 2: EXTRAINDO FACES DAS IMAGENS...")
        face_images = []
        face_labels = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            faces = detect_faces(image)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Usar a maior face
                
                # Garantir coordenadas válidas
                x, y = max(0, x), max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w > 40 and h > 40:  # Tamanho mínimo
                    face_roi = image[y:y+h, x:x+w]
                    enhanced_face = enhance_face(face_roi)
                    
                    if enhanced_face is not None:
                        face_images.append(enhanced_face)
                        face_labels.append(label)
            
            if (i + 1) % 20 == 0:
                print(f"   Processadas {i + 1}/{len(images)} imagens")
        
        if not face_images:
            print("❌ Nenhuma face detectada no dataset")
            return
        
        print(f"✅ Faces extraídas: {len(face_images)}")
        
        # 3. Data augmentation para melhor generalização
        print("\n🔄 ETAPA 3: APLICANDO DATA AUGMENTATION...")
        augmented_faces = []
        augmented_labels = []
        
        for face, label in zip(face_images, face_labels):
            # Imagem original
            augmented_faces.append(face)
            augmented_labels.append(label)
            
            # Flip horizontal
            flipped = cv2.flip(face, 1)
            augmented_faces.append(flipped)
            augmented_labels.append(label)
            
            # Versão com brilho ajustado
            bright = cv2.convertScaleAbs(face, alpha=1.1, beta=5)
            augmented_faces.append(bright)
            augmented_labels.append(label)
        
        print(f"✅ Dataset aumentado: {len(augmented_faces)} imagens")
        
        # 4. Treinar modelo DeepFace
        print("\n🧠 ETAPA 4: TREINANDO MODELO DEEPFACE...")
        recognizer = FaceRecognizer()
        recognizer.train(augmented_faces, augmented_labels)
        
        # 5. Reconhecimento em tempo real com webcam
        print("\n📷 ETAPA 5: INICIALIZANDO WEBCAM...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Não foi possível acessar a webcam")
            return
        
        print("✅ Webcam inicializada!")
        print("\n🎮 CONTROLES:")
        print("   Q - Sair do programa")
        print("   S - Salvar frame atual")
        print("   ESPAÇO - Pausar/continuar")
        print("   R - Limpar histórico de confiança")
        
        paused = False
        confidence_history = []
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Espelhar frame para comportamento natural
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                
                # Detectar faces
                faces = detect_faces(frame)
                
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        try:
                            # Extrair região da face
                            face_roi = frame[y:y+h, x:x+w]
                            enhanced_face = enhance_face(face_roi)
                            
                            if enhanced_face is not None:
                                # Reconhecer face
                                person, confidence = recognizer.predict(enhanced_face)
                                
                                # Estabilizar confiança com média móvel
                                confidence_history.append(confidence)
                                if len(confidence_history) > 5:
                                    confidence_history.pop(0)
                                avg_confidence = np.mean(confidence_history)
                                
                                # Definir cor baseada na confiança
                                if avg_confidence > 75:
                                    color = (0, 255, 0)  # Verde - Alta confiança
                                    status = f"{person} ({avg_confidence:.1f}%)"
                                elif avg_confidence > 50:
                                    color = (0, 255, 255)  # Amarelo - Confiança média
                                    status = f"{person}? ({avg_confidence:.1f}%)"
                                else:
                                    color = (0, 0, 255)  # Vermelho - Baixa confiança
                                    status = f"Desconhecido ({avg_confidence:.1f}%)"
                                
                                # Desenhar na imagem
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                                cv2.putText(display_frame, status, (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                        except Exception as e:
                            print(f"⚠️  Erro no reconhecimento: {e}")
                            continue
                
                # Adicionar informações na tela
                cv2.putText(display_frame, "DeepFace - Reconhecimento Facial", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Faces detectadas: {len(faces)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostrar frame
                cv2.imshow('Reconhecimento Facial DeepFace | Q para sair', display_frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('captura_deepface.png', display_frame)
                print("📸 Frame salvo como 'captura_deepface.png'")
            elif key == ord(' '):
                paused = not paused
                print("⏸️  Pausado" if paused else "▶️  Continuando")
            elif key == ord('r'):
                confidence_history = []
                print("🔄 Histórico de confiança limpo")
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("\n🎉 Programa finalizado com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no programa: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()