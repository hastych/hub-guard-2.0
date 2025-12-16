import os
import cv2
import dlib
import numpy as np
from sklearn.preprocessing import LabelEncoder

class SafetyDetector:
    def __init__(self):
        # Configurações ajustáveis
        self.FACE_DIR = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"
        self.FIRE_DIR = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\perigo\fogo"
        self.MIN_FACE_SIZE = 100  # Tamanho mínimo da face em pixels
        self.FIRE_THRESHOLD = 50   # Sensibilidade ao fogo (0-100)
        
        # Inicializar detectores
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_encoder = LabelEncoder()
        
        # Carregar modelos
        self.load_models()

    def load_models(self):
        """Carrega modelos de reconhecimento facial"""
        try:
            # Verificar se existem dados de treinamento
            if not os.path.exists(self.FACE_DIR) or not os.listdir(self.FACE_DIR):
                print("AVISO: Pasta de treinamento facial vazia ou não encontrada")
                return False

            # Carregar imagens de treino
            images, labels = [], []
            for person in os.listdir(self.FACE_DIR):
                person_dir = os.path.join(self.FACE_DIR, person)
                if os.path.isdir(person_dir):
                    for img_file in os.listdir(person_dir):
                        img_path = os.path.join(person_dir, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            images.append(img)
                            labels.append(person)

            if not images:
                print("ERRO: Nenhuma imagem válida encontrada")
                return False

            # Treinar reconhecedor
            self.label_encoder.fit(labels)
            self.face_recognizer.train(images, np.array(self.label_encoder.transform(labels)))
            print(f"Modelo treinado com {len(images)} imagens")
            return True
            
        except Exception as e:
            print(f"Erro no treinamento: {str(e)}")
            return False

    def detect_faces(self, frame):
        """Detecta e reconhece faces com debug visual"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Verificar tamanho mínimo
            if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                cv2.putText(frame, "FACE PEQUENA", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                continue
                
            # Reconhecimento
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (150, 150))
            label, confidence = self.face_recognizer.predict(face_roi)
            
            if confidence < 70:  # Limiar de confiança
                name = self.label_encoder.inverse_transform([label])[0]
                color = (0, 255, 0)
            else:
                name = "Desconhecido"
                color = (0, 0, 255)
            
            # Desenhar resultados
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.1f})", (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, len(faces) > 0

    def detect_fire(self, frame):
        """Detecta fogo com múltiplas técnicas e debug visual"""
        # 1. Detecção por cor
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 100, 100])
        upper = np.array([20, 255, 255])
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # 2. Detecção de movimento (simplificada)
        if hasattr(self, 'prev_frame'):
            diff = cv2.absdiff(cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        else:
            motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.prev_frame = frame.copy()
        
        # Combinar máscaras
        combined = cv2.bitwise_and(color_mask, motion_mask)
        
        # Processar resultados
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fire_detected = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Área mínima
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, "FOGO DETECTADO", (x, y-15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                fire_detected = True
        
        # Debug visual
        debug_frame = np.hstack([
            frame,
            cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        ])
        
        return debug_frame if fire_detected else frame, fire_detected

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Processar frame
            frame = cv2.resize(frame, (640, 480))
            
            # Detecções
            face_frame, has_face = self.detect_faces(frame.copy())
            fire_frame, has_fire = self.detect_fire(frame.copy())
            
            # Exibir resultados
            if has_fire:
                cv2.imshow('Debug Fogo', fire_frame)
            else:
                cv2.imshow('Reconhecimento Facial', face_frame)
            
            # Controles
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('t'):
                print("Testando parâmetros...")
                self.adjust_parameters()
        
        cap.release()
        cv2.destroyAllWindows()

    def adjust_parameters(self):
        """Ajusta parâmetros dinamicamente"""
        self.FIRE_THRESHOLD = min(100, self.FIRE_THRESHOLD + 5)
        print(f"Nova sensibilidade ao fogo: {self.FIRE_THRESHOLD}")

if __name__ == "__main__":
    detector = SafetyDetector()
    detector.run()