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
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    labels.append(person_name)
    return images, labels

# Função para treinar o modelo LBPH
def train_lbph_recognizer(images, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Requer opencv-contrib-python
    recognizer.train(images, np.array(labels))
    return recognizer

# Função para reconhecer faces
def recognize_faces(recognizer, face_detector, label_encoder, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)
        person_name = label_encoder.inverse_transform([label])[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Diretório onde as imagens estão armazenadas
data_dir = r"C:\Users\Pichau\Desktop\testes hub-guard\hub-guard detect\faces"

# Carregar imagens e labels
images, labels = load_images_and_labels(data_dir)

# Codificar labels para números
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Treinar o reconhecedor LBPH
recognizer = train_lbph_recognizer(images, labels_encoded)

# Inicializar o detector de faces do dlib
face_detector = dlib.get_frontal_face_detector()

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