import cv2
from deepface import DeepFace
import os

# Caminho para a pasta com os rostos cadastrados
path_faces = r"C:\Users\gustavo\Desktop\testes hub-guard\hub-guard detect\faces"

# Inicializar webcam
cap = cv2.VideoCapture(0)

print("[INFO] Carregando faces conhecidas...")

# Carrega todas as imagens cadastradas
db = DeepFace.find(
    img_path=os.path.join(path_faces, os.listdir(path_faces)[0], os.listdir(os.path.join(path_faces, os.listdir(path_faces)[0]))[0]),
    db_path=path_faces,
    enforce_detection=False,
    detector_backend='opencv',
    model_name='VGG-Face',
    silent=True
)
# Isso carrega o modelo e cria cache – evita demora depois

print("[INFO] Sistema pronto. Iniciando webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Detecta rostos no frame com o backend do OpenCV
        faces = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)
        
        for face in faces:
            x, y, w, h = face["facial_area"].values()
            cropped_face = frame[y:y+h, x:x+w]

            try:
                result = DeepFace.find(
                    img_path=cropped_face,
                    db_path=path_faces,
                    model_name='VGG-Face',
                    detector_backend='opencv',
                    enforce_detection=False,
                    silent=True
                )

                if result[0].shape[0] > 0:
                    # Extrai o nome da pasta do caminho do arquivo reconhecido
                    identity_path = result[0]['identity'].values[0]
                    name = os.path.basename(os.path.dirname(identity_path))
                    color = (0, 255, 0)  # Verde
                else:
                    name = "Desconhecido"
                    color = (0, 0, 255)  # Vermelho

            except:
                name = "Desconhecido"
                color = (0, 0, 255)  # Vermelho

            # Desenha retângulo e nome
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    except Exception as e:
        print("[ERRO]", e)

    # Exibe o frame
    cv2.imshow("Reconhecimento Facial - Hub Guard", frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finaliza
cap.release()
cv2.destroyAllWindows()
