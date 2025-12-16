import dlib
import cv2

# Caminho da imagem
caminho_imagem = r'C:\Users\Pichau\Desktop\testes hub-guard\hub-guard detect\testes de biblioteca\imagem para testes\pessoa 1.jpg'

# Carrega o detector de rostos do dlib
detector = dlib.get_frontal_face_detector()

# Carrega a imagem
image = cv2.imread(caminho_imagem)

# Verifica se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecta rostos na imagem
    faces = detector(gray)

    # Desenha retângulos ao redor dos rostos detectados
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe a imagem com os rostos detectados
    cv2.imshow('Rostos Detectados', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()