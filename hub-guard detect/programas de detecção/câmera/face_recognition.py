import cv2
import face_recognition
import sqlite3
import numpy as np

# Carregar rostos conhecidos do banco de dados
known_face_encodings, known_face_names = load_known_faces()

# Inicialize a webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capturar uma única imagem da webcam
    ret, frame = webcam.read()

    # Redimensione a imagem para acelerar o processamento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converta a imagem de BGR (OpenCV) para RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Localize todos os rostos e codifique-os
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Redimensiona as coordenadas para o tamanho original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Verifique se o rosto corresponde a algum rosto conhecido
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"
        color = (0, 0, 255)  # Vermelho para "Desconhecido"

        # Se houver uma correspondência
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            color = (0, 255, 0)  # Verde para "Reconhecido"

        # Desenhe um quadrado ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Desenhe o nome abaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Exiba a imagem com os rostos detectados
    cv2.imshow('Video', frame)

    # Pressione 'Esc' para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libere o controle da webcam e feche as janelas
webcam.release()
cv2.destroyAllWindows()