import sqlite3
import face_recognition
import numpy as np

# Conectar ao banco de dados (ou criar)
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# Criar a tabela se não existir
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL
)
''')

# Função para adicionar um rosto conhecido ao banco de dados
def add_face_to_db(name, image_path):
    # Carregar a imagem e gerar o embedding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) > 0:
        encoding = encodings[0]
        encoding_blob = sqlite3.Binary(np.array(encoding).tobytes())

        # Inserir o rosto no banco de dados
        cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_blob))
        conn.commit()
        print(f"Rosto de {name} adicionado ao banco de dados.")
    else:
        print("Nenhum rosto encontrado na imagem.")

# Exemplo: Adicionar rostos conhecidos ao banco de dados
add_face_to_db("Pessoa1", "pessoa1.jpg")
add_face_to_db("Pessoa2", "pessoa2.jpg")

# Fechar a conexão quando terminar
conn.close()