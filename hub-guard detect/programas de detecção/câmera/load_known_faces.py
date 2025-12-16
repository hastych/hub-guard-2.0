def load_known_faces():
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, encoding FROM faces")
    known_face_encodings = []
    known_face_names = []
    
    for row in cursor.fetchall():
        name = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float64)
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    
    conn.close()
    return known_face_encodings, known_face_names