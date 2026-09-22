import sqlite3 as sql
import numpy as np
from config import DB_PATH, EYE_DISTANCE_THRESHOLD, FACE_DISTANCE_THRESHOLD

#If there's no database called facedata, create a database named facedata.
#This database contains 4 columns, which are id, name, face_encoding and eye_encoding respectively.
def init_db(db_path=DB_PATH):
    conn = sql.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS facedata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            face_encoding BLOB NOT NULL,
            eye_encoding BLOB NOT NULL
        )
    """)

    conn.commit()
    conn.close()

#Numpy vector to bytes, to store in SQLite BLOB, since BLOB only accept string.
#convert vector to np.float32 to imporve processing speed.
def encode_vector(vec) -> bytes:
    vec = vec.astype(np.float32)
    return vec.tobytes()

#SQLite BLOB to numpy vector
def decode_vector(blob) -> np.ndarray:
    return np.frombuffer(blob,dtype=np.float32)

#Add face in database.
def add_face(name:str,face_encoding,eye_encoding,db_path=DB_PATH):
    conn = sql.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO facedata(name,face_encoding,eye_encoding)VALUES(?,?,?)",
            (name,encode_vector(face_encoding),encode_vector(eye_encoding)),
        )
        conn.commit()

    except sql.IntegrityError:
        print(f"Name'{name}' already exists.")
        return False
    
    conn.close()

"""
Face data in database is first stored in rows, as list of tuples.
Then, through a loop, all face data in the row is loaded into a list named face_data.
Face_data list contains several dictionaries.
"""
def load_faces(db_path=DB_PATH) -> list:
    conn = sql.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, face_encoding, eye_encoding FROM facedata",
    )
    rows = cur.fetchall()
    conn.close()

    face_data=[]

    for row in rows:
        id_ = row[0]
        name_ = row[1]
        face_encoding_ = decode_vector(row[2])
        eye_encoding_ = decode_vector(row[3])

        face_data.append({
            "id":id_,
            "name":name_,
            "face_encoding":face_encoding_,
            "eye_encoding":eye_encoding_,
        })

    return face_data

"""
1. Compare all eye_distance between eye_encoding(ndarray) and eye_encoding_db(ndarray);
2. Find the eye_distance smaller than EYE_DISTANCE_THRESHOLD, and generate a candidates list;
3. Compare the face_encoding with every face in the candidates list, and find the most similar one within faces that smaller than FACE_DISTANCE_THRESHOLD;
4. Return name or None.
"""
def match_face(eye_encoding, face_encoding,db_path=DB_PATH) :
    face_data=load_faces(db_path)
    if face_data is None:
        face_data = []

    if len(face_data) == 0:
        return None

    #First phase: match eyes
    eye_encoding_list = []
    #First, extract the values corresponding to the key "eye_encoding" from face_data.
    #Second, Add the extracted values to a list named eye_encoding_list and iterate through the process (for all relevant entries).
    #Third, Convert this list into an ndarray.
    for d in face_data:
        eye_enc = d["eye_encoding"]
        eye_encoding_list.append(eye_enc)
    eye_encoding_db = np.array(eye_encoding_list)

    #Calculate the Euclidean Distance between face data in database and the face data to be recognised.
    eye_distance = np.linalg.norm(eye_encoding_db - eye_encoding, axis = 1)

    candidates = []
    for i in range(len(eye_distance)):
        d = eye_distance[i]
        if d <= EYE_DISTANCE_THRESHOLD:
            candidates.append(i)

    if len(candidates) == 0:
        return None
    
    #Second phase: match whole faces
    
    best_name = None
    best_distance = 1000
        
    for j in candidates:
        current_face = face_data[j]["face_encoding"]
        dist = np.linalg.norm(current_face - face_encoding)
        if dist < best_distance:
            best_distance = dist
            best_name = face_data[j]["name"]

    if best_distance < FACE_DISTANCE_THRESHOLD:
        return best_name
    else:
        return None