import os
import face_recognition
import pickle

KNOWN_FACES_DIR = 'known_faces'
ENCODINGS_FILE = 'known_faces_encodings.pkl'

def encode_known_faces():
    encodings = []
    names = []
    for root, dirs, files in os.walk(KNOWN_FACES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                person_name = os.path.basename(root)
                image = face_recognition.load_image_file(image_path)
                face_encs = face_recognition.face_encodings(image)
                if face_encs:
                    encodings.append(face_encs[0])
                    names.append(person_name)
                else:
                    print(f'No face found in {image_path}')
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names}, f)
    print(f'Encoded {len(encodings)} faces.')

if __name__ == '__main__':
    encode_known_faces()
