import os
import pickle
import face_recognition as fr 

KNOWN_FACES_DIR = "known_faces"

known_faces_encoding = []
known_faces_name = []

print("loading known faces and extracting faces features.....")
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(filename)
        image = fr.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        # Encode the first face found in image
        encoding = fr.face_encodings(image)[0]
        print(encoding)
        known_faces_encoding.append(encoding)
        known_faces_name.append(name)
print(known_faces_name)




with open('known_faces_feature.pkl','wb') as f:
    pickle.dump(known_faces_encoding,f)
    pickle.dump(known_faces_name,f)