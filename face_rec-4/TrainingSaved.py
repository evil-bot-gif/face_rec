import face_recognition as fr
import os
import cv2
import pickle

# Youtube video link:https://www.youtube.com/watch?v=2G_9LM4OVpM&list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_&index=40

# Encoding of training data array in Encodings array
Encodings = []
# Label of images
Names = []

# loop through the files inside the faces folders to train the model 
for rootpath, dirs, files in os.walk("./faces"):
    for img_file in files:
        print(img_file)
        name = os.path.splitext(img_file)[0]
        face = fr.load_image_file("faces/" + img_file)
        # Encoded training data array
        encoding = fr.face_encodings(face)[0]
        Encodings.append(encoding)
        Names.append(name)
print(Names)

with open('train.pkl','wb') as f:
    pickle.dump(Encodings,f)
    pickle.dump(Names,f)