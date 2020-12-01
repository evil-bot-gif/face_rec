import face_recognition as fr 
import cv2
import os
import numpy as np
from time import sleep

# Function to apppend the name and encoding of each face in the faces folder
def get_faces():
    # List of Encoding and Name for faces you want to identify
    Encodings = []
    Names = []
    for root, dirs, files in os.walk("./faces"):
        for image_fie in files:
            name = os.path.splitext(image_fie)[0]
            face = fr.load_image_file(f'faces/{image_fie}')
            encoding = fr.face_encodings(face)[0]
            Encodings.append(encoding)
            Names.append(name.title())
    return(Encodings,Names)

# Function to compare known and unknown faces within the image and display the image with a rounding box around thr face detected 
def compare_display(Encodings,Names):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for root, dirs, files in os.walk('./unknown_faces'):
        for image_file in files:
            face = fr.load_image_file(f'unknown_faces/{image_file}')
            face_positions = fr.face_locations(face)
            all_faceEncodings = fr.face_encodings(face,face_positions)
            face = cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
            for (top,right,bottom,left), faceEncoding in zip(face_positions,all_faceEncodings):
                name = "Unknown"
                matches = fr.compare_faces(Encodings,faceEncoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = Names[first_match_index]
                cv2.rectangle(face, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
                cv2.rectangle(face, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
                cv2.putText(face, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)
            cv2.imshow('Picture',face)
            cv2.moveWindow('Picture',0,0)
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()

# Main program

compare_display(*get_faces())