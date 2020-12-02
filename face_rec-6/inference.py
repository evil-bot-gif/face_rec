import face_recognition as fr
import cv2 
import pickle
import math

from face_recognition.api import face_distance, face_encodings 

# Function that provides confidence level based on euclidean distance
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
Encodings = []
Names = []
font = cv2.FONT_HERSHEY_DUPLEX
MODEL = 'cnn' 

with open('known_faces_feature.pkl','rb') as f:
    Encodings = pickle.load(f)
    Names= pickle.load(f)

# create a cam instance 
cam = cv2.VideoCapture(0) # ip webcam https://192.168.1.96:4747/video
# check if the camera is open
if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret , frame = cam.read()
    frameSmall = cv2.resize(frame,(0,0),fx=.33,fy=.33)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions = fr.face_locations(frameRGB, model = MODEL)
    allEncoding = fr.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left), faceEncoding in zip(facePositions,allEncoding):
        name = 'Unknown Person'
        acc = 100.0
        matches = fr.compare_faces(Encodings,faceEncoding)
        face_distances = fr.face_distance(Encodings,faceEncoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]
            print(f'Match Found: {name}')
            face_distance_match = face_distances[first_match_index]
            conf = face_distance_to_conf(face_distance_match)
            acc = conf * 100
        print(acc)
        top *= 3
        right *= 3
        bottom *= 3
        left *= 3 
        # Draw a boxes around the face 
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, name.title(), (left +6, bottom -6), font, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f'{round(acc,1)}%', (right +6, bottom -6), font, 0.65, (0, 0, 255), 2)
    cv2.imshow('Live Video',frame)
    cv2.moveWindow('Live Video',0,0)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()