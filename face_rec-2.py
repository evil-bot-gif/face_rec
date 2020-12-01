import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
# Youtube video link:https://www.youtube.com/watch?v=2G_9LM4OVpM&list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_&index=40

# Encoding of training data array in Encodings array
Encodings = []
# Label of images
Names = []

# loop through the files inside the faces folders to train the model 
for rootpath, dirs, files in os.walk("./faces"):
    for img_file in files:
        name = os.path.splitext(img_file)[0]
        face = fr.load_image_file("faces/" + img_file)
        # Encoded training data array
        encoding = face_recognition.face_encodings(face)[0]
        Encodings.append(encoding)
        Names.append(name)
print(Names)
# font for the box that is gonna be drawn around the faces detected in the image
font=cv2.FONT_HERSHEY_SIMPLEX
# for loop to compare the trained model with unknown images
for rootpath, dirs, files in os.walk("./unknown_faces"):
    for testImage_file in files:
        # load the testing image for comparision
        testImage = fr.load_image_file("unknown_faces/" + testImage_file)
        # Find the position of the faces inside the test image
        facePositions = fr.face_locations(testImage)
        # Encode all the faces found in the test image
        allEncodings = fr.face_encodings(testImage,facePositions)
        # Convert the testImage from RGB to BGR for cv2 to display
        testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
        # Create the bounding boxes using the facePositions and faces_encoding found in the test image
        for (top,right,bottom,left), testImage_face_encoding in zip(facePositions,allEncodings):
            # Default label for unknown faces detected
            name = 'unknown'
            # compare the faces in the test image with trained image
            matches = fr.compare_faces(Encodings,testImage_face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
            # If the faces in the train folder is found in the testImage select the matching name in Names[].
                name = Names[first_match_index]
            # Draw a box around the face
            cv2.rectangle(testImage, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(testImage, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            cv2.putText(testImage, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)
        # Display the image in a window
        cv2.imshow("Pictures",testImage)
        cv2.moveWindow('Pictures',0,0)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()

