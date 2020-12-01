"""Access IP Camera in Python OpenCV"""

import cv2

stream = cv2.VideoCapture('http://192.168.1.96:4747/video?640x480')

# Use the next line if your camera has a username and password
#stream = cv2.VideoCapture('protocol://username:password@IP:port/1')  
print(stream)
while True:

    r, f = stream.read()
    print(f)
    cv2.imshow('IP Camera stream',f)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()