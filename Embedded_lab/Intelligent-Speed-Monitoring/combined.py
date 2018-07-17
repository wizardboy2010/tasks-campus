import cv2
import os

os.system("pscp pi@10.9.19.31:/home/pi/webcam/img.jpg C:\\Users\\user\\Desktop\\acads\\sem6\\Embedded-lab\\Capstone\\rasp")

face_cascade = cv2.CascadeClassifier('haar_models/haarcascade_frontalface_default.xml') # We load the cascade for the face.
plate_cascade = cv2.CascadeClassifier('aar_models/haarcascade_russian_plate_number.xml') # We load the cascade for the plate.

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    plates = plate_cascade.detectMultiScale(gray, 1.1, 1) # We apply the detectMultiScale method from the plate cascade to locate one or several vehichles in the image.
    for (x, y, w, h) in plates: # For each detected plate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1) # We paint a rectangle around the plate.
    
    faces = face_cascade.detectMultiScale(gray, 1.2, 4) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 1) # We paint a rectangle around the face.

    return frame # We return the image with the detector rectangles.

while (1): # We repeat infinitely (until break):
    img_capture = cv2.imread('img.jpg') # scan the image
    gray = cv2.cvtColor(img_capture, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, img_capture) # We get the output of our detect function.
    cv2.imshow('Image', canvas) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.
