import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(gray, frame, step=50, log = False):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
    	#cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    	if log and i%step ==0 and i//step<500 :
        	cv2.imwrite('suprise/face'+str((i//step)+1)+'.jpg', frame[y:y+h, x:x+w])
        	print('saved')


    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)

i = 0

while True:
    _, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame, step = 1, log = False)
    cv2.imshow('Video', canvas)
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or i>500:
    	print(cv2.face.LBPHFaceRecognizer_create(gray))
    	break

video_capture.release()
cv2.destroyAllWindows()