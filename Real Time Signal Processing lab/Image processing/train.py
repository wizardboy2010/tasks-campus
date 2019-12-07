import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.svm import LinearSVC
import argparse
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from features import lbp_hist, hog

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-data", "--train",
	help="path to the training images")
ap.add_argument("-model", "--test",
	help="path to the trained model")
args = vars(ap.parse_args())

def detect(gray, frame, model):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
    	cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    	#label = model.predict(lbp_hist(gray[y:y+h, x:x+w] ,25,8).reshape(1,-1))
    	label = model.predict(hog(cv2.resize(gray[y:y+h, x:x+w], (150,150))).reshape(1,-1))

    	cv2.putText(frame, label[0], (x+w,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
    	#print(label)

    return frame

if args['train'] is not None:
	path = args["train"]

	data_paths = [os.path.join(r, n) for r, _, f in os.walk(path) for n in f]

	#data = np.asarray([lbp_hist(i ,25,8) for i in tqdm(data_paths)])
	data = np.asarray([hog(i) for i in tqdm(data_paths)])
	labels = np.asarray([i.split('\\')[1] for i in tqdm(data_paths)])

	pkl.dump([data,labels], open('/data_array_hog.pkl', 'wb'))
	# data, labels = pkl.load(open('/data_array.pkl', 'rb'))
	
	arr = np.random.permutation(len(labels))

	data = data[arr]
	labels = labels[arr]

	scaler = StandardScaler()
	scaler.fit(data)
	#scaler.transform(data)

	x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7, random_state = 42)

	model = LinearSVC(C = 150, random_state=42)
	model.fit(x_train, y_train)
	print('train accuracy: ', model.score(x_train, y_train))
	print('test accuracy: ', model.score(x_test, y_test))
	pkl.dump(model, open('svm_model_hog', 'wb'))

elif args['test'] is not None:
	model = pkl.load(open(args['test'], 'rb'))

	video_capture = cv2.VideoCapture(0)

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	while True:
	    _, frame = video_capture.read()
	    
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    canvas = detect(gray, frame, model)
	    cv2.imshow('Video', canvas)

	    if cv2.waitKey(1) & 0xFF == ord('q'):
	    	break

	video_capture.release()
	cv2.destroyAllWindows()
