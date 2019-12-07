import cv2
import os
from skimage import feature
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import argparse
import pickle as pkl

ap = argparse.ArgumentParser()
ap.add_argument("-model", "--test", help="path to the model for test")
args = vars(ap.parse_args())

data_paths = [os.path.join(r, n) for r, _, f in os.walk('data') for n in f]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(gray, frame, model, save = False):
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		age =  model.predict(hog(cv2.resize(gray[y:y+h, x:x+w], (150,150))).reshape(1,-1))
		cv2.putText(frame, age[0], (x+w,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
		if save:
			cv2.imwrite('data/'+str(i)+'.jpg', frame[y:y+h, x:x+w])
	return frame

def hog(image):
	if isinstance(image, str):
		image = cv2.imread(image)
		image = cv2.resize(image, (150,150))
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), 
						visualize=False, visualise=None, transform_sqrt=True, feature_vector=True, block_norm ='L1')
	return image


data = np.asarray([hog(i) for i in tqdm(data_paths)])
labels = np.asarray([i.split('\\')[1] for i in tqdm(data_paths)])

arr = np.random.permutation(len(labels))

data = data[arr]
labels = labels[arr]

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7, random_state = 42)

model = LinearSVC(random_state=42)
model.fit(x_train, y_train)
print('train accuracy: ', model.score(x_train, y_train))
print('test accuracy: ', model.score(x_test, y_test))
#pkl.dump(model, open('svm_model_hog', 'wb'))

if args['test'] is not None:
    model = pkl.load(open(args['test'], 'rb'))
	# path = args['test']
	# image = cv2.imread(path)
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# frame = detect(gray, image, model)
	# cv2.imshow('age', frame)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	video_capture = cv2.VideoCapture(0)

	while True:
	    _, frame = video_capture.read()
	    
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    canvas = detect(gray, frame, model)
	    cv2.imshow('Video', canvas)

	    if cv2.waitKey(1) & 0xFF == ord('q'):
	    	break

	video_capture.release()
	cv2.destroyAllWindows()

