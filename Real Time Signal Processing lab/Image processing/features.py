from skimage import feature
import numpy as np
import cv2

def lbp_hist(image, num_points, radius, eps = 1e-7):
	if isinstance(image, str):
		image = cv2.imread(image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")

	(hist, _) = np.histogram(lbp.ravel(), bins = np.arange(0, num_points + 1), range=(0, num_points))

	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)

	return hist

def hog(image):
	if isinstance(image, str):
		image = cv2.imread(image)
		image = cv2.resize(image, (150,150))
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), 
						visualize=False, visualise=None, transform_sqrt=True, feature_vector=True, block_norm ='L1')
	return image


