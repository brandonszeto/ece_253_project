"""
	Run file: noise_distortions.py <image_path> <noise_type: 0(none),1(awgn),2(s&p)>

	Or use function: distort_img(img_path, noise_type, mean=0, std=20)
		Parameters
			img_path = directory path to image
			noise_type = 1:apply AWGN, 2:apply SaltPepper
			(optional/default) mean, std = AWGN parameters
		returns distorted image as np.ndarray

"""

import cv2
import numpy as np
import os
import sys
import skimage as ski

def add_awgn(image, mean, std):
	# add awgn to image: mu=0, std = 20
	noise = np.random.normal(mean,std,size=image.shape)
	awgn_img = image.astype(np.float32) + noise
	awgn_img = np.clip(awgn_img, 0, 255).astype(np.uint8)

	return awgn_img

"""
 This code was adding salt and pepper noise,
 but applied per channel, so it came out as colored speckling
"""
#def add_saltpepper(image):	
#	# add salt&pepper to image
#	sp_img = ski.util.random_noise(image, mode='s&p')
#	sp_img = (sp_img * 255).astype(np.uint8)
#
#	return sp_img


def add_saltpepper(image, amount=0.05):
	# add B&W salt/pepper noise manually
	sp_img = image.copy()
	h, w, c = image.shape

	# num distorted pixels
	num_pixels = int(amount * h * w)

	#  randomize distorted pixels
	coords = np.random.choice(h * w, num_pixels, replace=False)
	ys = coords // w
	xs = coords % w

	# split salt vs pepper
	half = num_pixels // 2

	# pepper (black)
	sp_img[ys[:half], xs[:half]] = 0

	# salt (white)
	sp_img[ys[half:], xs[half:]] = 255

	return sp_img


# Returns noise_type distorted image as np.ndarray
# noise_type 1: AWGN, 2: S&P, optionally adjust mean, std if noise_type 1
def distort_img(image_path, noise_type, mean = 0, std = 20):
	# Load the image (given img path)
	image = cv2.imread(image_path)
	if image is None:
		raise FileNotFoundError(image_path)

	# Generate and display noise distorted images
	if noise_type ==1:
		awgn_img = add_awgn(image, mean, std)
	elif noise_type ==2:
		sp_img = add_saltpepper(image)
	else:
		raise ValueError("noise_type must be 1 or 2")

	# Return distorted image 
	if noise_type ==1:
		return awgn_img
	elif noise_type ==2:
		return sp_img 
	else:
		print("No distortion selected: returned None")
		return None


if __name__ == "__main__":	
	if len(sys.argv) != 3:
		print("Usage: python main.py <path_to_image> <noise_type: 1=AWGN, 2=s&p")
		sys.exit(1)

	img_path = sys.argv[1] 
	noise_type = int(sys.argv[2])

	if not os.path.exists(img_path):
		print(f"Error: Image file '{img_path}' not found.")
		sys.exit(1)

	noised_img = distort_img(img_path, noise_type)

	if noise_type ==1:
		title_string = "AWGN"
	elif noise_type ==2:
		title_string = "Salt and Pepper"
	else:
		title_string = "img"

	cv2.imshow(title_string, noised_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
