"""
	Run file: sim_distortion.py <image_path> 

	Or use function: add_waterStains(img_path)
		returns "water"-distorted image as np.ndarray

"""

import cv2
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import sys
import os

def add_waterStains(image_path)->np.ndarray:
	# Load the image (given img path): Image.Image
	image = Image.open(image_path)

	# Fixed parameters
	opacity_max = 0.25
	opacity_min = 0.1
	n_stains = 25
	diam_max = 220 #inclusive
	diam_min = 40
	blur_frac = 0.05 #for applying blur to stain as fraction of diameter

	# Random generation 
	rng = np.random.default_rng()

	image = image.convert("RGBA")
	W,H = image.size

	# Generate and apply n water stains
	for _ in range(n_stains):
		# set random size and position (ensuring fit within img)
		d = int(rng.integers(diam_min, diam_max+1))
		x_pos = int(rng.integers(0, max(1,W-d)))
		y_pos = int(rng.integers(0, max(1,H-d)))

		# set random stain color (dark->light grey) and opacity (within min/max)
		gray = int(rng.integers(40, 120))
		opacity = float(rng.uniform(opacity_min, opacity_max))
		alpha = int(opacity*255)

		# Create stain drawing	

		pad = int(d*blur_frac*2) # 2*blur_radius
		size = d+2*pad #prevent clip from stain drawing box when blurring
		stain= Image.new("RGBA", (size,size), (0,0,0,0)) #mode, size, color
		stain_core = ImageDraw.Draw(stain)

		# draw ellipse ImageDraw.ellipse(xy, fill=<(R,G,B,A)>, outline=<(R,G,B,A)>, width=1)
		xy = [(pad,pad),(pad+d-1,pad+d-1)]
		fill = (gray, gray, gray, alpha)
		outline = (gray, gray, gray, int(alpha*0.9))
		stain_core.ellipse(xy, fill=fill, outline=outline, width=3)

		# blur ellipse to look like water stain
		blur_radius = max(1, int(d*blur_frac))
		stain = stain.filter(ImageFilter.GaussianBlur(radius=blur_radius))
		stain = stain.crop((pad, pad, pad+d, pad+d))

		# Apply stain to image

		image.alpha_composite(stain, (x_pos,y_pos))

	# change PIL (RGB) to cv2 (BGR)
	wet_img_PIL = image.convert("RGB")
	temp = np.array(wet_img_PIL)
	wet_img = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)

	return wet_img


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python main.py <path_to_image> ")
		sys.exit(1)

	img_path = sys.argv[1] 

	if not os.path.exists(img_path):
		print(f"Error: Image file '{img_path}' not found.")
		sys.exit(1)

	wet_img = add_waterStains(img_path)

	cv2.imshow("Simulated Distortion", wet_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
