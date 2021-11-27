import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,pi,cos,sin

IMAGE = "0148"

# Load in backgrounds
#img_path = ".\\data\\backgrounds_cleaned\\gt\\"+IMAGE+".png"
#img = cv2.imread(img_path)
#diluted_white_mask = (img > 0)
# Taking a matrix of size 5 as the kernel
#kernel = np.ones((3, 3), np.uint8)
#diluted_white_mask = cv2.dilate(diluted_white_mask.astype('uint8'), kernel, iterations=3)
#diluted_white_mask = cv2.erode(diluted_white_mask, kernel, iterations=3)
#plt.imshow(diluted_white_mask*255)
#plt.show()

img = cv2.imread(".\\data\\occlusion_00\\stem_mask\\0035.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img*255, cmap='gray')
plt.show()