import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import my_functions
import cv2

img = np.ones([15,15], dtype='uint8')*255
cv2.circle(img,(7,7),7,0,-1)

plt.imshow(img, cmap='gray')
plt.show()