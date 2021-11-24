import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import my_functions
import my_common_modules as my_modules
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import json
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Loading in data
random.seed(123)
folder = '..\\Mask_RCNN\\WeedCode\\Synthetic_sugarbeets\\data\\occlusion_20\\'


filepaths, numImages = my_modules.import_dataset_filepaths(folder, 'train1.txt', 'test1.txt')
images = filepaths['image']
stems = filepaths['stem']

# splitting up images
x_train = images[:1600]
y_train = stems[:1600]
x_val = images[1600:]
y_val = stems[1600:]
x_pred = images[1600:1604]
y_pred = stems[1600:1604]

print(x_pred)

batch_size = 4

encoder_decoder = my_functions.getModel([360, 480])

encoder_decoder.summary()

encoder_decoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

encoder_decoder.load_weights('weights_11_12_a.h5')
traingen = my_functions.CustomDataGen(input_filepaths=x_train, output_filepaths=y_train, batch_size=batch_size, input_size=[360, 480, 3])
means, vars, range95, range05 = traingen.get_calibration_values()

predictgen = my_functions.CustomDataGen(input_filepaths=x_val, output_filepaths=y_val, batch_size=4, input_size=[360, 480, 3], means=means, vars=vars, range95=range95, range05=range05)
result = encoder_decoder.predict(x=predictgen)

ret, thresh0 = cv2.threshold(result[0, :, :], 0.1, 255, cv2.THRESH_BINARY)
ret, thresh1 = cv2.threshold(result[1, :, :], 0.1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(result[2, :, :], 0.1, 255, cv2.THRESH_BINARY)
ret, thresh3 = cv2.threshold(result[3, :, :], 0.1, 255, cv2.THRESH_BINARY)
threshs = [thresh0.astype('uint8'), thresh1.astype('uint8'), thresh2.astype('uint8'), thresh3.astype('uint8')]

# Find contours
#predictions = np.zeros([4,360,480])
for thresh in threshs:
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for cnt in contours:
    #    if cv2.contourArea(cnt) > 50:
    circles = cv2.HoughCircles(thresh, method=cv2.HOUGH_GRADIENT, dp=2, minDist=15, param1=255)
    print(circles)
    #for i in range(np.size(circles[0])):
    #    center = (int(circles[i][0]), int(circles[i][1]))
    #    radius = int(circles[i][2])
    #    cv2.circle(thresh, center, radius, 128, thickness=3)



kernel = np.ones((3, 3), np.uint8)

plt.figure(figsize=(21, 21))
plt.subplot(3, 4, 1)
img = cv2.imread(x_pred[0])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 5)
img = cv2.imread(y_pred[0])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 9)
#threshs[0] = cv2.erode(threshs[0], kernel, iterations=6)
plt.imshow(threshs[0], cmap='gray')

plt.subplot(3, 4, 2)
img = cv2.imread(x_pred[1])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 6)
img = cv2.imread(y_pred[1])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 10)
#threshs[1] = cv2.erode(threshs[1], kernel, iterations=6)
plt.imshow(threshs[1], cmap='gray')

plt.subplot(3, 4, 3)
img = cv2.imread(x_pred[2])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 7)
img = cv2.imread(y_pred[2])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 11)
#threshs[2] = cv2.erode(threshs[2], kernel, iterations=6)
plt.imshow(threshs[2], cmap='gray')

plt.subplot(3, 4, 4)
img = cv2.imread(x_pred[3])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 8)
img = cv2.imread(y_pred[3])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 12)
#threshs[3] = cv2.erode(threshs[3], kernel, iterations=6)
plt.imshow(threshs[3], cmap='gray')

plt.show()