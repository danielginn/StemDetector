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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Loading in data
random.seed(123)
folder = '.\\data\\occlusion_10\\'


filepaths, numImages = my_modules.import_dataset_filepaths(folder, 'train1.txt', 'test1.txt')
images = filepaths['image']
stems = filepaths['stem']
metas = filepaths['meta']

# splitting up images
x_train = images[:1600]
y_train = stems[:1600]
x_val = images[1600:]
y_val = stems[1600:]
x_pred = images[1600:]
y_pred = stems[1600:]
meta_pred = metas[1600:]

print(x_pred)

batch_size = 4
traingen = my_functions.CustomDataGen(input_filepaths=x_train, output_filepaths=y_train, batch_size=batch_size, input_size=[360, 480, 3])
means, vars, range95, range05 = traingen.get_calibration_values()
testgen = my_functions.CustomDataGen(input_filepaths=x_val, output_filepaths=y_val, batch_size=batch_size, input_size=[360, 480, 3], means=means, vars=vars, range95=range95, range05=range05)
predictgen = my_functions.CustomDataGen(input_filepaths=x_pred, output_filepaths=y_pred, batch_size=4, input_size=[360, 480, 3], means=means, vars=vars, range95=range95, range05=range05)


encoder_decoder = my_functions.getModel([360, 480])
encoder_decoder.load_weights('weights-50-10p.h5')
encoder_decoder.compile(optimizer='Adam', loss='categorical_crossentropy')
result = encoder_decoder.predict(x=predictgen)

circleFinder = my_functions.extractCiclesModel([360, 480])
circleFinder.compile(optimizer='Adam', loss=losses.MeanSquaredError())

for i in range(np.shape(result)[0]):
    ret, thresh = cv2.threshold(result[i, :, :, 0], 0.5, 1.0, 0)
    result[i, :, :, 0] = thresh

circleCenters = circleFinder.predict(x=result)
total_TP = 0
total_FP = 0
total_FN = 0

for i in range(np.shape(circleCenters)[0]):
    ret, thresh = cv2.threshold(circleCenters[i, :, :, 0], 148, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.astype('uint8'), 1, 2)

    # Find prediction circles
    circles_pred = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(result[i,:,:,0], center, 1, 0.5, -1)
        circles_pred.append(center)

    # Extract true circle locations
    circles_true = []
    f = open(meta_pred[i], 'r')
    data = json.load(f)
    for d in data:
        circles_true.append((d["stem"]["x"],d["stem"]["y"]))

    # Calculate Accuracy Score
    TP,FP,FN = my_functions.matchCircles(circles_true, circles_pred)
    total_TP += TP
    total_FP += FP
    total_FN += FN

print("True Positives: " + str(total_TP))
print("False Positives: " + str(total_FP))
print("False Negatives: " + str(total_FN))

plt.figure(figsize=(21, 21))
plt.subplot(3, 4, 1)
img = cv2.imread(x_pred[0])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 5)
img = cv2.imread(y_pred[0])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 9)
plt.imshow(result[0, :, :], cmap='gray')

plt.subplot(3, 4, 2)
img = cv2.imread(x_pred[1])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 6)
img = cv2.imread(y_pred[1])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 10)
plt.imshow(result[1, :, :], cmap='gray')

plt.subplot(3, 4, 3)
img = cv2.imread(x_pred[2])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 7)
img = cv2.imread(y_pred[2])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 11)
plt.imshow(result[2, :, :], cmap='gray')

plt.subplot(3, 4, 4)
img = cv2.imread(x_pred[3])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3, 4, 8)
img = cv2.imread(y_pred[3])
plt.imshow(img*255, cmap='gray')
plt.subplot(3, 4, 12)
plt.imshow(result[3, :, :], cmap='gray')

plt.show()

