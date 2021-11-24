import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def getPredictionModel():
    encoder_input = keras.Input(shape=(360, 480, 1), name="img")
    decoder_output = layers.Conv2D(1, 15, padding='same', activation=None, use_bias=False, kernel_initializer='Zeros', name='conv1')(encoder_input)
    encoder_decoder = keras.Model(encoder_input, decoder_output, name="encoder_decoder")
    return encoder_decoder

def add2Model(model):
    x = model.get_layer(name='conv1').output
    model2_output = layers.Conv2D(1, 30, padding='same', activation=None, use_bias=False, kernel_initializer='Zeros', name='conv2')(x)
    return keras.Model(model.inputs, model2_output)

# Model
model = getPredictionModel()
keras_layer = model.layers[1].get_weights()[0]
weights = np.zeros([15, 15, 1, 1], dtype=np.float32)
cv2.circle(weights[:,:,0,0],(7,7),7,1.0,-1)
model.layers[1].set_weights([weights])

# Model 2
model2 = add2Model(model)

model2.summary()

# Construct input image
img = np.zeros([1, 360, 480, 1], dtype=np.float32)
cv2.circle(img[0,:,:,0], (50,220), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (60,245), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (72,233), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (215,260), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (250,265), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (260,230), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (355,270), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (365,200), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (368,145), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (372,90), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (281,166), 10, 1.0, -1)
cv2.circle(img[0,:,:,0], (266,157), 10, 1.0, -1)

# Prediction
results = model.predict(x=img, batch_size=1)



ret, thresh = cv2.threshold(results[0,:,:,0],148, 255, 0)
contours,hierarchy = cv2.findContours(thresh.astype('uint8'), 1, 2)

for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(img[0,:,:,0], center, 1, 0.5, -1)

plt.imshow(img[0,:,:,0],cmap='gray')
plt.show()