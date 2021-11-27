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
folder = '.\\data\\occlusion_30\\'


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

#encoder_decoder.summary()

checkpoint = ModelCheckpoint("weights-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='auto', save_freq=400, save_weights_only=True)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=8000,
    decay_rate=0.1,
    staircase=True)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
lr_metric = get_lr_metric(optimizer)

encoder_decoder.compile(optimizer=optimizer, metrics=['accuracy', lr_metric], loss='categorical_crossentropy')
#encoder_decoder.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss=my_functions.custom_loss_function)

traingen = my_functions.CustomDataGen(input_filepaths=x_train, output_filepaths=y_train, batch_size=batch_size, input_size=[360, 480, 3])
means, vars, range95, range05 = traingen.get_calibration_values()
testgen = my_functions.CustomDataGen(input_filepaths=x_val, output_filepaths=y_val, batch_size=batch_size, input_size=[360, 480, 3], means=means, vars=vars, range95=range95, range05=range05)

encoder_decoder.fit(traingen,
                epochs=50,
                batch_size=batch_size,
                shuffle=True,
                validation_data=testgen,
                callbacks=[checkpoint])

#predictgen = my_functions.CustomDataGen(input_filepaths=x_pred, output_filepaths=y_pred, batch_size=4, input_size=[360, 480, 3], means=means, vars=vars, range95=range95, range05=range05)
#result = encoder_decoder.predict(x=predictgen)