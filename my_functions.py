import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
from tensorflow.keras.models import Model
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from math import sqrt

class CustomDataGen(tf.keras.utils.Sequence):

        def __init__(self, input_filepaths, output_filepaths,
                     batch_size,
                     input_size=(224, 224, 3),
                     shuffle=True,
                     means=None,
                     vars=None,
                     range95=None,
                     range05=None,
                     predict=False):
                self.input_filepaths = input_filepaths.copy()
                self.output_filepaths = output_filepaths.copy()
                self.batch_size = batch_size
                self.input_size = input_size
                self.shuffle = shuffle
                self.predict = predict

                self.n = len(self.input_filepaths)
                if means is not None and vars is not None and range95 is not None and range05 is not None:
                        self.means = means
                        self.vars = vars
                        self.range95 = range95
                        self.range05 = range05
                else:
                        self.__extract_calibration_values()

        def get_calibration_values(self):
                return self.means, self.vars, self.range95, self.range05

        def on_epoch_end(self):
                if self.shuffle:
                        group = list(zip(self.input_filepaths,self.output_filepaths))
                        random.shuffle(group)
                        self.input_filepaths, self.output_filepaths = zip(*group)

        def __extract_calibration_values(self):
                images = np.zeros([self.n, self.input_size[0], self.input_size[1], self.input_size[2]], dtype=np.double)

                for i in range(self.n):
                        image = cv2.imread(self.input_filepaths[i])
                        image[:, :, 0] = gaussian_filter(image[:, :, 0], sigma=5)
                        image[:, :, 1] = gaussian_filter(image[:, :, 1], sigma=5)
                        image[:, :, 2] = gaussian_filter(image[:, :, 2], sigma=5)
                        images[i, :, :, :] = image/255.0

                means = np.zeros(3)
                # calculate mean
                means[0] = np.mean(images[:, :, :, 0])
                means[1] = np.mean(images[:, :, :, 1])
                means[2] = np.mean(images[:, :, :, 2])
                self.means = means

                vars = np.zeros(3)
                # calculate variance
                vars[0] = np.var(images[:, :, :, 0])
                vars[1] = np.var(images[:, :, :, 1])
                vars[2] = np.var(images[:, :, :, 2])
                self.vars = vars

                # calculate ranges
                range95 = np.zeros(3)
                range05 = np.zeros(3)
                range95[0] = np.percentile(images[:, :, :, 0], 99.95)
                range05[0] = np.percentile(images[:, :, :, 0], 0.05)
                range95[1] = np.percentile(images[:, :, :, 1], 99.95)
                range05[1] = np.percentile(images[:, :, :, 1], 0.05)
                range95[2] = np.percentile(images[:, :, :, 2], 99.95)
                range05[2] = np.percentile(images[:, :, :, 2], 0.05)
                self.range95 = range95
                self.range05 = range05


        def __get_input(self, path):
                image = cv2.imread(path)
                image = image/255.0

                # mean
                image[:, :, 0] -= self.means[0]
                image[:, :, 1] -= self.means[1]
                image[:, :, 2] -= self.means[2]

                # variance
                image[:, :, 0] /= self.vars[0]
                image[:, :, 1] /= self.vars[1]
                image[:, :, 2] /= self.vars[2]

                # contrast stretch
                image[:, :, 0] = np.clip(image[:, :, 0], self.range05[0], self.range95[0])
                image[:, :, 1] = np.clip(image[:, :, 1], self.range05[1], self.range95[1])
                image[:, :, 2] = np.clip(image[:, :, 2], self.range05[2], self.range95[2])

                # Stretching
                image[:, :, 0] = (image[:, :, 0] - self.range05[0]) * 1 / (self.range95[0] - self.range05[0]) - 0.5
                image[:, :, 1] = (image[:, :, 1] - self.range05[1]) * 1 / (self.range95[1] - self.range05[1]) - 0.5
                image[:, :, 2] = (image[:, :, 2] - self.range05[2]) * 1 / (self.range95[2] - self.range05[2]) - 0.5

                return image.astype(np.double)

        def __get_output(self, path):
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                return image.astype(np.double)

        def __getitem__(self, index):
                img_batch_filepaths = self.input_filepaths[index * self.batch_size:(index + 1) * self.batch_size]
                x_batch = np.asarray([self.__get_input(filepath) for filepath in img_batch_filepaths])
                if self.predict == False:
                        stem_batch_filepaths = self.output_filepaths[index * self.batch_size:(index + 1) * self.batch_size]
                        y_batch = np.asarray([self.__get_output(filepath) for filepath in stem_batch_filepaths])
                        return x_batch, y_batch
                else:
                        return x_batch

        def __len__(self):
                return self.n // self.batch_size

def custom_loss_function(y_true, y_pred):
        # the y inputs are going to have the shape [batch_size, rows, cols]
        # IoU = I/U, I = sum(y_pred*y_true), U = sum(y_pred+y_true-y_pred*y_true)
        # Loss(IoU) = 1 - IoU
        # dL(IoU)/dy_pred = {-1/U if y_true = 1, I/(U*U) otherwise}
        XmY = tf.math.multiply(y_true, y_pred)
        I = tf.math.reduce_sum(XmY)
        XpY = tf.math.add(y_true, y_pred)
        U = tf.math.reduce_sum(tf.math.subtract(XpY, XmY))
        L = 1 - I/U

        Y1 = -1/U
        Y0 = I/(U*U)
        inv_y_pred = tf.math.subtract(tf.ones(tf.shape(y_pred), dtype=tf.dtypes.float32), y_pred)
        Y0_output = tf.math.scalar_mul(Y0, inv_y_pred)
        Y1_output = tf.math.scalar_mul(Y1, y_pred)
        dL = tf.math.add(Y1_output, Y0_output)

        return dL   #L

def custom_metric(y_true, y_pred):
        pass



def getModel(inputShape):
        # Input
        encoder_input = keras.Input(shape=(inputShape[0], inputShape[1], 3), name="img")
        #x = layers.Rescaling(scale=1. / 255)(encoder_input)

        # Level 1 encoder - eg 480x360
        #x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='HeUniform')(x)
        x = layers.Conv2D(64, 3, padding='same', activation=None, kernel_initializer='HeUniform')(encoder_input)
        x = layers.LeakyReLU(alpha=0.001)(x)
        lvl1 = layers.Conv2D(64, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(lvl1)
        x = layers.MaxPooling2D()(x)

        # Level 2 encoder - eg 240x180
        x = layers.Conv2D(128, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        lvl2 = layers.Conv2D(128, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(lvl2)
        x = layers.MaxPooling2D()(x)

        # Level 3 encoder - eg 120x90
        x = layers.Conv2D(256, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        lvl3 = layers.Conv2D(256, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(lvl3)
        x = layers.MaxPooling2D()(x)

        # Level 4 encoder - eg 60x45
        x = layers.Conv2D(512, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.Conv2D(512, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)

        # Level 3 decoder - eg 120x90
        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(256, 2, padding='same', activation=None)(x)
        x = layers.Concatenate(axis=3)([lvl3,x])
        x = layers.Conv2D(256, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.Conv2D(256, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)

        # Level 2 decoder - eg 240x180
        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(128, 2, padding='same', activation=None)(x)
        x = layers.Concatenate(axis=3)([lvl2, x])
        x = layers.Conv2D(128, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.Conv2D(128, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)

        # Level 1 decoder - eg 480x320
        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(64, 2, padding='same', activation=None)(x)
        x = layers.Concatenate(axis=3)([lvl1, x])
        x = layers.Conv2D(64, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.Conv2D(64, 3, padding='same', activation=None, kernel_initializer='HeUniform')(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        decoder_output = layers.Conv2D(1, 1, padding='valid', activation=None, kernel_initializer='HeUniform', name='model_output')(x)

        encoder_decoder = keras.Model(encoder_input, decoder_output, name="encoder-decoder")
        return encoder_decoder


def getPredictionModel(model):
        x = model.get_layer(name='model_output').output
        decoder_output = layers.Conv2D(1, 15, padding='same', activation=None, use_bias=False,
                                       kernel_initializer='Zeros', name='circles')(x)
        model2 = keras.Model(model.inputs, decoder_output, name="encoder_decoder2")
        weights = np.zeros([15, 15, 1, 1], dtype=np.float32)
        cv2.circle(weights[:, :, 0, 0], (7, 7), 7, 1.0, -1)
        model2.get_layer(name='circles').set_weights([weights])
        return model2

def extractCiclesModel(inputShape):
        input = keras.Input(shape=(inputShape[0], inputShape[1], 1), name="img")
        output = layers.Conv2D(1, 15, padding='same', activation=None, use_bias=False,
                                       kernel_initializer='Zeros', name='circles')(input)
        model = keras.Model(input, output)
        weights = np.zeros([15, 15, 1, 1], dtype=np.float32)
        cv2.circle(weights[:, :, 0, 0], (7, 7), 7, 1.0, -1)
        model.get_layer(name='circles').set_weights([weights])
        return model

def matchCircles(circles_true, circles_pred):
        TP = 0
        FP = len(circles_pred)
        FN = 0
        FP_array = np.ones(len(circles_pred))
        for t in range(len(circles_true)):
                matchFound = False
                for p in range(len(circles_pred)):
                        x_diff = circles_true[t][0] - circles_pred[p][0]
                        y_diff = circles_true[t][1] - circles_pred[p][1]
                        d = sqrt(x_diff*x_diff + y_diff*y_diff)
                        if d < 8.08:
                                FP_array[p] = 0
                                if matchFound:
                                        print("Double Match found")
                                else:
                                        TP += 1
                                        matchFound = True
                if not matchFound:
                        FN += 1
        FP = np.sum(FP_array)
        return TP,FP,FN

''' Preprocessing Images:
- Gaussian smooth with 5x5 kernal, mean=0, SD=1
- subtract mean
- divide by variance
- contrast stretch the input values to the range [-0.5,0.5]
'''
def preprocessImages(images, means, vars, range95, range05):
        # gaussian smooth
        for i in range(np.shape(images)[0]):
                images[i, :, :, 0] = gaussian_filter(images[i, :, :, 0], sigma=5)
                images[i, :, :, 1] = gaussian_filter(images[i, :, :, 1], sigma=5)
                images[i, :, :, 2] = gaussian_filter(images[i, :, :, 2], sigma=5)

        # subtract by mean
        images[:, :, :, 0] -= means[0]
        images[:, :, :, 1] -= means[1]
        images[:, :, :, 2] -= means[2]

        # divide by variance
        images[:, :, :, 0] /= vars[0]
        images[:, :, :, 1] /= vars[1]
        images[:, :, :, 2] /= vars[2]

        # clip
        images[:, :, :, 0] = np.clip(images[:, :, :, 0], range05[0], range95[0])
        images[:, :, :, 1] = np.clip(images[:, :, :, 1], range05[1], range95[1])
        images[:, :, :, 2] = np.clip(images[:, :, :, 2], range05[2], range95[2])

        # Stretching
        images[:, :, :, 0] = (images[:, :, :, 0] - range05[0]) * 1/(range95[0]-range05[0]) - 0.5
        images[:, :, :, 1] = (images[:, :, :, 1] - range05[1]) * 1 / (range95[1] - range05[1]) - 0.5
        images[:, :, :, 2] = (images[:, :, :, 2] - range05[2]) * 1 / (range95[2] - range05[2]) - 0.5

        return images