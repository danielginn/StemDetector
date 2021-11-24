import numpy as np
import cv2
import random
import my_functions
import my_common_modules as my_modules
import os

filepaths, numImages = my_modules.import_dataset_filepaths('..\\Mask_RCNN\\WeedCode\\Synthetic_sugarbeets\\data\\occlusion_00\\')

images = filepaths['image']

random.seed(8143)  # 123, 8143, 13424, 540932, 43090923
random.shuffle(images)

traintextfile = open("..\\Mask_RCNN\\WeedCode\\Synthetic_sugarbeets\\data\\occlusion_00\\train2.txt", "w")
for element in images[:1600]:
    traintextfile.write(os.path.basename(element) + "\n")
traintextfile.close()

testtextfile = open("..\\Mask_RCNN\\WeedCode\\Synthetic_sugarbeets\\data\\occlusion_00\\test2.txt", "w")
for element in images[1600:]:
    testtextfile.write(os.path.basename(element) + "\n")
testtextfile.close()
