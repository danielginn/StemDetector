import os
import my_common_modules as my_modules
import Occlusion_functions as my_functions
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import json
from math import sqrt,pi,cos,sin
from PIL import Image
import cv2
import itertools


IMAGE = "0143"

# Load in backgrounds
img_path = ".\\data\\backgrounds_cleaned\\rgb\\"+IMAGE+".png"
lbl_path = ".\\data\\backgrounds_cleaned\\gt\\"+IMAGE+".png"
col_lbl_path = ".\\data\\backgrounds_cleaned\\gt_color\\"+IMAGE+".png"
meta_path = ".\\data\\backgrounds_cleaned\\meta\\"+IMAGE+".json"

metadata = []
rows = 360
cols = 480
cx = cols / 2
cy = rows / 2
total_pix = rows * cols

sugarbeat_pct = 0
capsella_pct = 0
galium_pct = 0
seed = 123
random.seed(seed)
new_plant_data_bundles = my_functions.getShuffledPlants()

lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
col_lbl = cv2.imread(col_lbl_path)
img = cv2.imread(img_path)
meta = {"Background": lbl_path[-8:-4]}
print(meta)

# should background be flipped?
toFlip = random.choice([0, 1])
print("toFlip: " + str(toFlip))
meta["Background_flipped"] = toFlip
meta_json = open(meta_path)
stem_meta = json.load(meta_json)
meta_json.close()
stem = stem_meta["stem"]
if toFlip:
    print("Flipping")
    lbl = cv2.flip(lbl, 1)  # Horizontal flip left to right
    col_lbl = cv2.flip(col_lbl, 1)
    img = cv2.flip(img, 1)
    if stem is not None:
        stem_xs = stem["x"]
        for i in range(len(stem_xs)):
            stem_xs[i] = int(2 * cx - stem_xs[i])
        stem["x"] = stem_xs

# Bounding boxes for background
bboxes = []
white_mask = (lbl > 0)
contours, hierarchy = cv2.findContours(white_mask.astype('uint8'), 0, 2)
bboxid = 1
col_output = col_lbl.copy()
plant_id_mask = np.zeros([rows, cols], dtype='int')
radius_mask = np.zeros([rows, cols], dtype='int')
stem_mask = np.zeros([rows, cols], dtype='int')
plant_details = []
print("number of contours: "+str(len(contours)))

# Need to associate unconnected contours
compactContours = my_functions.associateUnconnectedContours(contours, stem)
circles = my_functions.findingEnclosingCircles(lbl, stem)

for stemCnts, circle in zip(compactContours,circles):
    center = circle[0]
    radius = circle[1]
    plant_mask = np.zeros([rows, cols], dtype='uint8')
    area = 0
    for cnt in stemCnts:
        cv2.drawContours(plant_mask, [cnt], 0, 1, -1)
        cv2.drawContours(plant_id_mask, [cnt], 0, bboxid, -1)
        cv2.drawContours(radius_mask, [cnt], 0, radius, -1)
        area += cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(plant_mask)
    cv2.rectangle(col_output, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 2)

    plant_id = np.max(lbl[y:y + h - 1, x:x + w - 1])
    plant_details.append([area, 0])
    stem_x, stem_y = my_functions.findStemForBBox(stem["x"], stem["y"], x, y, w, h)

    stem_mask[stem_y, stem_x] = plant_id
    plant_names = ["Sugarbeat", "Capsella", "Galium"]  # Plant list
    bbox = {
        "plant_id": bboxid,
        "inserted": False,
        "species_id": plant_id,
        "plant_name": plant_names[plant_id - 1],
        "original_pixels": area,
        "occluded_pixels": 0,
        "stem": {
            "x": int(stem_x),
            "y": int(stem_y)
        },
        "bndbox": {
            "xmin": x,
            "ymin": y,
            "xmax": x + w - 1,
            "ymax": y + h - 1},
        "bndcircle": {
            "center": center,
            "radius": radius
        }
    }
    bboxes.append(bbox)
    bboxid += 1

# Display background color label with bboxes
plt.figure(figsize=(16, 16))
plt.imshow(cv2.cvtColor(col_output, cv2.COLOR_BGR2RGB))
plt.show()
# Bundle Background image files together
plant_instances = (plant_id_mask, radius_mask, plant_details)
img_data_bundle = (img, lbl, col_lbl, plant_instances, bboxes, stem_mask)

# Add synthetic images
new_plants_added = random.randint(5, 12)  # How many new plants to add
new_plant_data_bundle_list = []
area_array = np.zeros(new_plants_added)
for i in range(new_plants_added):
    new_plant_filepaths = new_plant_data_bundles.pop()
    new_img = cv2.imread(new_plant_filepaths[0])
    new_lbl = cv2.imread(new_plant_filepaths[1], cv2.IMREAD_GRAYSCALE)
    new_col_lbl = cv2.imread(new_plant_filepaths[2])
    meta_json = open(new_plant_filepaths[3])
    stem_meta = json.load(meta_json)
    meta_json.close()
    new_stem = stem_meta["stem"]
    area = np.sum((new_lbl > 0).astype('int'))
    area_array[i] = area
    new_plant_data_bundle_list.append((new_img, new_lbl, new_col_lbl, new_stem))

sorted_images = np.flip(np.argsort(area_array))
sorted_plant_data_bundle_list = []
for i in range(new_plants_added):
    sorted_plant_data_bundle_list.append(new_plant_data_bundle_list[sorted_images[i]])

img_data_bundle = my_functions.addPlant(img_data_bundle, sorted_plant_data_bundle_list, occlusion_percent=0.20,
                           new_location_chance=0.2)
# plt.figure(figsize=(16,16))
img = img_data_bundle[0]
lbl = img_data_bundle[1]
print("lbl max: " + str(np.max(lbl)))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
plt.figure(figsize=(16, 16))
plt.imshow(cv2.cvtColor(img_data_bundle[2], cv2.COLOR_BGR2RGB))
plt.show()

stem_mask = img_data_bundle[5]
for bbox in img_data_bundle[4]:
    bndbox = bbox["bndbox"]
    xmin = bndbox["xmin"]
    ymin = bndbox["ymin"]
    xmax = bndbox["xmax"]
    ymax = bndbox["ymax"]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

for m in range(rows):
    for n in range(cols):
        if stem_mask[m, n] > 0:
            img[m - 2:m + 3, n - 2:n + 3, 0] = 147
            img[m - 2:m + 3, n - 2:n + 3, 1] = 20
            img[m - 2:m + 3, n - 2:n + 3, 2] = 255
plt.figure(figsize=(16, 16))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plant_instances = img_data_bundle[3]
plant_id_mask = plant_instances[0]

plant_id_mask = (plant_id_mask * 235 / np.max(plant_id_mask)).astype('uint8')
# print(np.shape(plant_id_mask))
# plt.figure(figsize=(16,16))
# plt.imshow(plant_id_mask,cmap='turbo')
# plt.show()

# plt.show()