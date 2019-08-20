from __future__ import print_function
import os
from PIL import Image, ImageOps, ImageChops
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from matplotlib import pyplot as plt

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

image1_path = '/Users/DrorBlum/Desktop/ll_K/rgb/FLIR_00003.jpg'
image2_path = '/Users/DrorBlum/Desktop/ll_K/thermal/FLIR_00003.tiff'
output_path = '/Users/DrorBlum/Desktop/ll_K/output'

# im = Image.open(image1_path)
# im1 = im.save(image1_path)

im1 = cv2.imread(image1_path,cv2.COLOR_BGR2GRAY)
im2 = cv2.imread(image2_path,cv2.COLOR_BGR2GRAY)
# im1 = cv2.resize(im1,im2.shape[:2])
imArr = np.array(im2)
imArr = (imArr - imArr.min())
imArr = np.int_(imArr.astype(float)/imArr.max()*255)

# im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# print(im1.shape[:2])
print(imArr)

plt.figure()
plt.imshow(im1.astype(np.uint8), cmap=plt.get_cmap('gray'))
plt.show()  # display it



#diff_image = ImageChops.difference(Image.open(image1_path), Image.open(image2_path))
#diff_image.save(output_path)