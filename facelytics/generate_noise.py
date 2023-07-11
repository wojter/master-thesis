import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from skimage.util import random_noise
import skimage
db_recognition_path = "CelebA\img_db_recognition"

img = cv2.imread(os.path.join(db_recognition_path, "00001_01.jpg"))
img = skimage.io.imread(os.path.join(db_recognition_path, "00001_01.jpg"))

blur = cv2.GaussianBlur(img, (5,5),1)

plt.subplot(141)
plt.imshow(img)
plt.title("original")

plt.subplot(142)
plt.imshow(blur)
plt.title("averaging")

gaussian_blur = random_noise(img, mode="gaussian", var=0.0 )
gaussian_blur2 = random_noise(img, mode="gaussian", var = 0.25 )

# gaussian_blur = cv2.GaussianBlur(img, (5,5), 100)
# gaussian_blur2 = cv2.GaussianBlur(img, (5,5), 1000)

gaussian_blur = skimage.filters.gaussian(img, 1)
gaussian_blur2 = skimage.filters.gaussian(img, 10)

plt.subplot(143)
plt.imshow(gaussian_blur)
plt.title("averaging")

plt.subplot(144)
plt.imshow(gaussian_blur2)
plt.title("averaging")

plt.show()

if __name__ == "__main__":
    pass

# image = img
# cv2.imshow("Original", image)
# kernelSizes = [(3, 3), (9, 9), (15, 15)]
# # loop over the kernel sizes
# for (kX, kY) in kernelSizes:
# 	# apply an "average" blur to the image using the current kernel
# 	# size
# 	blurred = cv2.blur(image, (kX, kY))
# 	cv2.imshow("Average ({}, {})".format(kX, kY), blurred)
# 	cv2.waitKey(0)