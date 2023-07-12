import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from skimage.util import random_noise
import skimage

import argparse

db_recognition_path = "CelebA\db_recognition"

noise_types = {
    "gaussian",
    "salt_vs_pepper",
    "poisson"
    "gaussian_blur"
}


def args_inpust():
    parser = argparse.ArgumentParser(
        description="Add noise to photos in dir")
    parser.add_argument("-n", "--num_identities", default=5000,
                        help="Specify number images to put in DB")
    parser.add_argument("-k", "--num_img_of_identity",
                        default=11, help="Number of imgs of one identity")
    parser.add_argument("-l", "--num_img_identity_as_ref",
                        default=3, help="Number of imgs as reference, unchanged")
    parser.add_argument("-t", "--noise_type", default="gaussian_blur",
                        help="Specify noise type: gaussian, salt_vs_pepper, poisson, gaussian_blur")
    parser.add_argument("-d", "--source_dir_path",
                        default="CelebA\db_recognition")
    parser.add_argument("-r", "--result_dir_path", default="img_db")
    return parser.parse_args()


def args_parser(args):
    source_dir = args.source_dir_path
    result_dir = args.result_dir_path
    num_img_as_ref = int(args.num_img_identity_as_ref)
    num_img_of_ident = int(args.num_img_of_identity)
    num_ident = int(args.num_identities)
    selected_noise = args.noise_type
    return source_dir, result_dir, num_img_as_ref, num_img_of_ident, num_ident, selected_noise


# img = cv2.imread(os.path.join(db_recognition_path, "00001_01.jpg"))
img = skimage.io.imread(os.path.join(db_recognition_path, "00001_01.jpg"))


# blur = cv2.GaussianBlur(img, (5,5),1)

plt.subplot(141)
plt.imshow(img)
plt.title("original")

plt.subplot(142)
# plt.imshow(blur)
plt.title("averaging")

gaussian_blur = random_noise(img, mode="gaussian", var=0.0)
gaussian_blur2 = random_noise(img, mode="gaussian", var=0.25)

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


def gaussian_noise_generator(img, var=0.01, mean=0 ):
    # variande = standard deviationt ** 2
    img_w_noise = skimage.util.random_noise(img, mode="gaussian", mean=mean,var=var)
    return img_w_noise

def salt_vs_pepper_noise_generator(img, amount=0.05, svp=0.5 ):
    img_w_noise = skimage.util.random_noise(img, mode="s&p", amount=amount, salt_vs_pepper=svp )
    return img_w_noise

def poisson_nosie_generator(img):
    img_w_noise = skimage.util.random_noise(img, mode="poisson" )
    return img_w_noise

def gaussian_blur_generator(img, s_dev=1):
    # s_dev - standard deviation (default 1)
    img_w_blur = skimage.filters.gaussian(img, s_dev)
    return img_w_blur


if __name__ == "__main__":
    args = args_inpust()
    source_dir, result_dir, num_img_as_ref, num_img_of_ident, num_ident, selected_noise = args_parser(
        args)
    if selected_noise not in noise_types:
        raise ValueError("This noise is not implemented")

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
