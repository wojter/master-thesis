import numpy as np
import os
from matplotlib import pyplot as plt

import skimage

import argparse

noise_types = {
    "gaussian",
    "s_v_p",
    "poisson",
    "speckle",
    "gaussian_blur"
}

source_db_path = "CelebA/img_prepared"


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


def gaussian_noise_generator(img, var=0.01, mean=0):
    # variande = standard deviationt ** 2
    img_w_noise = skimage.util.random_noise(
        img, mode="gaussian", mean=mean, var=var)
    return img_w_noise


def salt_vs_pepper_noise_generator(img, amount=0.05, svp=0.5):
    img_w_noise = skimage.util.random_noise(
        img, mode="s&p", amount=amount, salt_vs_pepper=svp)
    return img_w_noise


def poisson_nosie_generator(img):
    img_w_noise = skimage.util.random_noise(img, mode="poisson")
    return img_w_noise


def gaussian_blur_generator(img, s_dev=1):
    # s_dev - standard deviation (default 1)
    img_w_blur = skimage.filters.gaussian(img, s_dev)
    return img_w_blur


def show_noise_examples(file_path, img_name):
    img = skimage.io.imread(os.path.join(file_path, img_name))

    i = 0
    for s_dev in np.arange(0, 5, 0.5):
        plt.suptitle("Gaussian blur", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(gaussian_blur_generator(img, s_dev))
        plt.title(s_dev)
        i += 1
    plt.show()

    i = 0
    for var in np.arange(0, 1, 0.1):
        plt.suptitle("Gaussian noise", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(gaussian_noise_generator(img, var ** 2))
        plt.title(var)
        i += 1
    plt.show()
    plt.close()

    i = 0
    for var in np.arange(0.01, 0.2, 0.02):
        plt.suptitle("Salt vs pepper noise", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(salt_vs_pepper_noise_generator(img, amount=var))
        plt.title(var)
        i += 1
    plt.show()

    i = 0
    for var in np.arange(0, 2, 1):
        plt.suptitle("Poisson noise", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(poisson_nosie_generator(img))
        plt.title(var)
        i += 1
    plt.show()


def get_list_all_img(dir_path):
    list_files = []
    for path in os.scandir(dir_path):
        if path.is_file():
            list_files.append(path.name)
    return list_files


def read_img(img_name):
    img_path = os.path.join(source_db_path, img_name)
    source_img = skimage.io.imread(img_path)
    return source_img


def write_img(img, img_name, dest_path):
    dest_path = os.path.join(dest_path, img_name)
    skimage.io.imsave(dest_path, img)


def create_dest_dir(selected_noise_type, param=None):
    dest_path = ""
    if param is None:
        dest_path = f"db_imgs_{selected_noise_type}"
    else:
        dest_path = f"db_imgs_{selected_noise_type}_{param:.02f}"
        dest_path = dest_path.replace(".", "_")
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    return dest_path

if __name__ == "__main__":
    args = args_inpust()
    source_dir, result_dir, num_img_as_ref, num_img_of_ident, num_ident, selected_noise = args_parser(
        args)
    if selected_noise not in noise_types:
        raise ValueError("This noise is not implemented")

    list_imgs = get_list_all_img("CelebA/img_db_recognition")

    if selected_noise == "gaussian":
        for var in np.arange(0.5, 5.5, 0.5):
            result_dir = create_dest_dir(selected_noise, var)
            for img in list_imgs:
                result_img = gaussian_noise_generator(read_img(img))
                write_img(result_img, img,)