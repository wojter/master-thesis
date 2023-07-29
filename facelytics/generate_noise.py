import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
import skimage

from tqdm import tqdm

skimage.io.use_plugin("pil")

noise_types = {
    "all",
    "gaussian",
    "s_v_p",
    "poisson",
    "speckle",
    "gaussian_blur, exposure_dark, exposure_light",
}


def args_inpust():
    parser = argparse.ArgumentParser(description="Add noise to photos in dir")
    parser.add_argument(
        "-t",
        "--noise_type",
        default=None,
        help="Specify noise type: gaussian, salt_vs_pepper, poisson, gaussian_blur, exposure_dark, exposure_light, all",
    )
    parser.add_argument("-d", "--source_dir_path", default="CelebA/db-imgs")
    return parser.parse_args()


def args_parser(args):
    source_dir = args.source_dir_path
    selected_noise = args.noise_type
    return source_dir, selected_noise


def gaussian_noise_generator(img, s_dev=0.1, mean=0):
    # variande = standard deviationt ** 2
    img_w_noise = skimage.util.random_noise(
        img, mode="gaussian", mean=mean, var=s_dev**2
    )
    return img_w_noise


def salt_vs_pepper_noise_generator(img, amount=0.05, svp=0.5):
    img_w_noise = skimage.util.random_noise(
        img, mode="s&p", amount=amount, salt_vs_pepper=svp
    )
    return img_w_noise


def poisson_nosie_generator(img):
    img_w_noise = skimage.util.random_noise(img, mode="poisson")
    return img_w_noise


def gaussian_blur_generator(img, s_dev=1):
    # s_dev - standard deviation (default 1)
    img_w_blur = skimage.filters.gaussian(img, s_dev)
    return img_w_blur


def adjust_gamma_generator(img, gamma=1):
    # s_dev - standard deviation (default 1)
    img_adjusted = skimage.exposure.adjust_gamma(img, gamma)
    return img_adjusted


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
    for s_dev in np.arange(0, 1, 0.1):
        plt.suptitle("Gaussian noise", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(gaussian_noise_generator(img, s_dev))
        plt.title(s_dev)
        i += 1
    plt.show()
    plt.close()

    i = 0
    for amount in np.arange(0.02, 0.2, 0.02):
        plt.suptitle("Salt vs pepper noise", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(salt_vs_pepper_noise_generator(img, amount=amount))
        plt.title(amount)
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

    i = 0
    for gamma in np.arange(0.1, 1.1, 0.1):
        plt.suptitle("exposure lighter", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(adjust_gamma_generator(img, gamma))
        plt.title(gamma)
        i += 1
    plt.show()

    i = 0
    for gamma in np.arange(1, 4.1, 0.5):
        plt.suptitle("exposure darker", fontsize=18, y=0.95)
        plt.subplot(2, 5, 1 + i)
        plt.imshow(adjust_gamma_generator(img, gamma))
        plt.title(gamma)
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
        dest_path = f"db-imgs_{selected_noise_type}"
    else:
        dest_path = f"db-imgs_{selected_noise_type}_{param:.02f}"
        dest_path = dest_path.replace(".", "_")
    dest_path = os.path.join("CelebA", dest_path)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    return dest_path


if __name__ == "__main__":
    global source_db_path

    args = args_inpust()
    source_dir, selected_noise = args_parser(args)
    source_db_path = source_dir

    show_noise_examples("CelebA/db-ident", "00001_01.jpg")
    if selected_noise not in noise_types and selected_noise is not None:
        raise ValueError("This noise is not implemented")

    list_imgs = get_list_all_img(source_db_path)

    if selected_noise in ["gaussian", "all"]:
        print("-" * 80)
        print("generate GAUSSIAN noise\n")
        for s_dev in np.arange(0.05, 0.35, 0.05):
            print("generate gaussian noise, standard deviation:", s_dev)
            result_dir = create_dest_dir("gaussian", s_dev)
            for img in tqdm(list_imgs):
                result_img = gaussian_noise_generator(
                    read_img(img),
                    s_dev=s_dev,
                )
                write_img(result_img, img, result_dir)

    if selected_noise in ["s_v_p", "all"]:
        print("-" * 80)
        print("generate Salt_vs_Pepper noise\n")
        for var in np.arange(0.02, 0.22, 0.02):
            print("generate s_p noise, amount:", var)
            result_dir = create_dest_dir("s_v_", var)
            for img in tqdm(list_imgs):
                result_img = salt_vs_pepper_noise_generator(read_img(img), amount=var)
                write_img(result_img, img, result_dir)

    if selected_noise in ["poisson", "all"]:
        print("-" * 80)
        print("generate POISSON noise\n")
        result_dir = create_dest_dir("poisson")
        for img in tqdm(list_imgs):
            result_img = poisson_nosie_generator(read_img(img))
            write_img(result_img, img, result_dir)

    if selected_noise in ["gaussian_blur", "all"]:
        print("-" * 80)
        print("generate GAUSSIAN BLUR\n")
        for s_dev in np.arange(0.5, 5.5, 0.5):
            print("generate gaussian blur, standard deviation:", s_dev)
            result_dir = create_dest_dir("gaussian_blur", s_dev)
            for img in tqdm(list_imgs):
                result_img = gaussian_blur_generator(read_img(img), s_dev=s_dev)
                write_img(result_img, img, result_dir)

    if selected_noise in ["exposure_dark", "all"]:
        print("-" * 80)
        print("generate exposure darker\n")
        for gamma in np.arange(1.5, 3.5, 0.5):
            print("generate darker, gamma", gamma)
            result_dir = create_dest_dir("adj_gamma", gamma)
            for img in tqdm(list_imgs):
                result_img = adjust_gamma_generator(read_img(img), gamma)
                write_img(result_img, img, result_dir)

    if selected_noise in ["exposure_light", "all"]:
        print("-" * 80)
        print("generate exposure lighter\n")
        for gamma in np.arange(0.1, 0.6, 0.1):
            print("generate lighter, gamma", gamma)
            result_dir = create_dest_dir("adj_gamma", gamma)
            for img in tqdm(list_imgs):
                result_img = adjust_gamma_generator(read_img(img), gamma)
                write_img(result_img, img, result_dir)
