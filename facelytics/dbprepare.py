import pandas as pd
import os
import shutil
import argparse

total_images = 202_599
total_identities = 10_177

identity_path = "CelebA\Anno\identity_CelebA.txt"
imgs_path = "CelebA\img_celeba2"
result_imgs_path = "CelebA\img_prepared"


def args_inpust():
    parser = argparse.ArgumentParser(
        description="Prepare img db from CelebA dataset")
    parser.add_argument("-n", "--num_identities", default=5000,
                        help="Specify number images to put in DB")
    parser.add_argument("-k", "--num_img_of_identity",
                        default=8, help="Number of imgs of one identity")
    return parser.parse_args()

def load_parse_identities_list():
    assert os.path.isfile(
        identity_path), f"Not find identity file at path {identity_path}"
    
    colnames = ['file', 'identity']
    identities = pd.read_csv(identity_path, sep=" ",
                             names=colnames, header=None)
    assert identities['file'].nunique(
    ) == total_images, "Import error, wrong total img number"
    assert identities['identity'].nunique(
    ) == total_identities, "Import error, wrong total identities number"
    assert identities.isna().values.any() == False, "NaN in table detected"

    temp = identities.drop("file", axis=1)
    temp = temp.groupby('identity').size().reset_index(name='occurences')
    temp = temp.drop(temp.index[temp["occurences"] < min_images_of_person])

    return identities, temp

def cp_rename_img(source_dir_path, source_file_name, dest_dir_path, dest_file_name):
    print(os.path.isfile(os.path.join(source_dir_path, source_file_name)))
    if os.path.isfile(os.path.join(source_dir_path, source_file_name)):
        if not os.path.exists(os.path.join(dest_dir_path, dest_file_name)):
            shutil.copy(os.path.join(source_dir_path, source_file_name),
                        os.path.join(dest_dir_path, dest_file_name))
        else:
            raise FileExistsError("Img exist in dest dir")
    else:
        raise FileNotFoundError("Img file not exist")


if __name__ == "__main__":
    args = args_inpust()
    min_images_of_person = int(args.num_img_of_identity)
    total_individuals_db = int(args.num_identities)
    identities, identities_unique = load_parse_identities_list()
    try:
        cp_rename_img(imgs_path, '000001.jpg', result_imgs_path, '1.jpg')
    except FileNotFoundError:
        print("No file to cp an rename")
    except FileExistsError:
        print("File already exist in dest dir")
    print(os.path.isfile(os.path.join(imgs_path, '000001.jpg')))