import pandas as pd
import os
import shutil
import argparse

import cv2

from deepface.commons import functions
from deepface.detectors import FaceDetector

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


def get_face_detector(detector_backend):
    face_detector = FaceDetector.build_model(detector_backend)
    return face_detector


def cropp_rename_img(src_name, dst_name, face_detector):
    src_img_path = os.path.join(imgs_path, src_name)
    img = functions.load_image(src_img_path)
    face_obj = FaceDetector.detect_faces(
        face_detector, "mtcnn", img, align=True)
    if len(face_obj) == 0:
        return (src_name, dst_name)
    cv2.imwrite(os.path.join(result_imgs_path, dst_name), face_obj[0][0])
    return ()
    pass


def filter_cropp_rename_imgs(ident, ident_uniq, tot_ident_in_db, tot_imgs_of_ident, face_detector):
    # total imgs 5000, photos of one 11, no face on photo
    images_naming = {}
    total_identity_index = 0
    for index, row in ident.iterrows():
        # print(index, row.file, row.identity)
        if row.identity in ident_uniq["identity"].values:
            if row.identity in images_naming:
                total_index, total_img = images_naming[row.identity]
                if total_img < tot_imgs_of_ident:
                    cropp_rename_img(face_detector)
                    total_img += 1
                    images_naming[row.identity] = (total_index, total_img)
                else:
                    pass
            else:
                if total_identity_index < tot_ident_in_db:
                    cropp_rename_img(face_detector)
                    total_identity_index += 1
                    images_naming[row.identity] = (total_identity_index, 1)
                else:
                    pass
        else:
            pass

    print(len(images_naming))


if __name__ == "__main__":
    args = args_inpust()
    min_images_of_person = int(args.num_img_of_identity)
    total_individuals_db = int(args.num_identities)

    identities, identities_unique = load_parse_identities_list()

    print(identities_unique)
    face_detector = get_face_detector("mtcnn")
    filter_cropp_rename_imgs(
        identities, identities_unique, total_individuals_db, min_images_of_person, face_detector)
