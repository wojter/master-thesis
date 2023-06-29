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
imgs_path = "CelebA\img_celeba"
result_imgs_path = "CelebA\img_prepared"


def args_inpust():
    parser = argparse.ArgumentParser(
        description="Prepare img db from CelebA dataset")
    parser.add_argument("-n", "--num_identities", default=5000,
                        help="Specify number images to put in DB")
    parser.add_argument("-k", "--num_img_of_identity",
                        default=11, help="Number of imgs of one identity")
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
    temp = temp.drop(temp.index[temp["occurences"]
                     < (min_images_of_person + 2)])
    df = identities["identity"].value_counts().reset_index()
    print(df)
    # print(identities.groupby("identity").count())
    return identities, temp, df


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
        face_detector, "mtcnn", img, align=False)
    if len(face_obj) == 0:
        return (src_name, dst_name)
    cv2.imwrite(os.path.join(result_imgs_path, dst_name), face_obj[0][0])
    return None


def img_rename_generator(ident_idx, img_idx):
    dst_name = f"{ident_idx:05d}_{img_idx:02d}.jpg"
    return dst_name


def filter_cropp_rename_imgs(ident, ident_uniq, tot_ident_in_db, tot_imgs_of_ident, face_detector):
    images_naming = {}
    total_identity_index = 0
    no_detections = []
    for index, row in ident.iterrows():
        if row.identity in ident_uniq["identity"].values:
            if row.identity in images_naming:
                total_index, total_img = images_naming[row.identity]
                if total_img < tot_imgs_of_ident:
                    total_img += 1

                    res = cropp_rename_img(row.file, img_rename_generator(
                        total_index, total_img), face_detector)
                    if res is None:
                        images_naming[row.identity] = (total_index, total_img)
                    else:
                        total_img -= 1
                        no_detections.append(res)
                else:
                    pass
            else:
                if total_identity_index < tot_ident_in_db:
                    total_identity_index += 1
                    res = cropp_rename_img(row.file, img_rename_generator(
                        total_identity_index, 1), face_detector)
                    if res is None:
                        images_naming[row.identity] = (total_identity_index, 1)
                    else:
                        total_identity_index -= 1
                else:
                    pass
        else:
            pass

    print(len(images_naming))
    print("No detections list:\n", no_detections)


def parse(identities, df):
    df = df.drop(df.index[df["count"]
                            < min_images_of_person+2])
    img_paths = {}
    set_counts = set(df['identity'])
    for idx, row in identities.iterrows():
        if row['identity'] in set_counts:
            if row['identity'] in img_paths:
                actual_paths = img_paths[row['identity']]
                if len(actual_paths) < min_images_of_person:
                    actual_paths.append(row['file'])
                    img_paths[row['identity']] = actual_paths
                else:
                    pass
            else:
                img_paths[row['identity']] = [row['file']]
    print("Identities: ", len(img_paths))
    length = 0
    for key, value in img_paths.items():
        length += len(value)
    print("Total imgs:", length)
    return img_paths


if __name__ == "__main__":
    args = args_inpust()
    min_images_of_person = int(args.num_img_of_identity)
    total_individuals_db = int(args.num_identities)

    identities, identities_unique, df = load_parse_identities_list()
    print(identities_unique)
    img_paths = parse(identities, df)
    # face_detector = get_face_detector("mtcnn")
    # filter_cropp_rename_imgs(
    #     identities, identities_unique, total_individuals_db, min_images_of_person, face_detector)
