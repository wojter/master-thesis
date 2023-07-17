import pandas as pd
import os
import shutil
import argparse
import time

import cv2

from common import img_name_generator

from deepface.commons import functions
from deepface.detectors import FaceDetector

total_images = 202_599
total_identities = 10_177

identity_path = "CelebA\Anno\identity_CelebA.txt"
imgs_path = "CelebA\img_celeba"
result_imgs_path = "CelebA\img_prepared"


def args_inpust():
    parser = argparse.ArgumentParser(description="Prepare img db from CelebA dataset")
    parser.add_argument(
        "-n",
        "--num_identities",
        default=5000,
        help="Specify number images to put in DB",
    )
    parser.add_argument(
        "-k", "--num_img_of_identity", default=11, help="Number of imgs of one identity"
    )
    parser.add_argument(
        "-t", "--parse-type", default=1, help="Parse type: 0-detection, 1-only copy"
    )
    parser.add_argument("-s", "--index_start", default=1)
    parser.add_argument("-a", "--index_skip", default=0)
    return parser.parse_args()


def load_identities_list():
    assert os.path.isfile(
        identity_path
    ), f"Not find identity file at path {identity_path}"

    colnames = ["file", "identity"]
    identities = pd.read_csv(identity_path, sep=" ", names=colnames, header=None)
    assert (
        identities["file"].nunique() == total_images
    ), "Import error, wrong total img number"
    assert (
        identities["identity"].nunique() == total_identities
    ), "Import error, wrong total identities number"
    assert identities.isna().values.any() == False, "NaN in table detected"
    return identities


def parse_identities_list(identities):
    temp = identities.drop("file", axis=1)
    temp = temp.groupby("identity").size().reset_index(name="occurences")
    temp = temp.drop(temp.index[temp["occurences"] < (min_images_of_person + 2)])
    return temp


def parse_ident_list(identities):
    df = identities["identity"].value_counts().reset_index()
    return df


def cp_rename_img(source_dir_path, source_file_name, dest_dir_path, dest_file_name):
    if os.path.isfile(os.path.join(source_dir_path, source_file_name)):
        if not os.path.exists(os.path.join(dest_dir_path, dest_file_name)):
            shutil.copy(
                os.path.join(source_dir_path, source_file_name),
                os.path.join(dest_dir_path, dest_file_name),
            )
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
    face_obj = FaceDetector.detect_faces(face_detector, "mtcnn", img, align=False)
    if len(face_obj) == 0:
        return (src_name, dst_name)
    cv2.imwrite(os.path.join(result_imgs_path, dst_name), face_obj[0][0])
    return None


def filter_cropp_rename_imgs(
    ident, ident_uniq, tot_ident_in_db, tot_imgs_of_ident, face_detector
):
    images_naming = {}
    total_identity_index = 0
    no_detections = []
    for index, row in ident.iterrows():
        if row.identity in ident_uniq["identity"].values:
            if row.identity in images_naming:
                total_index, total_img = images_naming[row.identity]
                if total_img < tot_imgs_of_ident:
                    total_img += 1

                    res = cropp_rename_img(
                        row.file,
                        img_name_generator(total_index, total_img),
                        face_detector,
                    )
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
                    res = cropp_rename_img(
                        row.file,
                        img_name_generator(total_identity_index, 1),
                        face_detector,
                    )
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


def generate_imgs_list(identities, ident_count):
    start = time.time()
    ident_count = ident_count.drop(
        ident_count.index[ident_count["count"] < min_images_of_person + 9]
    )
    print(ident_count)
    img_paths = {}
    set_counts = set(ident_count["identity"])  # set of selected identities id
    for idx, row in identities.iterrows():
        if row["identity"] in set_counts:
            if row["identity"] in img_paths:
                actual_paths = img_paths[row["identity"]]
                if len(actual_paths) < min_images_of_person:
                    actual_paths.append(row["file"])
                    img_paths[row["identity"]] = actual_paths
                else:
                    pass
            else:
                img_paths[row["identity"]] = [row["file"]]
    print("Identities: ", len(img_paths))
    length = 0
    for key, value in img_paths.items():
        length += len(value)
    print("Total imgs:", length)
    print("Time to generate imgs paths: ", time.time() - start)
    return img_paths


def iterate_detection(imgs_paths, face_detector, index_skip):
    """Detect faces and crop_rename_copy img"""
    ident_index = 1
    no_detections = []
    ident_skiped = 0
    start_detections = time.time()
    for key, img_paths in imgs_paths.items():
        if index_start > ident_index:
            ident_index += 1
            continue
        if index_skip > 0:
            index_skip -= 1  # to skip iterations of images taken before
            continue
        print("---------------------------\nIDENT processing  ", ident_index)
        img_idx = 1
        start = time.time()
        for img_path in img_paths:
            res_img_name = img_name_generator(ident_index, img_idx)
            res = cropp_rename_img(img_path, res_img_name, face_detector)
            if res is not None:
                no_detections.append(res)
                continue
            else:
                img_idx += 1
                if img_idx > min_images_of_person:
                    break
        if img_idx <= min_images_of_person:
            ident_skiped += 1
        else:
            ident_index += 1
        if ident_index > total_individuals_db + index_start - 1:
            break
        print(ident_index, " processed in ", time.time() - start)
    print("-----------\nTotal time processing ", time.time() - start_detections)
    print("No detection list: ", res)
    print("--------------\nIdent skipped: ", ident_skiped)


def iteration_only_copy(imgs_paths):
    """Iterates throught identities and copy_rename images of identity"""
    ident_index = 1
    start_copy = time.time()
    for key, img_paths in imgs_paths.items():
        print("IDENT processing  ", ident_index)
        img_idx = 1
        for img_path in img_paths:
            res_img_name = img_name_generator(ident_index, img_idx)
            res = cp_rename_img(imgs_path, img_path, result_imgs_path, res_img_name)
            img_idx += 1
        ident_index += 1
        if ident_index > total_individuals_db:
            break
    print("-----------\nTotal time processing ", time.time() - start_copy)


if __name__ == "__main__":
    args = args_inpust()
    min_images_of_person = int(args.num_img_of_identity)
    total_individuals_db = int(args.num_identities)
    parse_type = int(args.parse_type)
    index_start = int(args.index_start)
    index_skip = int(args.index_skip)

    identities = load_identities_list()
    ident_count = parse_ident_list(identities)
    img_paths = generate_imgs_list(identities, ident_count)

    if not os.path.isdir(result_imgs_path):
        os.mkdir(result_imgs_path)

    if parse_type == 0:
        face_detector = get_face_detector("mtcnn")
        iterate_detection(img_paths, face_detector, index_skip)
    elif parse_type == 1:
        iteration_only_copy(img_paths)
# TODO
# refactor cp functions to take folders as parameters
# add arg to select face detedtor
# split db prepare to parts ex 1 - 50
