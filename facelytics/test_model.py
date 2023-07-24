import os
import random
from tqdm import tqdm
import argparse

import pandas as pd
from deepface.DeepFace import represent, build_model
from deepface.commons import distance, functions

import keras.backend as K

cfg = K.tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.compat.v1.Session(config=cfg))

from commons import img_name_generator, models

NUM_NEGATIVE_IDENT_ = 6
NUM_RESULT_DECIMAL_PLACES = 6


def args_input():
    parser = argparse.ArgumentParser(description="Prepare img db from CelebA dataset")
    parser.add_argument(
        "-n",
        "--num_identities_to_test",
        default=1000,
        help="Specify number identities in db to test",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default="VGG-Face",
        help="""Select one of recognition models: 
                        VGG-Face, OpenFace, Facenet, Facenet512, DeepFace, DeepID, Dlib, ArcFace, SFace""",
    )
    parser.add_argument(
        "-f",
        "--test_db_dir",
        default="img_prepared",
        help="""Specify path to dir with imgs to test""",
    )
    parser.add_argument(
        "-c",
        "--num_clean",
        default=51,
        help="""Specify num iterations to clear sesion""",
    )
    return parser.parse_args()


def args_parser(args):
    model_name = args.model_name
    if model_name not in models:
        raise ValueError("Incorrect model name")
    num_ident = int(args.num_identities_to_test)
    if num_ident < 2:
        raise ValueError("To small number of img to test")
    db_test_path = args.test_db_dir
    num_clean = int(args.num_clean)
    return model_name, num_ident, db_test_path, num_clean


def generat_pair():
    l = random.randint(1, num_ident_to_test)
    m = random.randint(4, 9)
    return l, m


def generate_negative_pair(j, k):
    negative_names = []
    while len(negative_names) < NUM_NEGATIVE_IDENT_:
        l, m = generat_pair()
        if j == l:
            continue
        else:
            negative_name = img_name_generator(l, m)
            if negative_name in negative_names:
                continue
            else:
                negative_names.append(negative_name)
    return negative_names


if __name__ == "__main__":
    args = args_input()
    selected_model, num_ident_to_test, db_test_path, num_clean = args_parser(args)

    db_to_test = os.path.join("CelebA", db_test_path)
    db_identity = os.path.join("CelebA", "img_db")
    if db_test_path == "img_prepared":
        noise_and_value = ""
    else:
        noise_and_value = db_test_path.replace("db_imgs", "")

    positives_distances = []
    negatives_distances = []

    model = build_model(selected_model)
    target_size = functions.find_target_size(model_name=selected_model)

    print("-" * 40)
    print("Selected to test: ", noise_and_value)

    print("Analizing positive pairs")
    for ident in tqdm(range(1, num_ident_to_test + 1)):
        if ident % num_clean == 0:
            K.clear_session()
        for i in range(1, 4):
            img_obj = functions.extract_faces(
                os.path.join(db_identity, img_name_generator(ident, i)),
                target_size=target_size,
                detector_backend="skip",
            )
            res = represent(
                img_obj[0][0], model_name=selected_model, detector_backend="skip"
            )
            embedding_org = res[0]["embedding"]
            for j in range(4, 10):
                img_obj = functions.extract_faces(
                    os.path.join(db_to_test, img_name_generator(ident, j)),
                    target_size=target_size,
                    detector_backend="skip",
                )
                res = represent(
                    img_obj[0][0], model_name=selected_model, detector_backend="skip"
                )
                embedding = res[0]["embedding"]
                dist_cosine = distance.findCosineDistance(embedding_org, embedding)
                dist = distance.findEuclideanDistance(embedding_org, embedding)
                dist_l2 = distance.findEuclideanDistance(
                    distance.l2_normalize(embedding_org),
                    distance.l2_normalize(embedding),
                )

                dist_cosine_rounded = round(dist_cosine, NUM_RESULT_DECIMAL_PLACES)
                dist_rounded = round(dist, NUM_RESULT_DECIMAL_PLACES)
                dist_rounded_l2 = round(dist_l2, NUM_RESULT_DECIMAL_PLACES)
                positives_distances.append(
                    [dist_cosine_rounded, dist_rounded, dist_rounded_l2]
                )

    pos_dist = pd.DataFrame(
        positives_distances, columns=["distance_cos", "distance_euc", "distance"]
    )
    pos_dist["decision"] = "Yes"

    print("Analizing negative pairs")
    for ident in tqdm(range(1, num_ident_to_test + 1)):
        if ident % num_clean == 0:
            K.clear_session()
        for i in range(1, 4):
            img_obj = functions.extract_faces(
                os.path.join(db_identity, img_name_generator(ident, i)),
                target_size=target_size,
                detector_backend="skip",
            )
            res = represent(
                img_obj[0][0], model_name=selected_model, detector_backend="skip"
            )
            embedding_org = res[0]["embedding"]
            for j in generate_negative_pair(ident, i):
                img_obj = functions.extract_faces(
                    os.path.join(db_to_test, j),
                    target_size=target_size,
                    detector_backend="skip",
                )
                res = represent(
                    img_obj[0][0], model_name=selected_model, detector_backend="skip"
                )
                embedding = res[0]["embedding"]
                dist_cosine = distance.findCosineDistance(embedding_org, embedding)
                dist = distance.findEuclideanDistance(embedding_org, embedding)
                dist_l2 = distance.findEuclideanDistance(
                    distance.l2_normalize(embedding_org),
                    distance.l2_normalize(embedding),
                )

                dist_cosine_rounded = round(dist_cosine, NUM_RESULT_DECIMAL_PLACES)
                dist_rounded = round(dist, NUM_RESULT_DECIMAL_PLACES)
                dist_rounded_l2 = round(dist_l2, NUM_RESULT_DECIMAL_PLACES)
                negatives_distances.append(
                    [dist_cosine_rounded, dist_rounded, dist_rounded_l2]
                )

    neg_dist = pd.DataFrame(
        negatives_distances, columns=["distance_cos", "distance_euc", "distance"]
    )
    neg_dist["decision"] = "No"

    df = pd.concat([pos_dist, neg_dist]).reset_index(drop=True)

    result_file_name = "result_" + selected_model + noise_and_value + ".csv"
    if not os.path.exists("results"):
        os.makedirs("results")
    result_file_name = os.path.join("results", result_file_name)
    df.to_csv(result_file_name)
