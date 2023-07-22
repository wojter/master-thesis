import os
import time
import random
from tqdm import tqdm
import argparse

from deepface.DeepFace import represent, build_model
from deepface.commons import distance, functions

import matplotlib.pyplot as plt
from deepface import DeepFace
import pandas as pd

from common import img_name_generator

db_identity = os.path.join("CelebA", "img_db")
db_to_test = os.path.join("CelebA", "img_prepared")

models = {
    "VGG-Face",
    "OpenFace",
    "Facenet",
    "Facenet512",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace",
}

NUM_IDENT_TO_TEST = 2
NUM_NEGATIVE_IDENT_ = 6
selected_model = "Facenet"

positives_distances = []
negatives_distances = []

def args_inpust():
    parser = argparse.ArgumentParser(
        description="Prepare img db from CelebA dataset")
    parser.add_argument("-n", "--num_identities_to_test", default=1000,
                        help="Specify number identities in db")
    parser.add_argument("-k", "--num_img_of_identity",
                        default=11, help="Number of imgs of one identity")
    parser.add_argument("-l", "--num_img_identity_as_ref",
                        default=3, help="Number of imgs as reference, unchanged")
    parser.add_argument("-m", "--model_name", default="VGG-Face", help="""Select one of recognition models: 
                        VGG-Face, OpenFace, Facenet, Facenet512, DeepFace, DeepID, Dlib, ArcFace, SFace""")
    return parser.parse_args()

def args_parser(args):
    model_name = args.model_name
    num_img_as_ref = int(args.num_img_identity_as_ref)
    num_img_of_ident = int(args.num_img_of_identity)
    num_ident = int(args.num_identities_to_test)
    return model_name, num_img_as_ref, num_img_of_ident, num_ident

def generat_pair():
    l = random.randint(1, NUM_IDENT_TO_TEST)
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
        
def generate_negative_pairs_list():
    for ident in range(1, NUM_IDENT_TO_TEST + 1):
        for i in range(1, 4):
            for j in generate_negative_pair(ident, i):
                print(j)

def main():
    model = build_model(selected_model)
    target_size = functions.find_target_size(model_name=selected_model)

    print("Analizing possitive pairs")
    for ident in tqdm(range(1, NUM_IDENT_TO_TEST + 1)):
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
                dist = distance.findEuclideanDistance(
                    distance.l2_normalize(embedding_org),
                    distance.l2_normalize(embedding),
                )
                dist_rounded = round(dist, 4)
                positives_distances.append(dist_rounded)
    df1 = pd.DataFrame(positives_distances)
    pos_dist = pd.DataFrame(positives_distances, columns = ["distance"])
    pos_dist["decision"] = "Yes"
    print("Mean post: ", df1.mean())
    print("Std pos: ", df1.std())
    print(pos_dist[pos_dist["decision"] == "Yes"].distance.mean())
    print(pos_dist[pos_dist["decision"] == "Yes"].distance.std())

    for ident in range(1, NUM_IDENT_TO_TEST + 1):
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
                dist = distance.findEuclideanDistance(
                    distance.l2_normalize(embedding_org),
                    distance.l2_normalize(embedding),
                )
                dist_rounded = round(dist, 4)
                negatives_distances.append(dist_rounded)

    df2 = pd.DataFrame(negatives_distances)
    neg_dist = pd.DataFrame(negatives_distances, columns=["distance"])
    neg_dist["decision"] = "No"

    print("Mean neg: ", df2.mean())
    print("Std neg: ", df2.std())
    print(neg_dist[neg_dist["decision"] == "No"].distance.mean())
    print(neg_dist[neg_dist["decision"] == "No"].distance.std())

    df = pd.concat([pos_dist, neg_dist]).reset_index(drop = True)
    tp_mean = round(df[df.decision == "Yes"].distance.mean(), 4)
    tp_std = round(df[df.decision == "Yes"].distance.std(), 4)
    fp_mean = round(df[df.decision == "No"].distance.mean(), 4)
    fp_std = round(df[df.decision == "No"].distance.std(), 4)


    print(tp_mean, tp_std, fp_mean, fp_std)
    
    # df1.plot.kde()
    # plt.figure()
    df2.plot.kde()

    # df[df.decision == "Yes"].distance.plot.kde()
    # df[df.decision == "No"].distance.plot.kde()
    plt.show()

if __name__ == "__main__":
    main()




# for i in range(6000):
#     negative = []
#     j, k, l, m = generate_negative_pair()
#     negative.append(img_name_generator(j, k))
#     negative.append(img_name_generator(l, m))
#     negatives.append(negative)


# negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
# negatives.file_x = "CelebA/img_db/"+negatives.file_x
# negatives.file_y = "CelebA/img_prepared/"+negatives.file_y

# print(positives)
# print(negatives)

# instances = positives[["file_x", "file_y"]].values.tolist()
# print(instances)
# resp_obj = DeepFace.verify(instances, model_name = SELECTED_MODEL, distance_metric = "cosine")
# print(resp_obj)
