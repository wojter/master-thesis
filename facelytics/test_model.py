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
# start_time = time.time()
# for i in range(4, 14):
#     img_path = os.path.join(db_img_to_test, img_name_generator(4, i))
#     res = DeepFace.find(
#         img_path=img_path,
#         db_path=db_identity,
#         detector_backend="skip",
#         model_name="ArcFace",
#         distance_metric="euclidean_l2",
#     )
#     print(res[0])
# print("END. Took ", time.time() - start_time, " s")

NUM_IDENT_TO_TEST = 10
selected_model = "Facenet"

positives_distances = []
negatives_distances = []

def main():
    model = build_model(selected_model)
    target_size = functions.find_target_size(model_name=selected_model)

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
                    distance.l2_normalize(embedding_org), distance.l2_normalize(embedding)
                )
                dist_rounded = round(dist, 4)
                positives_distances.append(dist_rounded)

    df = pd.DataFrame(positives_distances)
    print(df.mean())
    print(df.std())

    plt.figure()
    df.plot.kde()
    plt.show()

if __name__ == "__main__":
    main()
# def generat_pair():
#     j = random.randint(1, NUM_IDENT_TO_TEST)
#     k = random.randint(1, 3)
#     l = random.randint(1, NUM_IDENT_TO_TEST)
#     m = random.randint(4, 9)
#     return j, k, l, m


# def generate_negative_pair():
#     while True:
#         j, k, l, m = generat_pair()
#         if j == l:
#             continue
#         else:
#             return j, k, l, m


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
