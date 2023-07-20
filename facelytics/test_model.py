import os
import time
from deepface import DeepFace

from common import img_name_generator

db_identity = os.path.join("CelebA", "img_db")
db_img_to_test = os.path.join("CelebA", "img_prepared")

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
start_time = time.time()
for i in range(4, 14):
    img_path = os.path.join(db_img_to_test, img_name_generator(4, i))
    res = DeepFace.find(
        img_path=img_path,
        db_path=db_identity,
        detector_backend="skip",
        model_name="ArcFace",
        distance_metric="euclidean_l2",
    )
    print(res[0])
print("END. Took ", time.time() - start_time, " s")
