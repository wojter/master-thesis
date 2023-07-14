from deepface import DeepFace
import os
from deepface.DeepFace import represent, build_model
from deepface.commons import distance, functions
import cv2 
import numpy as np

# tf_version = tf.__version__
# tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
# tf_minor_version = int(tf_version.split(".")[1])
# if tf_major_version == 1:
#     from keras.preprocessing import image
# elif tf_major_version == 2:
from tensorflow.keras.preprocessing import image

db_recognition_path = "CelebA\img_prepared"
db_detection_path = "CelebA\img_db_detection"

def name_generator(ident_idx, img_idx):
    name = f"{ident_idx:05d}_{img_idx:02d}.jpg"
    return name

target_size = functions.find_target_size("VGG-Face")

def resize_to_target_size(current_img, target_size, grayscale=False):
    if current_img.shape[0] > 0 and current_img.shape[1] > 0:
        factor_0 = target_size[0] / current_img.shape[0]
        factor_1 = target_size[1] / current_img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (
            int(current_img.shape[1] * factor),
            int(current_img.shape[0] * factor),
        )
        current_img = cv2.resize(current_img, dsize)

        diff_0 = target_size[0] - current_img.shape[0]
        diff_1 = target_size[1] - current_img.shape[1]
        if grayscale is False:
            # Put the base image in the middle of the padded image
            current_img = np.pad(
                current_img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                    (0, 0),
                ),
                "constant",
            )
        else:
            current_img = np.pad(
                current_img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                ),
                "constant",
            )
    if current_img.shape[0:2] != target_size:
        current_img = cv2.resize(current_img, target_size)

    # normalizing the image pixels
    # what this line doing? must?
    img_pixels = image.img_to_array(current_img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]
    return img_pixels

photo1 = os.path.join(db_recognition_path,name_generator(3,1))
photo2 = os.path.join(db_recognition_path,"00003_11.jpg")
photo3 = os.path.join(db_detection_path,"00003_02.jpg")
photo4 = os.path.join(db_detection_path,"00003_11.jpg")

res = represent(os.path.join(db_recognition_path,name_generator(3,11)), detector_backend="skip")
embedding = res[0]["embedding"]

for i in range(3):
    res = represent(os.path.join(db_recognition_path,name_generator(3,i+1)), detector_backend="skip")
    embedding2 = res[0]["embedding"]
    dist = distance.findCosineDistance(
        embedding2, embedding)
    print(dist)
# res = represent(photo1, detector_backend="skip")
# embedding = res[0]["embedding"]
# res = represent(photo2, detector_backend="skip")
# embedding2 = res[0]["embedding"]

# dist = distance.findCosineDistance(
#         embedding2, embedding)
# print(dist)

result = DeepFace.verify(os.path.join(db_recognition_path,name_generator(3,11)), os.path.join(db_recognition_path,name_generator(3,1)), detector_backend="skip", distance_metric="cosine")
# result = DeepFace.verify(photo1, photo2, detector_backend="opencv", distance_metric="euclidean")
print(result["distance"])
result = DeepFace.verify(os.path.join(db_recognition_path,name_generator(3,11)), os.path.join(db_recognition_path,name_generator(3,2)), detector_backend="skip", distance_metric="cosine")
print(result["distance"])
result = DeepFace.verify(os.path.join(db_recognition_path,name_generator(3,11)), os.path.join(db_recognition_path,name_generator(3,3)), detector_backend="skip", distance_metric="cosine")
print(result["distance"])

result = DeepFace.verify(photo3, photo4, detector_backend="mtcnn", distance_metric="cosine", align=False)
print(result["distance"])