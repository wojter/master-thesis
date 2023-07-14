import shutil
import os
from deepface import DeepFace

def name_generator(ident_idx, img_idx):
    name = f"{ident_idx:05d}_{img_idx:02d}.jpg"
    return name

db_recognition_path = "CelebA\img_prepared"
db_rec = "CelebA\img_db"
# for i in range(1, 1000 + 1):
#     for j in range(1, 3 + 1):
#         shutil.move(os.path.join(db_recognition_path, name_generator(i, j)), os.path.join(db_rec, name_generator(i,j)))

res = DeepFace.find(os.path.join(db_recognition_path,name_generator(3,11)),db_rec,detector_backend="skip")
print(res)