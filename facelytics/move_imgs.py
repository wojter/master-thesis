import shutil
import os
import tqdm

from common import img_name_generator


db_recognition_path = "CelebA\img_prepared"
db_rec = "CelebA\img_db"
for i in tqdm(range(1, 1000 + 1)):
    for j in range(1, 3 + 1):
        shutil.move(
            os.path.join(db_recognition_path, img_name_generator(i, j)),
            os.path.join(db_rec, img_name_generator(i, j)),
        )
