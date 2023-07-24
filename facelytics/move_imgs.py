import shutil
import os
from tqdm import tqdm

from commons import img_name_generator


source_db_path = os.path.join("CelebA", "db-img")
ident_db = os.path.join("CelebA", "db-ident")
test_db = os.path.join("CelebA", "db-imgs")


def main():
    if not os.path.isdir(source_db_path):
        raise NotADirectoryError("Source directory not exsist")

    if not os.path.isdir(ident_db):
        os.mkdir(ident_db)
    if not os.path.isdir(test_db):
        os.mkdir(test_db)
    
    for i in tqdm(range(1, 1000 + 1)):
        for j in range(1, 3 + 1):
            shutil.copy(
                os.path.join(source_db_path, img_name_generator(i, j)),
                os.path.join(ident_db, img_name_generator(i, j)),
            )
        for k in range(4, 9 + 1):
            shutil.copy(
                os.path.join(source_db_path, img_name_generator(i, k)),
                os.path.join(test_db, img_name_generator(i, k)),
            )
if __name__ == "__main__":
    main()
