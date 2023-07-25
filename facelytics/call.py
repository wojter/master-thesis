import os
import numpy as np

from commons import models

model_name = "DeepFace"
img_dir = "db_imgs_gaussian_"


if model_name not in models:
    raise ValueError("Wrong model name")


img_dir = "db_imgs_gaussian_"
for i in np.arange(0.05, 0.3, 0.05):
    command = (
        "python3 facelytics/test_model.py -m "
        + model_name
        + " -f "
        + img_dir
        + "{:.2f}".format(i).replace(".", "_")
    )
    try:
        print("\n\n")
        print("-" * 80)
        print("Running: ", command, "\n")
        os.system(command)
    except:
        pass
