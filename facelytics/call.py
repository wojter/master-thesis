import os
import numpy as np

from commons import models

model_name = "DeepFace"
img_dir = "db-imgs_gaussian_"

if model_name not in models:
    raise ValueError("Wrong model name")

commands = []
for i in np.arange(0.05, 0.3, 0.05):
    command = (
        "python3 facelytics/test_model.py -m "
        + model_name
        + " -f "
        + img_dir
        + "{:.2f}".format(i).replace(".", "_")
    )
    commands.append(command)

for com in commands:
    try:
        print("\n\n")
        print("-" * 80)
        print("Running: ", com, "\n")
        os.system(com)
    except:
        pass
