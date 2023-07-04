import os
import argparse
import time
from deepface.DeepFace import represent

db_recognition_path = "CelebA\img_db_recognition"
db_detection_path = "CelebA\img_db_detection"

models = {
    "VGG-Face",
    "OpenFace",
    "Facenet",
    "Facenet512",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace"
}
# TODO skip backend


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


if __name__ == "__main__":
    args = args_inpust()
    model_name, num_img_as_ref, num_img_of_ident, num_ident = args_parser(args)
    if model_name not in models:
        raise ValueError("Wrong model name given. Pass correct name")
    
    print("GO")
    start_recognition_time = time.time()
    res = represent(os.path.join(db_recognition_path, "00001_03.jpg"), model_name=model_name, detector_backend="skip")
    print(res)
    print("Total time", time.time() - start_recognition_time)