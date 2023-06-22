import pandas as pd
import os

total_images = 202_599
total_identities = 10_177

identity_path = "./CelebA/Anno/identity_CelebA.txt"

assert os.path.isfile(identity_path), f"Not find identity file at path {identity_path}"

colnames = ['file', 'identity']
identities = pd.read_csv(identity_path, sep=" ", names=colnames, header=None)

assert identities['file'].nunique() == total_images, "Import error, wrong total img number"
assert identities['identity'].nunique() == total_identities, "Import error, wrong total identities number"
# print(identities)
print(identities.isna().any())
