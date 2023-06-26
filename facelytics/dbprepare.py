import pandas as pd
import os

total_images = 202_599
total_identities = 10_177
min_images_of_person = 8

identity_path = "./CelebA/Anno/identity_CelebA.txt"

assert os.path.isfile(identity_path), f"Not find identity file at path {identity_path}"

colnames = ['file', 'identity']
identities = pd.read_csv(identity_path, sep=" ", names=colnames, header=None)

assert identities['file'].nunique() == total_images, "Import error, wrong total img number"
assert identities['identity'].nunique() == total_identities, "Import error, wrong total identities number"
assert identities.isna().values.any() == False, "NaN in table detected"

temp = identities.drop("file", axis=1)
temp = temp.groupby('identity').size().reset_index(name='occurences')
temp = temp.drop(temp.index[temp["occurences"] < min_images_of_person])

print(temp)
