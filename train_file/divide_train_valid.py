import os
import shutil
import pandas as pd
import numpy as np
import scipy.io
from scipy.io import loadmat


ignore_list = "G:/Face age estimation/imdb_data/face_estimate/source_file/ignore_list.txt"
appa_dir = "G:/Face age estimation/appa-real-release"
output_dir = "G:/Face age estimation/appa-real-release"
img_dir = []
img_age = []
img_name = []

train_dir = os.path.join(appa_dir, "train")
valid_dir = os.path.join(appa_dir, "valid")

gt_train_path = os.path.join(appa_dir, "gt_avg_train.csv")
gt_valid_path = os.path.join(appa_dir, "gt_avg_valid.csv")

df_train = pd.read_csv(gt_train_path)
df_valid = pd.read_csv(gt_valid_path)

data_set = os.path.join(output_dir, "data")


with open(ignore_list, "r") as f:
    ignore_image_names = f.readlines()

print(len(ignore_image_names))

ignore_img = ""

for i in ignore_image_names:
    ignore_img += i


for i, row in df_train.iterrows():
    if row.file_name in ignore_img:
        continue
    else:
        age = min(100, int(row.real_age))
        dir = os.path.join(train_dir, row.file_name + "_face.jpg")
        name = row.file_name + "_face.jpg"
        img_name.append(name)
        img_age.append(age)
        img_dir.append(dir)

output_1 = {"full_path": img_dir, "age": np.array(img_age)}
output_path_1 = os.path.join(output_dir, "train_{}_inf.mat".format("finetune"))
scipy.io.savemat(output_path_1, output_1)

img_dir = []
img_age = []
img_name = []
for i, row in df_valid.iterrows():
    if str(row.file_name) in ignore_img:
        continue
    else:
        age = min(100, int(row.real_age))
        dir = os.path.join(valid_dir, row.file_name + "_face.jpg")
        name = row.file_name + "_face.jpg"
        img_name.append(name)
        img_age.append(age)
        img_dir.append(dir)

print(len(img_dir))
print(img_dir[0])
print(len(img_age))
print(img_age[0])
print(len(img_name))
print(img_name[0])


#
output_2 = {"full_path": img_dir, "age": np.array(img_age)}
output_path_2 = os.path.join(output_dir, "valid_{}_inf.mat".format("finetune"))
scipy.io.savemat(output_path_2, output_2)






