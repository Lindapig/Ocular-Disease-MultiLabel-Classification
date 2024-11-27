import os
import pandas
from utils import read_file, list_image_files

IMAGE_PATH = "/media/wang/data/proj/ODIR/data/preprocessed_images/"
IMAGE_INFO = "/media/wang/data/proj/ODIR/data/full_df.csv"

image_names = list_image_files(IMAGE_PATH)
preprocess_image_df = read_file(IMAGE_INFO)
left_image_ids = [
    int(item.split(".")[0].split("_")[0])
    for item in image_names
    if item.split(".")[0].split("_")[-1] == "left"
]
right_image_ids = [
    int(item.split(".")[0].split("_")[0])
    for item in image_names
    if item.split(".")[0].split("_")[-1] == "right"
]
left_image_ids.sort()
right_image_ids.sort()
common_image_ids = list(set(left_image_ids) & set(right_image_ids))


select_columns = [
    "ID",
    "Patient Age",
    "Patient Sex",
    "Left-Fundus",
    "Right-Fundus",
    "Left-Diagnostic Keywords",
    "Right-Diagnostic Keywords",
    "N",
    "D",
    "G",
    "C",
    "A",
    "H",
    "M",
    "O",
]
preprocess_image_df = preprocess_image_df[select_columns]
filtered_df = preprocess_image_df[preprocess_image_df["ID"].isin(common_image_ids)]

filtered_df = filtered_df.sort_values(by="ID")
filtered_df = filtered_df.drop_duplicates(subset=filtered_df.columns.difference(["ID"]))
filtered_df.to_csv("/media/wang/data/proj/ODIR/data/filtered_full_df.csv", index=False)
