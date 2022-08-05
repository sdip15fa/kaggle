import pandas as pd
import os

df = pd.read_csv('train.csv')

for k in ["train", "valid"]:
    for i in df["character"].unique():
        try:
            os.mkdir(f"train_img/{k}/{i}")
        except:
            pass
        for j in df[df["character"] == i]["id"]:
            try:
                os.rename(f"train_img/{k}/{j}.jpg",
                          f"train_img/{k}/{i}/{j}.jpg")
                print(f"{j} -> {i}")
            except:
                pass
