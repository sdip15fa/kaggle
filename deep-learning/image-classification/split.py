import os
import random

files = [i for i in os.listdir("train_img") if "jpg" in i]

files = sorted(files, key=lambda a: random.random())

valid = files[:int(len(files) * 0.2)]
train = files[int(len(files) * 0.2):]

for i in valid:
    os.rename(f"train_img/{i}", f"train_img/valid/{i}")

for i in train:
    os.rename(f"train_img/{i}", f"train_img/train/{i}")
