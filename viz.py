# -*- coding: utf-8 -*-
"""viz.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ah09IlKH-nvT7Hd5ViWng9iql6BTzeym
"""

import pickle
import matplotlib.pyplot as plt

NUM_CLIENTS = 3

with open("results/test.pkl", "rb") as f:
    res = pickle.load(f)

for idx in range(NUM_CLIENTS):
    with open(f"results/{idx}.pkl", "rb") as f:
        res[idx] = pickle.load(f)


plt.plot(res["mAP"], label="server")
for idx in range(NUM_CLIENTS):
    plt.plot(res[idx]["mAP"], label=f"client {idx}")
plt.xlabel("Round")
plt.ylabel("mAP")
plt.legend()
plt.grid()
plt.show()

plt.plot(res["recall"], label="server")
for idx in range(NUM_CLIENTS):
    plt.plot(res[idx]["recall"], label=f"client {idx}")
plt.xlabel("Round")
plt.ylabel("Recall")
plt.grid()
plt.show()

plt.plot(res["precision"], label="server")
for idx in range(NUM_CLIENTS):
    plt.plot(res[idx]["precision"], label=f"client {idx}")
plt.xlabel("Round")
plt.ylabel("Precision")
plt.grid()
plt.show()

for idx in range(NUM_CLIENTS):
    plt.plot(res[idx]["loss"], label=f"client {idx}")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid()
plt.show()

from glob import glob
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def show_image(image_path, result):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = result.xyxy.cpu().numpy()
    labels = result.cls.cpu().numpy().astype(int)
    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = list(map(int, box))
        if label == 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    plt.imshow(image)
    plt.axis("off")
    plt.show()

model = YOLO("runs/0/train/weights/best.pt")
file_paths = glob("data/clients/test/val/*.jpg")
image_path = random.choice(file_paths)
results = model.predict(image_path, save=False, imgsz=480, conf=0.5)
show_image(image_path, results[0].boxes)