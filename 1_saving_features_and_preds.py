import yolo
import numpy as np
import pandas as pd
import os
os.makedirs("features_and_pred_classes", exist_ok=True)

img_path='./coco_images/train'
#getting the class names
classNames = list(np.array(pd.read_csv("./coco_labels/train.csv", header=None, delimiter=","))[0,1:])
#getting the image names
train_image_names=np.array(pd.read_csv("./coco_labels/train.csv", header=0, delimiter=","))[:,0]
preds, features=yolo.pred_and_features(classNames, train_image_names, img_path)

np.savetxt('./features_and_pred_classes/features.csv', features, delimiter=",")
np.savetxt('./features_and_pred_classes/preds.csv', preds, delimiter=",")
