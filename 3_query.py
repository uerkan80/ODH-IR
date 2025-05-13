import shutil
import os
import yolo
import trained_AE_for_hash_code_generation as TAE
import numpy as np
import pandas as pd

number_of_retrieved_images=20
img_path='./coco_images/validation'
query_img_names=['000000466835.jpg',
                 '000000114871.jpg',
                 '000000052891.jpg',
                 '000000255664.jpg',
                 '000000171611.jpg',
                 '000000190140.jpg'
                 ]
hidden=[64]

################################################
hash_size=hidden[0]
#getting the hash codes saved previously
train_hash=np.array(pd.read_csv('hash_codes/h_codes_%d_train.csv'%(hash_size), header=None, delimiter=","))
#getting the features saved previously
train_feature=np.array(pd.read_csv('features_and_pred_classes/features.csv', header=None, delimiter=","))

#getting the class names
classNames = list(np.array(pd.read_csv("./coco_labels/train.csv", header=None, delimiter=","))[0,1:])
#getting the train image names
train_image_names=np.array(pd.read_csv("./coco_labels/train.csv", header=0, delimiter=","))[:,0]

#getting the predicted labels and features of the query image via YOLO
query_preds, query_features=yolo.pred_and_features(classNames, query_img_names, img_path)

for i in range(len(query_img_names)):
    print(query_img_names[i])
    query_pred = query_preds[i]
    query_feature = query_features[i]
    #getting the hash codes of the query images via trained AE
    query_hash=TAE.generate_hash_codes(query_pred, hidden)
    print("query_pred_labels:", np.array(classNames)[np.nonzero(query_pred)[0]])

    #Computing the hamming distance between the hash codes of query images and trained images
    hash_distance=np.sum(np.bitwise_xor(np.array(query_hash, dtype=int),
                                    np.array(train_hash, dtype=int)),
                     axis=1)
    # Computing the euclidean distance between features of the query images and trained images
    feature_distance = np.sum((query_feature - train_feature) ** 2, axis=1)
    sorted_hash_distance=np.sort(hash_distance)
    checking_index_of_retrieved_images=number_of_retrieved_images
    j=1
    while sorted_hash_distance[j-1]==sorted_hash_distance[j]:
        j+=1
    if j>number_of_retrieved_images:
        checking_index_of_retrieved_images=j
    a = []
    a.append(np.arange(len(hash_distance)))
    a.append(hash_distance)
    a.append(feature_distance)
    a = np.array(a).T
    indices_of_retrieved_data = np.array(a[np.lexsort((a[:, 2], a[:, 1]))][:, 0], dtype=int)[:checking_index_of_retrieved_images]
    indices_of_retrieved_data=indices_of_retrieved_data[:number_of_retrieved_images]
    print('retrieved_image names:')
    for r in np.array(train_image_names)[indices_of_retrieved_data]:
        print(r)
    print('=================================================')


