import numpy as np
import pandas as pd
import cv2
def global_avg_pooling(feature_map):
    # OpenCV DNN shape (N, C, H, W)
    return np.mean(feature_map, axis=(2, 3))  # (N, C)

def pred_and_features(classNames, img_names, img_path):
    model = cv2.dnn.readNetFromDarknet("yolov4-p6.cfg","yolov4-p6.weights")
    layers = model.getLayerNames()
    unconnect = model.getUnconnectedOutLayers()
    unconnect = unconnect-1
    output_layers = []
    feature_layers = []
    for i in unconnect:
        #gettin unconnected layers for prediction
        output_layers.append(layers[int(i)])
        #getting last CONV layers before the unconnected layers for features
        feature_layers.append(layers[int(i)-3])
    pred_list=[]
    feature_list=[]
    for i in range(len(img_names)):
        img_name=img_names[i]
        print("extracting features and predicting the objects of image %s"%(img_name,))
        img = cv2.imread('%s/%s'%(img_path,img_name))
        img_width = img.shape[1]
        img_height = img.shape[0]
        img_blob = cv2.dnn.blobFromImage(img,1/255,(1280,1280),swapRB=True)
        model.setInput(img_blob)

        detection_layers = model.forward(output_layers)
        features=model.forward(feature_layers)

        # apply pooling to all features
        vectors = [global_avg_pooling(out) for out in features]

        # vector concatenation
        combined_vector = np.concatenate(vectors, axis=1)  # (N, 1360)

        # Normalizing
        normalized_vector = combined_vector / np.linalg.norm(combined_vector, axis=1, keepdims=True)
        feature_list.append(normalized_vector[0])
        ids_list = []
        boxes_list = []
        confidences_list = []
        for detection_layer in detection_layers:
            for object_detection in detection_layer:
                scores = object_detection[5:]
                predicted_id = np.argmax(scores)
                confidence =scores[predicted_id]


                if confidence > 0.10:

                    label = classNames[predicted_id]
                    bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                    (box_center_x, box_center_y ,box_width ,box_height) = bounding_box.astype("int")
                    start_x = int(box_center_x- (box_width/2))
                    start_y = int(box_center_y - (box_height/2))

                    ids_list.append(predicted_id)
                    confidences_list.append(float(confidence))
                    boxes_list.append([start_x,start_y,int(box_width),int(box_height)])

        max_ids = cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)
        labels=[]
        label_index=[]
        for max_id in max_ids:
            max_class_id=max_id
            predicted_id = ids_list[max_class_id]
            label = classNames[predicted_id]
            confidence=confidences_list[max_class_id]
            labels.append(label)
            label_index.append(classNames.index(label))

        labels=list(set(labels))
        label_index=np.array(list(set(label_index)))
        pred_vector=np.zeros(len(classNames))
        if len(label_index)>0:
            pred_vector[label_index]=np.ones(len(labels))
        pred_list.append(pred_vector)
    return pred_list, feature_list