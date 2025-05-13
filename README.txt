Prerequisites Before Running the Code:

1. The training images must be placed in the directory .\coco_images\train, and the query images must be placed in .\coco_images\validation.

2. The file train.csv, located in the ./coco_labels directory, must contain the object names of the COCO dataset on its first line.

3. The filenames of the training images must exactly match the names listed in the first column of the train.csv file in the ./coco_labels directory.


Running the Code:

1. The script 1_saving_features_and_preds.py processes the images located in the .\coco_images\train directory. It uses the YOLO.py file to extract predicted objects and visual features from each image and then saves the results.

2. The script 2_generate_hash_codes_using_trained_autoencoder.py generates hash codes for the training images using the trained_AE_for_hash_code_generation.py script. This script utilizes the weights and biases located in the ./AE_weights_and_biasses directory, along with the predicted labels saved in step 1, and then saves the resulting hash codes.

3. The script 3.query.py performs the following steps:
   3.1. It generates the predicted objects and features of the query images using the YOLO.py file.
   3.2. It generates the hash codes of the query images using trained_AE_for_hash_code_generation.py, which utilizes the weights and biases from the ./AE_weights_and_biasses directory and the predicted labels of the query images.
   3.3. It retrieves the most relevant training images by comparing the Hamming distance between hash codes and the Euclidean distance between image features.