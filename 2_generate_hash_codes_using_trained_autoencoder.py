import trained_AE_for_hash_code_generation as TAE
import os
import pathlib
import numpy as np
import pandas as pd

pathlib.Path('hash_codes').mkdir(parents=True, exist_ok=True)
X=np.array(pd.read_csv('features_and_pred_classes/preds.csv', header=None, delimiter=",")).astype(float)
hiddens=[[64]] #64 is number of neurons in the AE which is the same as the hash lenght
for i in range(len(hiddens)):
    hidden=hiddens[i]
    hash_codes=TAE.generate_hash_codes(X,hidden)
    np.savetxt('hash_codes/h_codes_%d_train.csv' % (hidden[0]), hash_codes, delimiter=",")


