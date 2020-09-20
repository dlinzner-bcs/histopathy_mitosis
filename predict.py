import os

import pandas as pd
import numpy as np
from skimage import io
from natsort import natsorted, ns
import pickle
from model.hist_image import hist_image
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from skimage.segmentation import  felzenszwalb

if __name__ == '__main__' :

    filename = 'hist_list_A00_v1.sav'
    hist_list = pickle.load(open(filename, 'rb'))

    filename = 'RF_A00_v1_model_invar_d20.sav'
    rf = pickle.load(open(filename, 'rb'))

    d = 30
    nf = 6
    ns = 10
    r = int(d / 2)
    k_hist = 0
    for hist in hist_list:
        L1 = hist.xlim - r
        L2 = hist.ylim - r

        mask_label = hist.get_mask()
        mask_pred = hist.get_otsu()
        mask_pred_smooth = hist.get_otsu_avg()

        k_max = sum(sum(mask_pred_smooth))
        print(k_max)
        print(L1 * L2)

        X = np.zeros(((L1 - 1) * (L2 - 1), nf * 4 + 1))
        Y = np.zeros(((L1 - 1) * (L2 - 1),))

        k = 0
        for i in range(r, L1):
            for j in range(r, L2):

                if mask_pred_smooth[i, j] == 1:

                    Y[k] = mask_label[i, j]

                    crop = hist.im[i - r:i + r, j - r:j + r]
                    v = hist.extract_feature(crop, nf, ns)
                    X[k, :] = v
                    k += 1
                    if k % 1000 == 0:
                        print(k)


        Y_pred_p = rf.predict_proba(X)
        Y_pred = rf.predict(X)

        y = np.zeros((L1 + r, L2 + r))
        y_pred = np.zeros((L1 + r, L2 + r))
        y_pred_p = np.zeros((L1 + r, L2 + r))
        k = 0
        for i in range(r, L1):
            for j in range(r, L2):
                if mask_pred_smooth[i, j] == 1:
                    y[i, j] = Y[k]
                    y_pred[i, j] = (1 - Y_pred_p[k, 0])>0.7
                    y_pred_p[i, j] = 1 - Y_pred_p[k, 0]
                    k += 1

        plt.figure(4)
        plt.matshow(y)
        plt.show()
        plt.figure(5)
        plt.matshow(y_pred)
        plt.show()
        plt.figure(6)
        plt.matshow(y_pred_p)
        plt.show()

        filename = "y_A00_v1_%d.sav" % k_hist
        pickle.dump(y, open(filename, 'wb'))
        filename = "X_A00_v1_%d.sav" % k_hist
        pickle.dump(X, open(filename, 'wb'))
        filename = "y_pred_p_A00_v1_%d.sav" % k_hist
        pickle.dump(y_pred_p, open(filename, 'wb'))
        segments = felzenszwalb(y_pred)

        k_hist += 1
