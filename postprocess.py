import os

import pandas as pd
import numpy as np
from skimage import io
from natsort import natsorted, ns
import pickle
from model.hist_image import hist_image
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from skimage.segmentation import  chan_vese
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.color import label2rgb

if __name__ == '__main__' :

    filename = 'hist_list_A00_v1.sav'
    hist_list = pickle.load(open(filename, 'rb'))

    k_hist = 0
    for hist in hist_list:

        filename = "y_pred_p_A00_v1_%d.sav" % k_hist
        y_pred_p = pickle.load(open(filename, 'rb'))

        filename = "y_A00_v1_%d.sav" % k_hist
        y = pickle.load(open(filename, 'rb'))

        plt.figure(1)
        A =  (y_pred_p>0.85)*1.0
        plt.matshow(A)
        plt.show()
        plt.figure(2)
        plt.matshow(y)
        plt.show()
    

        # label image regions
        label_image = label(A)
        fig, ax = plt.subplots(figsize=(10, 6))
        image_label_overlay = label2rgb(label_image, image=A, bg_label=0)
        ax.imshow(image_label_overlay)
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 60:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                a = A[rect._y0:rect._y1,rect._x0:rect._x1]
                ax.add_patch(rect)


        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

        k_hist += 1