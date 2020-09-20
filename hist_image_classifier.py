import numpy as np
from skimage import io
import pickle
import pandas as pd
from model.hist_image import hist_image
import matplotlib.pyplot as plt
from copy import copy

from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.color import label2rgb

if __name__ == '__main__' :
    #load image
    filename = './data/A04_v2/A04_01.png'
    im = io.imread(filename, as_gray=True)
    im = im / np.max(np.max(im))
    hist = hist_image(im)
    plt.figure(1)
    plt.matshow(im)
    plt.show()

    #load pre-process forest
    model = pickle.load(open('./save/thresh_model_all_A_rf.sav', 'rb'))

    #pre-select
    hist.gen_p_im(model)
    hist.gen_otsu()
    hist.gen_otsu_avg()

    plt.figure(2)
    plt.matshow(hist.get_otsu_avg())
    plt.show()
    # load prediction forest
    filename = './save/RF_A_all_N100000_depth20.sav'
    rf = pickle.load(open(filename, 'rb'))

    # predict with forest
    d = 30
    hist.extract_design(d)
    Y_pred = rf.predict(hist.X)

    Y_pred_p = rf.predict_proba(hist.X)
    Y_pred = rf.predict(hist.X)

    # reshape prediction
    L1= hist.xlim
    L2 = hist.ylim
    mask_pred_smooth = hist.get_otsu_avg()
    r =15
    y = np.zeros((L1 + r, L2 + r))
    y_pred = np.zeros((L1 , L2 ))
    y_pred_p = np.zeros((L1 , L2 ))
    k = 0
    for i in range(r, L1):
        for j in range(r, L2):
            if mask_pred_smooth[i, j] == 1:
                y_pred_p[i, j] = 1 - Y_pred_p[k, 0]
                k += 1

    A = (y_pred_p > 0.5)*1.0
    #plot
    plt.figure(4)
    plt.matshow(y)
    plt.show()
    plt.figure(5)
    plt.matshow(A)
    plt.show()
    plt.figure(6)
    plt.matshow(y_pred_p)
    plt.show()

    #post process for extraction of regions
    A_r = A[r:(L1+r),r:(L2+r)]
    im_post = copy(im)

    # label image regions
    label_image = label(A)
    fig, ax = plt.subplots(figsize=(10, 6))
    image_label_overlay = label2rgb(label_image, image=im_post, bg_label=0)
    ax.imshow(image_label_overlay)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 60:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig('./predictions/A04_01_prediction.png')
    plt.show()
