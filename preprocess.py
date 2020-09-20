import os
import pickle

import pandas as pd
import numpy as np
from skimage import io
from natsort import natsorted
from model.hist_image import hist_image

if __name__ == '__main__' :

    #load all .png images and .csv files belonging to scanner A
    image_list = []
    mit_list = []
    dirs = [x[0] for x in os.walk('./data')]
    for d in dirs[1:-1]:
        if d[7] == 'A':
            list_files = os.listdir(d)
            list_files = natsorted(list_files)
            for filename in list_files:
                if filename[-3:] == 'png':
                    im = io.imread(d+'/'+filename, as_gray=True)
                    im = im/np.max(np.max(im))
                    image_list.append(im)


                if filename[-3:] == 'csv':
                    mit = pd.read_csv(d+'/'+filename,sep='\n', header=None)
                    x = mit[0].values
                    mit = []
                    for k in range(0,len(x)):
                        x0 = x[k].split(',')
                        x0 = np.array(list(map(int, x0)))
                        x0 = x0.reshape((int(len(x0)/2),2))
                        mit.append(x0)
                    mit_list.append(mit)

    # append all images as hist_images, add mitosis positions, generate (sub) samples
    hist_list = []
    X = np.array([0])
    Y = np.array([0])
    for i in range(0,len(mit_list)):
        im  =  image_list[i]
        mit =  mit_list[i]

        hist = hist_image(im)
        hist.load_mit(mit)

        hist.gen_mask()
        hist.gen_mit_pxls()
        hist.gen_non_mit_pxls()

        (x, y) = hist.im_dsamples(r = 0.05)
        X = np.concatenate((X, x),axis = 0)
        Y = np.concatenate((Y, y), axis=0)
        hist_list.append( hist)

    #train initial forest for thresholding intensity
    X = np.delete(X, 0, 0)
    Y = np.delete(Y, 0, 0)

    filename = './save/thresh_model_all_A_rf.sav'
    model = pickle.load(open(filename, 'rb'))

    for i in range(0, len(mit_list)):
        hist = hist_list[i]
        hist.gen_p_im(model)
        hist.gen_otsu()
        hist.gen_otsu_avg()

    filename = './save/hist_list_all_A_preprocessed.sav'
    pickle.dump(hist_list, open(filename, 'wb'))


