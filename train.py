import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__' :

    filename = './save/hist_list_all_A_preprocessed.sav'
    hist_list = pickle.load(open(filename, 'rb'))

    d = 30 #rectangle length in pxl
    r = int(d / 2)

    nf = 6 #number of features per slice
    ns = 10 #number of slices
    N_s = 100000 #number of samples for training
    X = np.zeros((N_s, nf * 4 + 1))
    Y = np.zeros((N_s,))
    k = 0
    for i in range(0 ,N_s):
        try:
            hist_id = np.random.randint(0 ,len(hist_list))
            hist = hist_list[hist_id]
            sample = hist.get_sample_otsu_crop(d)
            crop = sample[0]
            y    = sample[1]
            Y[i] = y
            v = hist.extract_feature(crop, nf, ns)
            X[i, :] = v
            if (np.sum(np.isnan(v))+np.sum(np.isinf(v))):
                hist_id = np.random.randint(0, len(hist_list))
                hist = hist_list[hist_id]
                sample = hist.get_sample_otsu_crop(d)
                crop = sample[0]
                y = sample[1]
                Y[i] = y
                v = hist.extract_feature(crop, nf, ns)
                X[i, :] = v

            if i % 1000 == 0:
                print(i)
        except:
            i -= 1
    print(sum(Y)/N_s)
    print(sum(sum(hist.mask)) / sum(sum(hist.get_otsu_avg())))
    #train random forest classifier
    rf = RandomForestClassifier(max_depth=20, random_state=0)
    rf.fit(X, y=Y)

    #save trained model
    filename = './save/RF_A_all_N100000_depth20.sav'
    pickle.dump(rf, open(filename, 'wb'))