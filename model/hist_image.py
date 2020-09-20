import numpy as np
from skimage.filters import threshold_otsu
from copy import copy
from scipy.stats   import  skew
from scipy.stats   import  kurtosis
from scipy.ndimage import  rotate


class hist_image():
    def __init__(self,im):
        self.im = im
        self.xlim = im.shape[0]
        self.ylim = im.shape[1]
        self.mit_pos = []
        self.mask = np.ones((self.xlim,self.ylim))
        self.all_pxls = self.get_all_pxls()
        self.mit_pxls = []
        self.otsu_pxls = []
        self.non_mit_pxls = []
        self.X = []


        #self.gen_mask()
        #self.gen_mit_pxls()
        #self.gen_non_mit_pxls()


        self.otsu = []
        self.otsu_pos =[]
        self.otsu_avg = []
        self.otsu_avg_pos = []
        self.p_im = []
        self.candidate_mit_pxls = []

    def load_mit(self,mit_pos):
        self.mit_pos = mit_pos

    def gen_mask(self):
        M = np.zeros((self.xlim,self.ylim))
        mit_pos = self.mit_pos
        for i in range(0,len(mit_pos)):
            p_i = mit_pos[i]
            for j in range(0,len(p_i)):
                p = p_i[j,:]
                M[p[1],p[0]] = 1
        self.mask = M
        return None

    def im_dsamples(self,r):
        x = self.im.flatten()
        y = self.mask.flatten()

        indx = np.asarray([range(0,len(x))]).flatten()
        indx  = np.random.choice(indx, size = int(len(x)*r), replace=False)

        x_r = x[indx]
        y_r = y[indx]
        return (x_r,y_r)

    def gen_mit_pxls(self):
        M = self.mask
        post = np.multiply(M,self.im)
        v = post.flatten()
        v = v[v!=0]
        self.mit_pxls = v
        return None

    def gen_otsu_pxls(self):
        M = self.otsu
        post = np.multiply(M,self.im)
        v = post.flatten()
        v = v[v!=0]
        self.otsu_mit_pxls = v
        return None

    def gen_non_mit_pxls(self):
        unit = np.ones((self.xlim,self.ylim))
        M = unit - self.mask
        post = np.multiply(M, self.im)
        v = post.flatten()
        v = v[v != 0]
        self.non_mit_pxls = v
        return None

    def gen_p_im(self,model):
        v = self.get_all_pxls()
        X =  np.atleast_2d(v)

        v_p = model.predict_proba(X.T)
        p_im = np.reshape(v_p[:,1], (self.xlim,self.ylim), order='C')
        self.p_im =  p_im
        return None

    def get_all_pxls(self):
        v = self.im
        return v.flatten()

    def get_mit_pxls(self):
        v = self.mit_pxls
        return v.flatten()

    def get_im(self):
        return self.im

    def get_p_im(self):
        return self.p_im

    def get_mask(self):
        return self.mask

    def gen_otsu(self):
        image = self.p_im
        thresh = threshold_otsu(image)
        binary = image > thresh
        self.otsu = np.array(binary.astype(int))
        self.otsu_pos = np.where(self.otsu==1)
        return None

    def get_otsu(self):
        return self.otsu

    def gen_otsu_avg(self):
        #TODO: speed up using cython
        M  = np.zeros((self.xlim,self.ylim))
        M0 = copy(self.otsu)
        r = 5
        for i in range(r,self.xlim-r):
            for j in range(r, self.ylim-r):
                b  = M0[i-r:i+r+1,j-r:j+r+1]
                l = sum(sum(b))
                M[i,j]=int(l>((r*r-1)/2))
               # if (b[1,1]==1)*(l==1):
               #     M[i,j]=0
        self.otsu_avg = M
        self.otsu_avg_pos = np.argwhere(self.otsu_avg == 1)

        k_max = (1-sum(sum(M))/(self.xlim*self.ylim))*100
        print('pre-processing reduced image by %d percent' % int(k_max))
        return None

    def gen_otsu_avg_pos(self):
        self.otsu_avg_pos = np.argwhere(self.otsu_avg == 1)
        return None

    def get_otsu_avg(self):
        return self.otsu_avg

    def get_stratified_samples(self,N,r):
        v_mit = self.mit_pxls
        v_non_mit = self.non_mit_pxls

        samples = []
        for i in range(0,N):
            if np.random.uniform(0,1,1)>r:
                pxl = v_mit[np.random.randint(0,len(v_mit))]
            else:
                pxl = v_non_mit[np.random.randint(0, len(v_non_mit))]
            samples.append(pxl)

        return np.asarray(samples)

    def get_sample_crop(self,d):
        r = int(d / 2)
        im = self.im
        i = np.random.randint(0, self.xlim)
        j = np.random.randint(0, self.xlim)
        sample = im[i - r:i + r, j - r:j + r]
        return sample

    def get_sample_otsu_crop(self, d):
        r = int(d / 2)
        im = self.im
        mit_pos = self.otsu_avg_pos
        ind1 = np.random.randint(0,mit_pos.shape[0])
        p = (mit_pos[ind1,0],mit_pos[ind1,1])
        i, j = p[0], p[1]
        crop = im[i - r:i + r, j - r:j + r]
        sample = (crop, self.mask[i,j])
        return sample


    def get_mit_sample_crop(self,d):
        r = int(d / 2)
        im = self.im
        mit_pos = self.mit_pos
        ind = np.random.randint(0, len(mit_pos))
        p_m = mit_pos[ind]
        ind = np.random.randint(0, p_m.shape[0])
        p = p_m[ind, :]
        i, j = p[1], p[0]
        crop = im[i - r:i + r, j - r:j + r]
        sample = (crop, self.mask[i,j])
        return sample

    def extract_feature(self,sample,nf,ns):
        crop = sample
        r = int(crop.shape[0]/2)
        v = np.zeros(shape=(nf, ns))
        for s in range(0, ns):
            angle = s * 180 / ns
            data = rotate(crop, angle)
            slice = data[r, :]
            slice = np.delete(slice, np.where(slice == 0), axis=0)
            v[0, s] = np.mean(slice)
            v[1, s] = np.var(slice)
            slice = slice
            slice = slice / sum(slice)
            v[2, s] = -np.sum(np.multiply(slice, np.log(slice)))
            v[3, s] = skew(slice)
            v[4, s] = kurtosis(slice)
            v[5, s] = np.sum(np.multiply(np.arange(-1, 1, len(slice)), slice))

        w = np.zeros(shape=(nf * 4,))
        h = 0
        for f in range(0, nf - 1):
            w[h] = np.mean(v[f, :])
            h += 1
            w[h] = np.var(v[f, :])
            h += 1
            w[h] = skew(v[f, :])
            h += 1
            w[h] = kurtosis(v[f, :])
            h += 1

        v = w.flatten()
        pxl = (crop[r, r],)
        v = np.concatenate((v, pxl), axis=0)
        return v

    def extract_design(self,d):

        nf = 6
        ns = 10
        r = int(d / 2)

        L1 = self.xlim - r
        L2 = self.ylim - r

        mask_pred_smooth = self.get_otsu_avg()

        X = np.zeros(((L1 - 1) * (L2 - 1), nf * 4 + 1))
        k = 0
        for i in range(r, L1):
            for j in range(r, L2):
                if mask_pred_smooth[i, j] == 1:
                    crop = self.im[i - r:i + r, j - r:j + r]
                    v = self.extract_feature(crop, nf, ns)
                    X[k, :] = v
                    k += 1
                    if k % 1000 == 0:
                        print('%d percent processed' % int(100*k/(sum(sum(mask_pred_smooth)))))
        self.X = X
        return None
