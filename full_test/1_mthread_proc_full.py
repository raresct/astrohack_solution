import pandas as pd
import numpy as np
from imagecleanup2 import cleanupImage as cleanupImage2
import cv2

df = pd.read_csv('Train.csv',sep=';')
df['lin_mass'] = np.power(10, df.logMstar)
df['lin_err'] = df.lin_mass * np.log(10) * df.err_logMstar

df = df[df.logMstar!=-99]
df = df[df.err_logMstar!=0]

N=len(df.SDSS_ID.values)

ids = df.SDSS_ID.values[:N]


print(len(ids))


Y = df.logMstar.values[:N]
err = df.err_logMstar.values[:N]
Y_lin = df.lin_mass.values[:N]
err_lin = df.lin_err.values[:N]

gids = ['Train/'+str(id)+'-g.csv' for id in ids]

def img_preproc(id):
    Xg = np.genfromtxt (id, delimiter=",")
    Xg = cleanupImage2(Xg)
    Xg -= np.mean(Xg)
    Xg /= np.std(Xg)
    h,w = Xg.shape
    cy, cx = h//2, w//2
    dy, dx = cy//2, cx//2
    Xg = Xg[cy-dy:cy+dy,cx-dx:cx+dx]
    Xgr = cv2.resize(Xg,(224,224),cv2.INTER_AREA)
    print('.',end='',flush=True)
    return Xgr

from joblib import Parallel, delayed
X_ = Parallel(n_jobs=40)(delayed(img_preproc)(i) for i in gids)

X = np.stack(X_)

np.save('Ximg',X)
