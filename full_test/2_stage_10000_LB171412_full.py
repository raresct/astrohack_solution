
import pandas as pd
import numpy as np
import cv2
import lightgbm as lgbm
import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
from keras.applications import *
from imagecleanup2 import cleanupImage as cleanupImage2
#import lightgbm as lgbm

r50 = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
#xc299 = Xception(include_top=True, weights='imagenet', input_shape=(299,299,3))
#xc299nf = Xception(include_top=False, weights='imagenet', input_shape=(299,299,3))
#incv3 = InceptionV3(include_top=True, weights='imagenet', input_shape=(299,299,3))
#vgg16nf = VGG16(weights='imagenet',include_top=True,input_shape=(224,224,3))
vgg16 = VGG16(weights='imagenet',include_top=True,input_shape=(224,224,3))
#vgg19nf = VGG19(weights='imagenet',include_top=True,input_shape=(224,224,3)) 
vgg19 = VGG19(weights='imagenet',include_top=True,input_shape=(224,224,3))

df = pd.read_csv('Train.csv',sep=';')
df['lin_mass'] = np.power(10, df.logMstar)
df['lin_err'] = df.lin_mass * np.log(10) * df.err_logMstar

df = df[df.logMstar!=-99]
df = df[df.err_logMstar!=0]
np.random.seed(0)

N=len(df.SDSS_ID.values)
M=N-4000
ids = df.SDSS_ID.values[:N]
print(len(ids))
Y = df.logMstar.values[:N]
err = df.err_logMstar.values[:N]
Y_lin = df.lin_mass.values[:N]
err_lin = df.lin_err.values[:N]

gids = ['Train/'+str(id)+'-g.csv' for id in ids]
print(len(gids))

print('loading Xg')
Xg = np.load('Ximg.npy')

print(Xg.shape)
print(np.min(Xg),np.max(Xg))

print('reshaping')
Xg3 = np.zeros((N,224,224,3))
Xg3[:,:,:,:] = Xg.reshape(N,224,224,1)
print(np.min(Xg3),np.max(Xg3))

print('r50')
Xg3r50 = r50.predict(Xg3).reshape(N,2048)
print('vgg16')
Xg3vgg16 = vgg16.predict(Xg3)
print('vgg19')
Xg3vgg19 = vgg19.predict(Xg3)
print('done')

print('Features X g band 3 ch features')

Distance = df.Distance.values[:N].reshape(N,1)

csize = 2

Xg3f = np.hstack ( ( 
    Xg3r50, 
    Xg3vgg16, 
#    Xg3vgg19,
    Distance,
    1/Distance,
    Distance**2,
    1/(Distance**2),
    Distance**3,
    1/(Distance**3),
    np.log(Distance),
    1/np.log(Distance),
    np.log(Distance**2),
    1/np.log(Distance**2),
    np.log(Distance)**2,
    1/np.log(Distance)**2,
    np.sum(Xg3.reshape(N,-1),axis=1).reshape(N,1),
    np.min(Xg3.reshape(N,-1),axis=1).reshape(N,1),
    np.max(Xg3.reshape(N,-1),axis=1).reshape(N,1),
    np.mean(Xg3.reshape(N,-1),axis=1).reshape(N,1),
    np.std(Xg3.reshape(N,-1),axis=1).reshape(N,1),
    Xg3[:,112,112,0].reshape(N,1),       # center
    np.mean(Xg3[:,112-csize:112+csize,112-csize:112+csize,0].reshape(N,-1),axis=1).reshape(N,-1), # mean center
) )


print(Xg3r50.shape)
print(Xg3vgg16.shape)
print(Xg3f.shape)

np.save('Xg3f',Xg3f)

dtrain = lgbm.Dataset(Xg3f[:M], label= Y[:M])
dtest = lgbm.Dataset(Xg3f[M:], label= Y[M:])


lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'nthread': 35,
    'silent': True,
    'num_leaves': 2**4,
    'learning_rate': 0.05,
    'max_depth': 10,
    'max_bin': 255,
    #'subsample_for_bin': 50000,
    #'subsample': 0.8,
    #'subsample_freq': 1,
    #'colsample_bytree': 0.8,
    #'reg_alpha': 1,
    #'reg_lambda': 0,
    #'min_split_gain': 0.5,
    #'min_child_weight': 1,
    #'min_child_samples': 60,
    #'scale_pos_weight': 1,
    #'device' : 'gpu',
    'metric' : 'rmse',
    #'eval_metric' : 'rmse',
    #'metric' : 'multi_error',
    'verbose':0,          
}

bst = lgbm.cv(lgbm_params, dtrain, num_boost_round=10000, data_splitter=None, nfold=3, stratified=False, shuffle=True, 
              metrics=None, fobj=None, feval=None, init_model=None, feature_name='auto', 
              categorical_feature='auto', early_stopping_rounds=200, fpreproc=None, 
              verbose_eval=10, show_stdv=True, seed=0, callbacks=None)

num_boost_round = len(bst['rmse-mean'])-1
print(num_boost_round)


model = lgbm.train(lgbm_params, dtrain, num_boost_round,
                   valid_sets=[dtest], valid_names=['test'], fobj=None, feval=None, 
                   init_model=None, feature_name='auto', categorical_feature='auto', 
                   early_stopping_rounds=num_boost_round, evals_result=None, verbose_eval=10, 
                   learning_rates=None, callbacks=None)

pred = model.predict(Xg3f[M:])

def xi2(true,pred,error):
    s=np.mean((true-pred)**2/error**2)
    return s

print('xi2',xi2(Y[M:],pred,err[M:]))
xi2lin = xi2(10**Y[M:],10**pred,err_lin[M:])
print('xi2lin',xi2lin)

model.save_model('lgbm'+str(xi2lin), num_iteration=-1)

### TEST

df_test = pd.read_csv('Test_Distance.csv',sep=';')
#df_valid = pd.read_csv('validationdata_SDSSID.csv',sep=';')
#ids = df_test[df_test.SDSS_ID.isin(df_valid.SDSS_ID)]['SDSS_ID']
ids = df_test.SDSS_ID

gids = ['Test/'+str(id)+'-g.csv' for id in ids]

Xg_,Xi_ = [],[]
for i in range(len(ids)):
    Xg = np.genfromtxt (gids[i], delimiter=",")
    Xg = cleanupImage2(Xg)
    Xg -= np.mean(Xg)
    Xg /= np.std(Xg)
    h,w = Xg.shape
    cy, cx = h//2, w//2
    dy, dx = cy//2, cx//2
    Xg = Xg[cy-dy:cy+dy,cx-dx:cx+dx]
    Xgr = cv2.resize(Xg,(224,224),cv2.INTER_AREA)
    Xg_.append(Xgr)
    if i%10==0:
        print(i,end=' ',flush=True)

N = len(ids)
Xg = np.stack(Xg_)
Xg3 = np.zeros((N,224,224,3))
Xg3[:,:,:,:] = Xg.reshape(N,224,224,1)
print('r50')
Xg3r50 = r50.predict(Xg3).reshape(N,2048)
print('vgg16')
Xg3vgg16 = vgg16.predict(Xg3)
#Xg3vgg19 = vgg19.predict(Xg3)
#Distance = df_test[df_test.SDSS_ID.isin(ids)].Distance.values.reshape(N,1)
Distance = df_test.Distance.values.reshape(N,1)

csize = 2

Xg3f = np.hstack ( (
        Xg3r50,
        Xg3vgg16,
#        Xg3vgg19,
        Distance,
        1/Distance,
        Distance**2,
        1/(Distance**2),
        Distance**3,
        1/(Distance**3),
        np.log(Distance),
        1/np.log(Distance),
        np.log(Distance**2),
        1/np.log(Distance**2),
        np.log(Distance)**2,
        1/np.log(Distance)**2,
        np.sum(Xg3.reshape(N,-1),axis=1).reshape(N,1),
        np.min(Xg3.reshape(N,-1),axis=1).reshape(N,1),
        np.max(Xg3.reshape(N,-1),axis=1).reshape(N,1),
        np.mean(Xg3.reshape(N,-1),axis=1).reshape(N,1),
        np.std(Xg3.reshape(N,-1),axis=1).reshape(N,1),
        Xg3[:,112,112,0].reshape(N,1),       # center
        np.mean(Xg3[:,112-csize:112+csize,112-csize:112+csize,0].reshape(N,-1),axis=1).reshape(N,-1), # mean center

        ) )


del Xg3r50
del Xg3vgg16
del Xg3

dtest_final = lgbm.Dataset(Xg3f)
pred = model.predict(Xg3f)
df_sub = pd.DataFrame({'pssid':ids, 'mass':pred}, columns=['pssid', 'mass'])
df_sub.to_csv('submission_gold_'+str(xi2lin)+'.csv', index=False)

