#!/usr/bin/env python
# coding: utf-8

# In[219]:


import re
import numpy as np
import pandas as pd

# In[2]:
GPXfile='Lunch_Walk.gpx'
data = open(GPXfile).read()

# In[3]:


lat = np.array(re.findall(r'lat="([^"]+)',data),dtype=float)
lon = np.array(re.findall(r'lon="([^"]+)',data),dtype=float)
time = re.findall(r'<time>([^\<]+)',data)
dt = pd.to_datetime(time)
s = dt.astype(int)

coords = np.array(list(zip(lat,lon,time,dt,s)))
coords = coords[::1]

# In[4]:
print ('coords')

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
from datetime import datetime
#

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_summary_fmt("notebook")


# In[461]:



Xi = coords[:, -2].reshape(-1, 1)#dt

Y0 = coords[:, 0].reshape(-1, 1).astype(np.float64)
Y1 = coords[:, 1].reshape(-1, 1).astype(np.float64)

Yi=np.array(list(zip(Y0,Y1))).astype(np.float64)


Xi = Xi



xx = [Xi[0],Xi[-1]]
Xi = Xi-xx[0]
Xi /= xx[1]-xx[0]



second = 1000000000

resolution = 0.0008
timegap = 60*second

Xarr = []
Yarr = []
Yold = [-999,-999]
Xold = -999
for j in range(len(Xi)):
    i = Yi[j][0][0]
    k = Yi[j][1][0]
    if abs(i-Yold[0])>resolution:
        Yarr.append([i,k])
        Yold=[i,k]
        Xold = Xi[j]
        Xarr.append(Xi[j])
    elif abs(Xi[j]-Xold)>timegap:
        Yarr.append([i,k])
        Yold=[i,k]
        Xold = Xi[j]
        Xarr.append(Xi[j])
    elif abs(k-Yold[1])>resolution:
        Yarr.append([i,k])
        Yold=[i,k]
        Xold = Xi[j]
        Xarr.append(Xi[j])
 


Xarr= np.array(Xarr).reshape(-1, 1).astype(np.float64)
Yarr= np.array(Yarr)#.reshape(-1, 1).astype(np.float64)




means=[]
scales = []
for q in [0,1]:

    scale = 1/Yarr[:,q].std()
    Yarr[:,q]=Yarr[:,q]*scale #1/resolution
    scales.append(scale)
    
    Ym = Yarr[:,q].mean()
    Yarr[:,q] = Yarr[:,q]-Ym
    means.append(Ym)
    


X=Xarr
Y=Yarr


# In[ ]:





# In[462]:


'''
FIX 
'''
start = (X[0]*(xx[1]-xx[0]))+xx[0]
end =  (X[-1]*(xx[1]-xx[0]))+xx[0]

start,end


def gx(x):
    return [(i*(xx[1]-xx[0]))+xx[0] for i in x]


# In[463]:


'''
FIY 
'''
scales = np.array(scales)
means  = np.array(means)


def gy(y):
    return np.array([(i+means)/scales for i in y])


scales , means,'v1',Y[10:12], 'v2',Yi[10:12],f,'','vout', gy(Y[10:12])


# In[ ]:





# In[464]:


import matplotlib.pyplot as plt

Ynew = gy(Y)


for z in [0,1]: 
    plt.figure(figsize=(15,3))

    _ = plt.plot(gx(Xi), Yi[:,z], ".", mew=.0005)

    plt.plot(gx(X), Ynew[:,z], "o", mew=.005)


    plt.xlabel('Time', horizontalalignment='right', x=1.0)
    plt.ylabel(['Latitude','Longitude'][z], horizontalalignment='right', y=1.0)
    plt.tight_layout()


# In[459]:


##########
#data


###########
# MAXITER = ci_niter(2000)
# D = 1  # number of input dimensions
# P = len(Y[0])  # number of observations = output dimensions
# M = len(Y)  # number of inducing points

# Zinit = np.linspace(0, 1, M)[:, None]



kernel =  gpflow.kernels.Matern52( variance=.1, lengthscales=.2)+gpflow.kernels.Matern52( variance=.1, lengthscales=.2)+ gpflow.kernels.White()# +gpflow.kernels.Exponential() 

m = gpflow.models.GPR(data=(Xarr, Yarr), kernel=kernel, mean_function=None)
m.trainable_variables

m.likelihood.variance.assign(0.01)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)

data = (X,Y)

# def optimize_model_with_scipy(model):
#     optimizer = gpflow.optimizers.Scipy()
#     optimizer.minimize(
#         m.training_loss,
#         variables=model.trainable_variables,
#         options={"disp": True, "maxiter": MAXITER},
#     )


# optimize_model_with_scipy(m)

# print_summary(m)


# In[ ]:





# In[473]:


import matplotlib.pyplot as plt



pred = np.linspace(X[0], X[-1], 100)[:, None][:,0,:]
pX=gx(pred)


samples = m.predict_f_samples(pred, 100)  # shape (10, 100, 1)
val,err = m.predict_y(pred)
# if val.ndim == 3:
#      val= val[:, 0, :]



Yerr = err/scales**2
Ynew2 = gy(val)
sz = [gy(i) for i in  samples]

for z in [0,1]: 
    plt.figure(figsize=(15,3))

    _ = plt.plot(gx(Xi), Yi[:,z], ".", mew=.0005)

    plt.plot(gx(X), Ynew[:,z], "o", mew=.005)
    plt.plot(pX, Ynew2[:,z], ".", mew=.005)
    
    
    top = Ynew2[:,z] +  1.96* Yerr[:,z]**.5 #1.96 *
    bot = Ynew2[:, z] - 1.96 * Yerr[:, z] ** 0.5
    
#     plt.plot(pX, top, "-", mew=.005)
    plt.fill_between(np.array(pX)[:,0], np.array(top), np.array(bot), alpha=0.3)
    
    for s in sz:#[100,10,2]
        plt.plot(pX,s[:,z] ,'coral', linewidth=0.5)
#     


    plt.xlabel('Time', horizontalalignment='right', x=1.0)
    plt.ylabel(['Latitude','Longitude'][z], horizontalalignment='right', y=1.0)
    plt.tight_layout()
    


# In[472]:


scales


# In[470]:



scales**2,361.18570063**2


# In[476]:



import tensorflow as tf
m.predict_f_compiled = tf.function(
    m.predict_f, input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
)


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
def geolocate(x):
    
    err, val = m.predict_f_compiled( (x-xx[0])/(xx[1]-xx[0]) )
    # 
    val = np.array([(i+means)/scales for i in val])
    # 
    err *= scales**2
                   
                   
    return x,val,err,xx

m.geolocate=geolocate

# m.add_to_collection('CONSTANTS', tf.constant(value=66, name=test))

print ('')
save_dir = './gpe_locations'
tf.saved_model.save(m, save_dir)
print('')
print('saved')


# In[ ]:





# In[ ]:


if pY.ndim == 3:
    pY = pY[:, 0, :]

for i in range(pY.shape[1]):
    plt.figure()
#         plt.plot(X, Y, "x")
    plt.gca().set_prop_cycle(None)
    plt.plot(pX, pY[:,i])
    
    
    top = pY[:, i] + 2.0 * pYv[:, i] ** 0.5
    bot = pY[:, i] - 2.0 * pYv[:, i] ** 0.5
    plt.fill_between(pX[:, 0], top, bot, alpha=0.3)
plt.xlabel("X")
plt.ylabel("f")
plt.title(f"")
plt.figure()
plt.plot(Z, Z * 0.0, "o")

