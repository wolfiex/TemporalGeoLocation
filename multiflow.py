#!/usr/bin/env python
# coding: utf-8

# https://app.dominodatalab.com/u/fonnesbeck/gp_showdown/view/GP+Showdown.ipynb

# In[1]:


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
# 
# plt.rcParams["figure.figsize"] = (12, 3)


X = coords[:, -2].reshape(-1, 1)
Xsec = coords[:, -1].reshape(-1, 1).astype(np.float64)

Y0 = coords[:, 0].reshape(-1, 1).astype(np.float64)
Y1 = coords[:, 1].reshape(-1, 1).astype(np.float64)

Y=np.array(list(zip(Y0,Y1)))



Yi = Y.astype(np.float64)
Xi = coords[:, -1].reshape(-1, 1).astype(np.float64)

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

xx = [Xarr.min()]
# Xsec = (Xsec - Xsec.mean()) / Xsec.std()
Xarr -= Xarr.min()
xx.append(Xarr.max())
Xarr /= Xarr.max()



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

##########
#data


###########
MAXITER = ci_niter(2000)
D = 1  # number of input dimensions
P = len(Y[0])  # number of observations = output dimensions
M = len(Y)  # number of inducing points

Zinit = np.linspace(0, 1, M)[:, None]


# In[8]:


# gpf=gpflow
# Create list of kernels for each output
kern_list = [gpflow.kernels.Matern52() for _ in range(P)]
# Create multi-output kernel from kernel list
kernel = gpflow.kernels.SeparateIndependent(kern_list)
# initialization of inducing input locations (M random points from the training inputs)
Z = Zinit.copy()
# create multi-output inducing variables from Z
iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
    gpflow.inducing_variables.InducingPoints(Z)
)

# create SVGP model as usual and optimize
m = gpflow.models.SVGP(kernel, gpflow.likelihoods.GaussianMC(), inducing_variable=iv, num_latent_gps=P)

print_summary(m)


# In[91]:


kernel =  gpflow.kernels.Matern52( variance=1, lengthscales=1.2)+ gpflow.kernels.White() #+gpflow.kernels.Exponential() 

m = gpflow.models.GPR(data=(Xarr, Yarr), kernel=kernel, mean_function=None)
m.trainable_variables

m.likelihood.variance.assign(0.01)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)




# In[93]:


data = (X,Y)

def optimize_model_with_scipy(model):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        m.training_loss,
        variables=model.trainable_variables,
        options={"disp": True, "maxiter": MAXITER},
    )


optimize_model_with_scipy(m)


print( m.predict_y(np.array([[.4]])) )


import tensorflow as tf
m.predict_f_compiled = tf.function(
    m.predict_f, input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
)


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
def geolocate(x):
    
    # err, val = m.predict_f_compiled( (x-xx[0])/xx[1] )
    # 
    # val *= scales
    # val += means
    # 
    # err *= scales 
    # err += means
    # 
    return m.predict_f_compiled( (x-xx[0])/xx[1] ), scales, x, (x-xx[0])/xx[1]

m.geolocate=geolocate

# m.add_to_collection('CONSTANTS', tf.constant(value=66, name=test))


save_dir = './gpe_locations'
tf.saved_model.save(m, save_dir)




l = tf.saved_model.load(save_dir)
l.predict_f_compiled(np.array([[0.2]]))





