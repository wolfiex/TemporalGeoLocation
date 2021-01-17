#!/usr/bin/env python
# coding: utf-8

# https://app.dominodatalab.com/u/fonnesbeck/gp_showdown/view/GP+Showdown.ipynb

# In[1]:



r
###########
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


kernel =  gpflow.kernels.Matern52( variance=1, lengthscales=1.2)+ gpflow.kernels.White() #+gpflow.kernels.Exponential() 

m = gpflow.models.GPR(data=(Xarr, Yarr), kernel=kernel, mean_function=None)
m.trainable_variables

m.likelihood.variance.assign(0.01)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)


# In[92]:


Y.shape,X.shape


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



