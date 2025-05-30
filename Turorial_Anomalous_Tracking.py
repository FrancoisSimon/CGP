# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:57:40 2025

@author: Franc

Example of algorithm that uses the CGP framework to build a tracking model that can consider a mixture 
of recurrent processes where each process represents the motion of a track that can be either directed 
or confined similarly to (https://doi.org/10.7554/eLife.99347.1). This encodes for a model-based approach
of CGP as the constrain function fully describes the physics of the model but the constrain function can
also include more degrees of freedom (see the end of the script for an example).
"""

import numpy as np
from cgp import transpose_layer, Initial_layer_constraints, Custom_RNN_layer, Final_layer
from simulate_tracks import anomalous_diff_mixture
from matplotlib import pyplot as plt
import tensorflow as tf

# Hyperparameters that define the model:
nb_obs_vars = 1 # Number of observed variables at each time step
nb_hidden_vars = 2 # Number of hidden variables at each time step
nb_gaussians = nb_obs_vars + nb_hidden_vars # Number of Gaussians at each time step (must always equal nb_obs_vars + nb_hidden_vars)
dtype = 'float64'

@tf.function
def constraint_function(all_params, all_initial_params, dtype):
    '''
    Here, we define a constrain function that link the tensorflow model parameters to the recurrent Gaussian processes
    In this model, we assume that particles can be either directed (negative params[2]) or confined (positive params[2]).
    '''
    print(all_params)
    nb_states = all_params.shape[0]
    print('nb_states', nb_states)

    hidden_vars=[]
    obs_vars=[]
    initial_hidden_vars=[]

    for k in range(nb_states):
        params = all_params[k]
        initial_params = all_initial_params[k]
        
        if params[2] > 0:
            d = params[1]
            LocErr = params[0]
            q = params[3]
            well_distance = initial_params[1] #params[1]/(2*tf.math.abs(params[2])+1e-20)**0.5
            initial_position_spread = initial_params[0]
            
            # hidden vars:                               pos_x,           ano_pos_x,        pos_x,    ano_pos_x,
            hidden_vars = hidden_vars + [tf.stack([[[[         1/LocErr,                   0,            0,            0]]],                   # Localization error     
                                                   [[[(1-params[2])/d, params[2]/d, -1/d,            0]]],                # Diffusion + anomalous drift
                                                   [[[                   0,         1/q,            0, -1/q]]]])]                    # Diffusion of the anomalous position
                                         
            
            obs_vars = obs_vars + [tf.stack([[[[-1/LocErr]]],
                                             [[[0]]],
                                             [[[0]]]])]
            
            # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
            # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
            
            well_distance = d/(2*tf.math.abs(params[2])+1e-10)**0.5
            
            initial_hidden_vars = initial_hidden_vars + [tf.stack([[[[1/initial_position_spread,                   0]]],
                                                                   [[[1/well_distance,     -1/well_distance]]]])]
            
        else:
            d = params[1]
            LocErr = params[0]
            q = params[3]
            initial_position_spread = initial_params[0]
            
            # hidden vars:                   pos_x,   ano_pos_x,        pos_x,    ano_pos_x,
            hidden_vars = hidden_vars + [tf.stack([[[[1/LocErr,           0,            0,            0]]],               # Localization error    
                                         [[[1/d, 1/d, -1/d,            0]]],               # Diffusion + anomalous drift
                                         [[[          0, 1/q,            0, -1/q]]],                    # Diffusion of the anomalous position
                                         ])]
            
            obs_vars = obs_vars + [tf.stack([[[[-1/LocErr]]],
                                             [[[0]]],
                                             [[[0]]]])]
            

            # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
            # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
        
            initial_hidden_vars = initial_hidden_vars + [tf.stack([[[[    1/initial_position_spread,                   0]]],
                                                                   [[[        0.0000000000001,     1/-params[2]]]]])]
            
    Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states), dtype = dtype)
    initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
    initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    initial_biases = tf.zeros((nb_hidden_vars, 1, nb_states), dtype = dtype)
        
    hidden_vars = tf.concat(hidden_vars, 2)
    obs_vars = tf.concat(obs_vars, 2)
    initial_hidden_vars = tf.concat(initial_hidden_vars, 2)
    
    return hidden_vars, obs_vars, Gaussian_stds, biases, initial_hidden_vars, initial_obs_vars, initial_Gaussian_stds, initial_biases

track_len = 50
nb_tracks = 500
batch_size = nb_tracks

#simulation of tracks that are either in confined motion or in directed motion
all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                           nb_tracks = nb_tracks,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([1.]),
                           Ds = np.array([0.25]),
                           nb_dims = 2,
                           velocities = np.array([0.]),
                           angular_Ds = np.array([0.]),
                           conf_forces = np.array([0.1]),
                           conf_Ds = np.array([0.0]),
                           conf_dists = np.array([0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10]))

all_tracks = np.array(all_tracks)

lim = 1
nb_rows = 5

plt.figure(figsize = (10, 10))
for i in range(nb_rows):
    for j in range(nb_rows):
        track = all_tracks[i*nb_rows+j]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], alpha = 1)
plt.gca().set_aspect('equal', adjustable='box')

tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)

loc_err = 0.02 # localization error
d = 0.1 # diffusion length
l = 0.05 # confinement factor
q = 0.01 # change of the anomalous factor

nb_states = 1 # Number of mutually exclusive states assumed by the model (mixture of Gaussian processes). This number can be increased to consider 2 or more states.
'''
Then `all_params` and `all_initial_params` inform the initial parameters for each state (columns)
params: (localization error, diffusion length, anomalous factor, evolution rate of the anomalous variable)
Add rows so the number of states equal the number of rows.
'''
all_params = tf.constant([[loc_err, d, l, q]], dtype = dtype) 
all_initial_params = tf.constant([[10, 0.01]], dtype = dtype)


'''
Then we build the keras (tensorflow) model
'''
inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])

Init_layer = Initial_layer_constraints(nb_states,
                                       nb_gaussians,
                                       nb_obs_vars,
                                       nb_hidden_vars,
                                       all_params,
                                       all_initial_params,
                                       constraint_function,
                                       dtype = dtype)

tensor1, initial_states = Init_layer(transposed_inputs)

Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases = initial_states

sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)

layer = Custom_RNN_layer(Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, dtype = dtype)
states = layer(sliced_inputs, Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases)

F_layer = Final_layer(Init_layer.final_sequence_phase_1, dtype = dtype)
outputs = F_layer(states)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Diffusion_model")

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP# + 0.1*tf.reduce_sum(y_pred, 1, keepdims = True)
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

'''
We can then start the fitting. Modifying the length of the tracks or important changes of parameter scales might require to adapt the learning rate.
'''
adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/GPU:0'):
    history0 = model.fit(tracks, tracks, epochs = 700, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

'''
Once the model is fitted, we can collect the learned parameters
'''

Init_layer.param_vars.numpy()
Init_layer.initial_param_vars.numpy()

if Init_layer.param_vars.numpy()[0,2] <0:
    print('Type of motion detected: Directed')
    print('Localization error:', np.abs(Init_layer.param_vars.numpy()[0,0]))
    print('Diffusion length per step:', np.abs(Init_layer.param_vars.numpy()[0,1]))
    print('Velocity:', -Init_layer.param_vars.numpy()[0,2]*2**0.5)
    print('Orientation changes:', np.abs(Init_layer.param_vars.numpy()[0,0]))

if Init_layer.param_vars.numpy()[0,2] >0:
    print('Type of motion detected: Confined')
    print('Localization error:', np.abs(Init_layer.param_vars.numpy()[0,0]))
    print('Diffusion length per step:', np.abs(Init_layer.param_vars.numpy()[0,1]))
    print('Confinement factor per step:', Init_layer.param_vars.numpy()[0,2]*2**0.5)
    print('Diffusion length of the confinement area:', np.abs(Init_layer.param_vars.numpy()[0,0]))

track_len = 30
nb_tracks = 500
batch_size = nb_tracks

'''
Here we describe a function to generate directed tracks if the user aims to fit such tracks instead of confined tracks.

all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                           nb_tracks = nb_tracks,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([1]),
                           Ds = np.array([0.25]),
                           nb_dims = 2,
                           velocities = np.array([0.1]),
                           angular_Ds = np.array([0]),
                           conf_forces = np.array([0.]),
                           conf_Ds = np.array([0.0]),
                           conf_dists = np.array([0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10]))
'''

'''
We can use CGP in a data driven fashion by allowing more degrees of freedom than the physics-based approach
all_params = tf.constant([[a, b, c, d, e, f, g]], dtype = dtype) 
all_initial_params = tf.constant([[h, g, k]], dtype = dtype)
The following constrain function for instance can model both directed and confined models with one set of parameters

@tf.function
def constraint_function(all_params, all_initial_params, dtype):
    '''
    Here, we define a constrain function that link the tensorflow model parameters to the recurrent Gaussian processes
    In this model, we assume that particles can be either directed (negative params[2]) or confined (positive params[2]).
    '''
    print(all_params)
    nb_states = all_params.shape[0]
    print('nb_states', nb_states)

    hidden_vars=[]
    obs_vars=[]
    initial_hidden_vars=[]

    for k in range(nb_states):
        params = all_params[k]
        initial_params = all_initial_params[k]
    
        a, b, c, d, e, f, g = params
        h, j, k = initial_params
        
        # hidden vars:                               pos_x,   ano_pos_x,     pos_x,  ano_pos_x,
        hidden_vars = hidden_vars + [tf.stack([[[[         a,         0,         0,          0]]],                   # Localization error     
                                               [[[         b,         c,         d,          0]]],                # Diffusion + anomalous drift
                                               [[[         0,         e,         0,          f]]]])]                    # Diffusion of the anomalous position
        
        obs_vars = obs_vars + [tf.stack([[[[g]]],
                                         [[[0]]],
                                         [[[0]]]])]
            
        # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
        # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
                    
        initial_hidden_vars = initial_hidden_vars + [tf.stack([[[[      h,      0]]],
                                                               [[[      j,      k]]]])]
            
    Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states), dtype = dtype)
    initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
    initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    initial_biases = tf.zeros((nb_hidden_vars, 1, nb_states), dtype = dtype)
        
    hidden_vars = tf.concat(hidden_vars, 2)
    obs_vars = tf.concat(obs_vars, 2)
    initial_hidden_vars = tf.concat(initial_hidden_vars, 2)
    
    return hidden_vars, obs_vars, Gaussian_stds, biases, initial_hidden_vars, initial_obs_vars, initial_Gaussian_stds, initial_biases
'''


