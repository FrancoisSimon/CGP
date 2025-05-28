# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:53:25 2025

@author: Franc
"""

import numpy as np
import tensorflow as tf
from CGP import Custom_RNN_layer, Final_layer, transpose_layer, Initial_layer_constraints
from matplotlib import pyplot as plt
from numba import njit, typed

@njit
def anomalous_diff_mixture(track_len=640,
                           nb_tracks = 100,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([0.5, 0.5]),
                           Ds = np.array([0.0, 0.05]),
                           nb_dims = 2,
                           velocities = np.array([0.1, 0.0]),
                           angular_Ds = np.array([0.0, 0.0]),
                           conf_forces = np.array([0.0, 0.2]),
                           conf_Ds = np.array([0.0, 0.0]),
                           conf_dists = np.array([0.0, 0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10])):
    '''
    Code to generate tracks in directed or confined motion
    '''
            
    nb_states = len(velocities)
    if not np.all(np.array([len(Fs), len(Ds), len(velocities), len(angular_Ds), len(conf_forces), len(conf_Ds), len(conf_dists)]) == nb_states):
        raise ValueError('Fs, Ds, velocities, angular_Ds, conf_forces, conf_Ds and conf_dists must all be 1D arrays of the same number of element (one element per state)')
    # diff + persistent motion + elastic confinement
    conf_sub_forces = conf_forces / nb_sub_steps
    sub_dt = dt / nb_sub_steps
   
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
    
    all_states = np.zeros(nb_tracks)
    
    for kkk in range(nb_tracks):
        state = np.argmin(np.random.rand()>cum_Fs)
        all_states[kkk] = state
        D, velocity, angular_D, conf_sub_force, conf_D, conf_dist = (Ds[state], velocities[state], angular_Ds[state], conf_sub_forces[state], conf_Ds[state], conf_dists[state])
       
        positions = np.zeros((track_len * nb_sub_steps, nb_dims))
        
        positions[0] = np.random.rand(nb_dims)*field_of_view
        disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
       
        anchor_positions = np.random.normal(0, np.sqrt(2*conf_D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
        anchor_positions[0] = positions[0] + np.random.normal(0,conf_dist, nb_dims)
       
        for i in range(1, len(anchor_positions)):
            anchor_positions[i] += anchor_positions[i-1]
       
        d_angles = np.random.normal(0, 1, ((track_len) * nb_sub_steps)-1) * (2*angular_D*sub_dt)**0.5
        angles = np.zeros((track_len * nb_sub_steps-1))
        angles[0] = np.random.rand()*2*np.pi
        for i in range(1, len(d_angles)):
            angles[i] = angles[i-1] + d_angles[i]
       
        for i in range(len(positions)-1):
            angle = angles[i-1]
            pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
            positions[i+1] = positions[i] + pesistent_disp + disps[i]
            positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
       
        final_track = np.zeros((track_len, nb_dims))
        for i in range(track_len):
            final_track[i] = positions[i*nb_sub_steps]
       
        final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
       
        if kkk ==0:
            final_tracks = typed.List([final_track])
        else:
            final_tracks.append(final_track)
    return final_tracks, all_states


'''
Tracking with constraints to extract the model parameters 
'''

# Hyperparameters
dtype = 'float64'
nb_obs_vars = 2
nb_hidden_vars = 4
nb_gaussians = nb_obs_vars + nb_hidden_vars
nb_states = 1 # number of states of the model

def constraint_function(params, initial_params, dtype):
    
    print('test1')
    if params[2] > 0:
        # hidden vars:                              pos_x,           ano_pos_x,        pos_x,    ano_pos_x,
        hidden_vars = tf.stack([[[[(1-params[2])/params[1], params[2]/params[1], -1/params[1],            0]]],                    # Diffusion + anomalous drift
                                   [[[                   0,         1/params[3],            0, -1/params[3]]]],                    # Diffusion of the anomalous position
                                   [[[         1/params[0],                   0,            0,            0]]]])    # Localization error     
        print('test2')
    
        obs_vars = tf.stack([[[[0]]],
                             [[[0]]],
                             [[[-1/params[0]]]]])
        
        Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states), dtype = dtype)
        
        # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
        # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
    
        initial_hidden_vars = tf.stack([[[[    1/initial_params[0],                   0]]],
                                        [[[-1/initial_params[1],     1/initial_params[2]]]]])
        
        initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
        initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        initial_biases =  tf.zeros((nb_hidden_vars, 1, nb_states), dtype = dtype)
        
    else:
        # hidden vars:                   pos_x,   ano_pos_x,        pos_x,    ano_pos_x,
        hidden_vars = tf.stack([[[[1/params[1], 1/params[1], -1/params[1],            0]]],                    # Diffusion + anomalous drift
                                [[[          0, 1/params[3],            0, -1/params[3]]]],                    # Diffusion of the anomalous position
                                [[[1/params[0],           0,            0,            0]]]])    # Localization error     
        print('test2')
    
        obs_vars = tf.stack([[[[0]]],
                                [[[0]]],
                                [[[-1/params[0]]]]])
        
        Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states), dtype = dtype)
        
        # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
        # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
    
        initial_hidden_vars = tf.stack([[[[    1/initial_params[0],                   0]]],
                                        [[[ 0.0000000000001,     1/-params[2]]]]])
        
        initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
        initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        initial_biases =  tf.zeros((nb_hidden_vars, 1, nb_states), dtype = dtype)

    return hidden_vars, obs_vars, Gaussian_stds, biases, initial_hidden_vars, initial_obs_vars, initial_Gaussian_stds, initial_biases

'''
Estimation of the diffusion length of Brownian tracks
'''

track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

loc_err = 0.025 # localization error
d = 0.1 # diffusion length
l = 0.5 # confinement factor
q = 0.01 # change of the anomalous factor

# initial parameters of the model.
params = tf.constant([loc_err, d, l, q], dtype = dtype) 
initial_params = tf.constant([10, 0.01, 0.1], dtype = dtype)

ds = np.arange(0, 0.101,0.01)

est_ds = []

for d in ds:
    
    D = d**2/(2*0.02)
    #Brownian
    all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                               nb_tracks = nb_tracks,
                               LocErr=0.02, # localization error in x, y and z (even if not used)
                               Fs = np.array([1.]),
                               Ds = np.array([D]),
                               nb_dims = 2,
                               velocities = np.array([0.0]),
                               angular_Ds = np.array([0.]),
                               conf_forces = np.array([0.0]),
                               conf_Ds = np.array([0.0]),
                               conf_dists = np.array([0.0]),
                               LocErr_std = 0,
                               dt = 0.02,
                               nb_sub_steps = 10,
                               field_of_view = np.array([10,10]))
        
    all_tracks = np.array(all_tracks)
    tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)

        
    #inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])
    
    Init_layer = Initial_layer_constraints(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
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
    
    preds = model.predict(tracks, batch_size = batch_size)
    
    np.mean(preds)
    model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        
        max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
        reduced_LP = y_pred - max_LP
        pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
        
        return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis
    
    nb_epochs = 2000
    nb_data_points = batch_size*(track_len-1)
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history0 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history1 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 1000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    est_ds.append(Init_layer.param_vars[1])
    
plt.figure(figsize = (3,3))
plt.plot(ds, np.abs(est_ds), marker = '+', markersize = 6)
plt.grid()
plt.xlabel('True diffusion length [$\mathrm{\mu m}$]')
plt.xticks([0,0.02,0.04,0.06,0.08,0.1])
plt.ylabel('Estimated diffusion length [$\mathrm{\mu m}$]')
plt.tight_layout()




'''
Estimation of the velocity of the directed motion
'''
track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

vs = np.arange(0, 0.051,0.005)
est_vs = []

for v in vs:
    
    #Brownian
    # Directed
    all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                               nb_tracks = nb_tracks,
                               LocErr=0.02, # localization error in x, y and z (even if not used)
                               Fs = np.array([1.]),
                               Ds = np.array([0.0]),
                               nb_dims = 2,
                               velocities = np.array([v]),
                               angular_Ds = np.array([0.5]),
                               conf_forces = np.array([0.]),
                               conf_Ds = np.array([0.0]),
                               conf_dists = np.array([0.0]),
                               LocErr_std = 0,
                               dt = 0.02,
                               nb_sub_steps = 10,
                               field_of_view = np.array([10,10]))
    
    all_tracks = np.array(all_tracks)
    tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)

        
    #inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])
    
    Init_layer = Initial_layer_constraints(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
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
    
    preds = model.predict(tracks, batch_size = batch_size)
    
    np.mean(preds)
    model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        
        max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
        reduced_LP = y_pred - max_LP
        pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
        
        return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis
    
    nb_epochs = 2000
    nb_data_points = batch_size*(track_len-1)
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history0 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history1 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 1000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    est_vs.append(-Init_layer.param_vars[2]*2**0.5)
    
plt.figure(figsize = (3,3))
plt.plot(vs, np.abs(est_vs), marker = '+', markersize = 6)
plt.grid()
plt.xlabel('True velocity [$\mathrm{\mu m / step}$]')
plt.ylabel('Estimated velocity [$\mathrm{\mu m / step}$]')
plt.xticks([0,0.01,0.02,0.03,0.04,0.05])
plt.tight_layout()




'''
Estimation of the confinement factor
'''
track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

ls = np.arange(0, 0.71, 0.05)
est_ls = []

for l in ls:
    
    cont_l = - np.log(1-l)

    # Confined
    all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                                                   nb_tracks = nb_tracks,
                                                   LocErr=0.02, # localization error in x, y and z (even if not used)
                                                   Fs = np.array([1.]),
                                                   Ds = np.array([0.25]),
                                                   nb_dims = 2,
                                                   velocities = np.array([0.0]),
                                                   angular_Ds = np.array([0.0]),
                                                   conf_forces = np.array([cont_l]),
                                                   conf_Ds = np.array([0.0]),
                                                   conf_dists = np.array([0.0]),
                                                   LocErr_std = 0,
                                                   dt = 0.02,
                                                   nb_sub_steps = 10,
                                                   field_of_view = np.array([10,10]))
    
    all_tracks = np.array(all_tracks)
    tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)
    
    #inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])
    
    Init_layer = Initial_layer_constraints(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
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
    
    preds = model.predict(tracks, batch_size = batch_size)
    
    np.mean(preds)
    model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        
        max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
        reduced_LP = y_pred - max_LP
        pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
        
        return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis
    
    nb_epochs = 2000
    nb_data_points = batch_size*(track_len-1)
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history0 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history1 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/8000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    est_ls.append(Init_layer.param_vars[2])


plt.figure(figsize = (3,3))
plt.plot(ls[1:], np.abs(est_ls[1:]), marker = '+', markersize = 6)
plt.grid()
plt.xlabel('True confinement factor')
plt.ylabel('Estimated confinement factor')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
plt.tight_layout()



















































































# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:20:48 2024

@author: franc
"""


import numpy as np
import tensorflow as tf
from irgp import Initial_layer, Custom_RNN_layer, Final_layer, anomalous_diff_mixture, transpose_layer
from matplotlib import pyplot as plt
import pandas as pd
from numba import njit, typed, prange, jit

@njit
def anomalous_diff_mixture(track_len=640,
                           nb_tracks = 100,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([0.5, 0.5]),
                           Ds = np.array([0.0, 0.05]),
                           nb_dims = 2,
                           velocities = np.array([0.1, 0.0]),
                           angular_Ds = np.array([0.0, 0.0]),
                           conf_forces = np.array([0.0, 0.2]),
                           conf_Ds = np.array([0.0, 0.0]),
                           conf_dists = np.array([0.0, 0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10])):
            
    nb_states = len(velocities)
    if not np.all(np.array([len(Fs), len(Ds), len(velocities), len(angular_Ds), len(conf_forces), len(conf_Ds), len(conf_dists)]) == nb_states):
        raise ValueError('Fs, Ds, velocities, angular_Ds, conf_forces, conf_Ds and conf_dists must all be 1D arrays of the same number of element (one element per state)')
    # diff + persistent motion + elastic confinement
    conf_sub_forces = conf_forces / nb_sub_steps
    sub_dt = dt / nb_sub_steps
   
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
    
    all_states = np.zeros(nb_tracks)
    
    for kkk in range(nb_tracks):
        state = np.argmin(np.random.rand()>cum_Fs)
        all_states[kkk] = state
        D, velocity, angular_D, conf_sub_force, conf_D, conf_dist = (Ds[state], velocities[state], angular_Ds[state], conf_sub_forces[state], conf_Ds[state], conf_dists[state])
       
        positions = np.zeros((track_len * nb_sub_steps, nb_dims))
        
        positions[0] = np.random.rand(nb_dims)*field_of_view
        disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
       
        anchor_positions = np.random.normal(0, np.sqrt(2*conf_D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
        anchor_positions[0] = positions[0] + np.random.normal(0,conf_dist, nb_dims)
       
        for i in range(1, len(anchor_positions)):
            anchor_positions[i] += anchor_positions[i-1]
       
        d_angles = np.random.normal(0, 1, ((track_len) * nb_sub_steps)-1) * (2*angular_D*sub_dt)**0.5
        angles = np.zeros((track_len * nb_sub_steps-1))
        angles[0] = np.random.rand()*2*np.pi
        for i in range(1, len(d_angles)):
            angles[i] = angles[i-1] + d_angles[i]
       
        for i in range(len(positions)-1):
            angle = angles[i-1]
            pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
            positions[i+1] = positions[i] + pesistent_disp + disps[i]
            positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
       
        final_track = np.zeros((track_len, nb_dims))
        for i in range(track_len):
            final_track[i] = positions[i*nb_sub_steps]
       
        final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
       
        if kkk ==0:
            final_tracks = typed.List([final_track])
        else:
            final_tracks.append(final_track)
    return final_tracks, all_states


'''
Generate xy correlated anomalous tracks
'''

dtype = 'float64'
nb_states = 1
nb_obs_vars = 2
nb_hidden_vars = 4
nb_gaussians = nb_obs_vars + nb_hidden_vars

# hidden vars:            pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
hidden_vars = np.array([[[[   1,     0,       0.2,      0.01,     -1,     0,         0,       0]]],
                        [[[   0,     1,      0.01,       0.2,      0,    -1,         0,       0]]],
                        [[[   0,     0,         1,         0,      0,     0,        -1,     0.01]]],
                        [[[   0,     0,         0,         1,      0,     0,      0.01,      -1]]],
                        [[[   1,     0,         0,         0,      0,     0,         0,       0]]],
                        [[[   0,     1,         0,         0,      0,     0,         0,       0]]]], dtype = dtype)*(0.99+0.02*np.random.rand(nb_gaussians, 1, 1, nb_hidden_vars*2))

obs_vars = np.array([[[[0, 0]]],
                     [[[0, 0]]],
                     [[[0, 0]]],
                     [[[0, 0]]],
                     [[[-1, 0]]],
                     [[[0, -1]]]], dtype = dtype)

indices_hidden_vars = np.array(np.where(hidden_vars != 0)).T
indices_obs_vars = np.array(np.where(obs_vars != 0)).T

hidden_var_coef_values = hidden_vars[np.where(hidden_vars != 0)]
obs_var_coef_values = obs_vars[np.where(obs_vars != 0)]

Gaussian_stds = np.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1))

biases = np.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states))

# It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
# hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
initial_hidden_vars = np.array([[[[    1,     0,         0,         0,]]],
                                [[[    0,     1,         0,         0,]]],
                                [[[   -1,     0,         1,      0.01,]]],
                                [[[    0,    -1,      0.01,         1,]]]], dtype = dtype)

initial_obs_vars = np.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars))

indices_initial_hidden_vars = np.array(np.where(initial_hidden_vars != 0)).T
indices_initial_obs_vars = np.array(np.where(initial_obs_vars != 0)).T

initial_obs_var_coef_values = indices_initial_obs_vars[np.where(indices_initial_obs_vars != 0)].astype(dtype)
initial_hidden_var_coef_values = initial_hidden_vars[np.where(initial_hidden_vars != 0)]

initial_Gaussian_stds = np.ones((nb_hidden_vars, 1, nb_states, 1))
initial_biases =  np.zeros((nb_hidden_vars, 1, nb_states))

track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                           nb_tracks = nb_tracks,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([1.]),
                           Ds = np.array([0.0]),
                           nb_dims = 2,
                           velocities = np.array([0.02]),
                           angular_Ds = np.array([0.5]),
                           conf_forces = np.array([0.0]),
                           conf_Ds = np.array([0.0]),
                           conf_dists = np.array([0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10]))
all_tracks = np.array(all_tracks)

tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)

lim = 1
nb_rows = 5
plt.figure(figsize = (10, 10))
for i in range(nb_rows):
    for j in range(nb_rows):
        plt.subplot(nb_rows, nb_rows, i*nb_rows+j+1)
        track = all_tracks[i*nb_rows+j]
        plt.plot(track[:,0], track[:,1], alpha = 1)
        plt.xlim([np.mean(track[:, 0])-lim, np.mean(track[:, 0])+lim])
        plt.ylim([np.mean(track[:, 1])-lim, np.mean(track[:, 1])+lim])
        plt.gca().set_aspect('equal', adjustable='box')

#inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])

Init_layer = Initial_layer(obs_var_coef_values,
                           indices_obs_vars,
                           hidden_var_coef_values,
                           indices_hidden_vars,
                           Gaussian_stds,
                           biases,
                           nb_states,
                           nb_gaussians,
                           nb_obs_vars,
                           nb_hidden_vars,
                           initial_obs_var_coef_values,
                           indices_initial_obs_vars,
                           initial_hidden_var_coef_values,
                           indices_initial_hidden_vars,
                           initial_Gaussian_stds,
                           initial_biases,
                           trainable_params = {'obs': True, 'hidden': True, 'stds': False, 'biases': True},
                           trainable_initial_params = {'obs': False, 'hidden': True, 'stds': False, 'biases': True},
                           dtype = dtype)

tensor1, initial_states = Init_layer(transposed_inputs)

Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases = initial_states

sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)

layer = Custom_RNN_layer(Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, dtype = dtype)
states = layer(sliced_inputs, Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases)

F_layer = Final_layer(Init_layer.final_sequence_phase_1, dtype = dtype)
outputs = F_layer(states)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Diffusion_model")

preds = model.predict(tracks, batch_size = batch_size)

np.mean(preds)
model.summary()

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

nb_epochs = 2000
nb_data_points = batch_size*(track_len-1)

adam = tf.keras.optimizers.Adam(learning_rate=1/10, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history = model.fit(tracks, tracks, epochs = 10000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history = model.fit(tracks, tracks, epochs = 10000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history = model.fit(tracks, tracks, epochs = 10000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])



'''
Generate xy independent anomalous tracks
'''

dtype = 'float64'
nb_states = 1
nb_obs_vars = 1
nb_hidden_vars = 2
nb_gaussians = nb_obs_vars + nb_hidden_vars

# hidden vars:            pos_x, ano_pos_x,    pos_x, ano_pos_x,
hidden_vars = np.array([[[[   0.5,       0.2,       -1,        0]]],
                        [[[   0,         1,        0,       -1]]],
                        [[[   1,         0,        0,        0]]]], dtype = dtype)*(0.99+0.02*np.random.rand(nb_gaussians, 1, 1, nb_hidden_vars*2))

obs_vars = np.array([[[[0]]],
                     [[[0]]],
                     [[[-1]]]], dtype = dtype)

# hidden vars:            pos_x, ano_pos_x,    pos_x, ano_pos_x,
hidden_vars = np.array([[[[   0.8/0.15, 0.2/0.15,       -1/0.15,        0]]],
                        [[[   0,         1/0.01,        0,       -1/0.01]]],
                        [[[   1/0.025,         0,        0,        0]]]], dtype = dtype)*(0.99+0.02*np.random.rand(nb_gaussians, 1, 1, nb_hidden_vars*2))

obs_vars = np.array([[[[0]]],
                     [[[0]]],
                     [[[-1/0.025]]]], dtype = dtype)

indices_hidden_vars = np.array(np.where(hidden_vars != 0)).T
indices_obs_vars = np.array(np.where(obs_vars != 0)).T

hidden_var_coef_values = hidden_vars[np.where(hidden_vars != 0)]
obs_var_coef_values = obs_vars[np.where(obs_vars != 0)]

Gaussian_stds = np.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1))

biases = np.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states))

# It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
# hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
initial_hidden_vars = np.array([[[[    1,     0]]],
                                [[[   -1,     1]]]], dtype = dtype)

initial_hidden_vars = np.array([[[[    1/10,     0]]],
                                [[[   -1/0.1,     1/0.1]]]], dtype = dtype)

initial_obs_vars = np.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars))

indices_initial_hidden_vars = np.array(np.where(initial_hidden_vars != 0)).T
indices_initial_obs_vars = np.array(np.where(initial_obs_vars != 0)).T

initial_obs_var_coef_values = indices_initial_obs_vars[np.where(indices_initial_obs_vars != 0)].astype(dtype)
initial_hidden_var_coef_values = initial_hidden_vars[np.where(initial_hidden_vars != 0)]

initial_Gaussian_stds = np.ones((nb_hidden_vars, 1, nb_states, 1))
initial_biases =  np.zeros((nb_hidden_vars, 1, nb_states))

track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

#Brownian
all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                           nb_tracks = nb_tracks,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([1.]),
                           Ds = np.array([0.25]),
                           nb_dims = 2,
                           velocities = np.array([0.0]),
                           angular_Ds = np.array([0.]),
                           conf_forces = np.array([0.0]),
                           conf_Ds = np.array([0.0]),
                           conf_dists = np.array([0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10]))

# Directed
all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                           nb_tracks = nb_tracks,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([1.]),
                           Ds = np.array([0.0]),
                           nb_dims = 2,
                           velocities = np.array([0.02]),
                           angular_Ds = np.array([0.5]),
                           conf_forces = np.array([0.0]),
                           conf_Ds = np.array([0.0]),
                           conf_dists = np.array([0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10]))

(2*0.5*0.02)**0.5
2**0.5/10
# Confined
all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                           nb_tracks = nb_tracks,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           Fs = np.array([1.]),
                           Ds = np.array([0.25]),
                           nb_dims = 2,
                           velocities = np.array([0.0]),
                           angular_Ds = np.array([0.0]),
                           conf_forces = np.array([0.3]),
                           conf_Ds = np.array([0.0]),
                           conf_dists = np.array([0.0]),
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           field_of_view = np.array([10,10]))

all_tracks = np.array(all_tracks)

tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)

lim = 1
nb_rows = 5
plt.figure(figsize = (10, 10))
for i in range(nb_rows):
    for j in range(nb_rows):
        plt.subplot(nb_rows, nb_rows, i*nb_rows+j+1)
        track = all_tracks[i*nb_rows+j]
        plt.plot(track[:,0], track[:,1], alpha = 1)
        plt.xlim([np.mean(track[:, 0])-lim, np.mean(track[:, 0])+lim])
        plt.ylim([np.mean(track[:, 1])-lim, np.mean(track[:, 1])+lim])
        plt.gca().set_aspect('equal', adjustable='box')

#inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])

Init_layer = Initial_layer(obs_var_coef_values,
                           indices_obs_vars,
                           hidden_var_coef_values,
                           indices_hidden_vars,
                           Gaussian_stds,
                           biases,
                           nb_states,
                           nb_gaussians,
                           nb_obs_vars,
                           nb_hidden_vars,
                           initial_obs_var_coef_values,
                           indices_initial_obs_vars,
                           initial_hidden_var_coef_values,
                           indices_initial_hidden_vars,
                           initial_Gaussian_stds,
                           initial_biases,
                           trainable_params = {'obs': True, 'hidden': True, 'stds': False, 'biases': True},
                           trainable_initial_params = {'obs': False, 'hidden': True, 'stds': False, 'biases': True},
                           dtype = dtype)

tensor1, initial_states = Init_layer(transposed_inputs)

Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases = initial_states

sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)

layer = Custom_RNN_layer(Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, dtype = dtype)
states = layer(sliced_inputs, Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases)

F_layer = Final_layer(Init_layer.final_sequence_phase_1, dtype = dtype)
outputs = F_layer(states)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Diffusion_model")

preds = model.predict(tracks, batch_size = batch_size)

np.mean(preds)
model.summary()

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

nb_epochs = 2000
nb_data_points = batch_size*(track_len-1)

adam = tf.keras.optimizers.Adam(learning_rate=1/10, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history0 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history1 = model.fit(tracks, tracks, epochs = 10000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history2 = model.fit(tracks, tracks, epochs = 10000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history3 = model.fit(tracks, tracks, epochs = 10000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])


est_obs_var_coef_values, est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_stds, est_initial_obs_var_coefs_values, est_initial_stds = model.layers[3].weights

sparse_obs_var_coefs = tf.SparseTensor(indices = indices_obs_vars,
                                       values = est_obs_var_coef_values,
                                       dense_shape = (nb_gaussians, 1, nb_states, nb_obs_vars))

est_obs_var_coefs = tf.sparse.to_dense(sparse_obs_var_coefs)

sparse_hidden_var_coefs = tf.SparseTensor(indices = indices_hidden_vars,
                                       values = est_hidden_var_coef_values,
                                       dense_shape = (nb_gaussians, 1,nb_states, 2*nb_hidden_vars))

est_hidden_var_coefs = tf.sparse.to_dense(sparse_hidden_var_coefs)

sparse_initial_obs_var_coefs = tf.SparseTensor(indices = indices_initial_obs_vars,
                                       values = est_initial_obs_var_coefs_values,
                                       dense_shape = (nb_hidden_vars, 1, nb_states, nb_obs_vars))

est_initial_obs_var_coefs = tf.sparse.to_dense(sparse_initial_obs_var_coefs)

sparse_initial_hidden_var_coefs = tf.SparseTensor(indices = indices_initial_hidden_vars,
                                       values = est_initial_hidden_var_coefs_values,
                                       dense_shape = (nb_hidden_vars, 1, nb_states, 2*nb_hidden_vars))

est_initial_hidden_var_coefs = tf.sparse.to_dense(sparse_initial_hidden_var_coefs)

def sample_multivariate_Gaussian(est_initial_obs_var_coefs, est_initial_hidden_var_coefs, est_initial_biases, est_obs_var_coefs, est_hidden_var_coefs, est_biases, nb_hidden_vars, nb_obs_vars, nb_steps, nb_samples, state = 0):
    
    init_obs_vars = est_initial_obs_var_coefs[:,0,state]
    init_hidden_vars = est_initial_hidden_var_coefs[:,0,state]
    init_biases = est_initial_biases[:,0,state]
    
    obs_vars = est_obs_var_coefs[:,0,state]
    hidden_vars = est_hidden_var_coefs[:,0,state]
    biases = est_biases[:,0,state]
    
    init_obs_zeros = tf.zeros(init_obs_vars.shape, dtype = dtype)
    init_hidden_zeros = tf.zeros((init_hidden_vars.shape[0], nb_hidden_vars), dtype = dtype)

    N = nb_steps
    A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*(N-1)
    A0 = tf.concat(tuple(A0), axis = 1)
    
    biases_univariate = [init_biases]
    A = [A0]
    for i in range(0, N):
        obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
        hidden_zeros = tf.zeros((hidden_vars.shape[0], nb_hidden_vars), dtype = dtype)
        
        var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1)
        Ai = tf.concat(tuple(var_list), axis = 1)
        
        A.append(Ai)
        biases_univariate.append(biases)
    
    A = tf.concat(A, axis = 0)
    biases_univariate = tf.concat(biases_univariate, axis = 0)[:,None]

    inv_A = tf.linalg.inv(A)
    #print('A', A)
    #A[0,2] =0
    
    Sigma_XY = tf.matmul(inv_A, tf.transpose(inv_A))
    
    Sigma_X = Sigma_XY[:nb_steps*nb_obs_vars, :nb_steps*nb_obs_vars]
    
    est_bias_XY = tf.matmul(inv_A, biases_univariate)
    est_bias_X = est_bias_XY[:nb_steps*nb_obs_vars, 0]
    
    est_samples = np.random.multivariate_normal(est_bias_X, Sigma_X, nb_samples)
    est_samples = est_samples.reshape((nb_samples, nb_steps, nb_obs_vars))
    
    '''
    est_vars = np.random.multivariate_normal(est_bias_XY[:,0], Sigma_XY, nb_samples)
    est_samples = est_vars[:,:nb_steps*nb_obs_vars].reshape((nb_samples, nb_steps, nb_obs_vars))
    est_hidden_vars = est_vars[:,(nb_steps)*nb_obs_vars+nb_hidden_vars:].reshape((nb_samples, nb_steps, nb_hidden_vars))
    est_vars = np.concatenate((est_samples, est_hidden_vars), axis = 2)
    '''
    return est_samples

nb_steps = track_len
nb_samples = 10000
state = 0

est_samples_x = sample_multivariate_Gaussian(est_initial_obs_var_coefs, est_initial_hidden_var_coefs, est_initial_biases, est_obs_var_coefs, est_hidden_var_coefs, est_biases, nb_hidden_vars, nb_obs_vars, nb_steps, nb_samples, state = state)
est_samples_y = sample_multivariate_Gaussian(est_initial_obs_var_coefs, est_initial_hidden_var_coefs, est_initial_biases, est_obs_var_coefs, est_hidden_var_coefs, est_biases, nb_hidden_vars, nb_obs_vars, nb_steps, nb_samples, state = state)
est_samples = np.concatenate((est_samples_x, est_samples_y), axis = -1)

est_samples.shape

nb_rows = 6

est_samples_y.shape

lim = 2

plt.figure(figsize = (10, 10))

for i in range(nb_rows):
    for j in range(nb_rows):
        track = est_samples[i*nb_rows+j]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], alpha = 1)
plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize = (10, 10))
for i in range(nb_rows):
    for j in range(nb_rows):
        track = all_tracks[i*nb_rows+j]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], alpha = 1)
plt.gca().set_aspect('equal', adjustable='box')

k = 1

track_len = 40
nb_tracks = 10000
batch_size = nb_tracks

MSDs1 = []
for k in range(1,21):
    MSDs1.append(np.mean(np.sum((est_samples[:, k:] - est_samples[:, :-k])**2, -1)))

MSDs2 = []
for k in range(1,21):
    MSDs2.append(np.mean(np.sum((all_tracks[:, k:] - all_tracks[:, :-k])**2, -1)))

plt.figure(figsize = (3.5,3.5))
plt.plot(np.arange(1,21), MSDs2)
plt.plot(np.arange(1,21), MSDs1)
plt.xlabel('Time step')
plt.ylabel('MSD')
plt.legend(['Original tracks', 'Generated by IRGP'])
plt.tight_layout()


MSDs1 = []
STDs1 = []
for k in range(1,21):
    MSDs1.append(np.mean(np.sum((est_samples[:, k:] - est_samples[:, :-k])**2, -1)))
    STDs1.append(np.std(np.sum((est_samples[:, k:] - est_samples[:, :-k])**2, -1)))

MSDs2 = []
STDs2 = []
for k in range(1,21):
    MSDs2.append(np.mean(np.sum((all_tracks[:, k:] - all_tracks[:, :-k])**2, -1)))
    STDs2.append(np.std(np.sum((all_tracks[:, k:] - all_tracks[:, :-k])**2, -1)))

plt.figure(figsize = (3.5,3.5))
#plt.errorbar(np.arange(1,21), MSDs2, yerr = STDs2, ecolor='k')
plt.errorbar(np.arange(1,21), MSDs1, yerr = STDs1, ecolor='k')
plt.xlabel('Time step')
plt.ylabel('MSD')
plt.legend(['Original tracks', 'Generated by IRGP'])
plt.tight_layout()




'''
Tracking with constraints to extract the models parameters 
'''

loc_err = 0.025 # localization error
d = 0.1 # diffusion length
l = 0.5 # confinement factor
q = 0.01 # change of the anomalous factor

params = tf.constant([loc_err, d, l, q], dtype = dtype)
initial_params = tf.constant([10, 0.01, 0.1], dtype = dtype)

def constraint_function(params, initial_params, dtype):
    
    print('test1')
    if params[2] > 0:
        # hidden vars:                              pos_x,           ano_pos_x,        pos_x,    ano_pos_x,
        hidden_vars = tf.stack([[[[(1-params[2])/params[1], params[2]/params[1], -1/params[1],            0]]],                    # Diffusion + anomalous drift
                                   [[[                   0,         1/params[3],            0, -1/params[3]]]],                    # Diffusion of the anomalous position
                                   [[[         1/params[0],                   0,            0,            0]]]])    # Localization error     
        print('test2')
    
        obs_vars = tf.stack([[[[0]]],
                             [[[0]]],
                             [[[-1/params[0]]]]])
        
        Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states), dtype = dtype)
        
        # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
        # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
    
        initial_hidden_vars = tf.stack([[[[    1/initial_params[0],                   0]]],
                                        [[[-1/initial_params[1],     1/initial_params[2]]]]])
        
        initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
        initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        initial_biases =  tf.zeros((nb_hidden_vars, 1, nb_states), dtype = dtype)
        
    else:
        # hidden vars:                   pos_x,   ano_pos_x,        pos_x,    ano_pos_x,
        hidden_vars = tf.stack([[[[1/params[1], 1/params[1], -1/params[1],            0]]],                    # Diffusion + anomalous drift
                                [[[          0, 1/params[3],            0, -1/params[3]]]],                    # Diffusion of the anomalous position
                                [[[1/params[0],           0,            0,            0]]]])    # Localization error     
        print('test2')
    
        obs_vars = tf.stack([[[[0]]],
                                [[[0]]],
                                [[[-1/params[0]]]]])
        
        Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states), dtype = dtype)
        
        # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
        # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
    
        initial_hidden_vars = tf.stack([[[[    1/initial_params[0],                   0]]],
                                        [[[ 0.0000000000001,     1/-params[2]]]]])
        
        initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
        initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
        initial_biases =  tf.zeros((nb_hidden_vars, 1, nb_states), dtype = dtype)

    return hidden_vars, obs_vars, Gaussian_stds, biases, initial_hidden_vars, initial_obs_vars, initial_Gaussian_stds, initial_biases

track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

ds = np.arange(0, 0.101,0.01)

est_ds = []

for d in ds:
    
    D = d**2/(2*0.02)
    #Brownian
    all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                               nb_tracks = nb_tracks,
                               LocErr=0.02, # localization error in x, y and z (even if not used)
                               Fs = np.array([1.]),
                               Ds = np.array([D]),
                               nb_dims = 2,
                               velocities = np.array([0.0]),
                               angular_Ds = np.array([0.]),
                               conf_forces = np.array([0.0]),
                               conf_Ds = np.array([0.0]),
                               conf_dists = np.array([0.0]),
                               LocErr_std = 0,
                               dt = 0.02,
                               nb_sub_steps = 10,
                               field_of_view = np.array([10,10]))
        
    all_tracks = np.array(all_tracks)
    tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)

        
    #inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])
    
    Init_layer = Initial_layer_constraints(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
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
    
    preds = model.predict(tracks, batch_size = batch_size)
    
    np.mean(preds)
    model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        
        max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
        reduced_LP = y_pred - max_LP
        pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
        
        return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis
    
    nb_epochs = 2000
    nb_data_points = batch_size*(track_len-1)
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history0 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history1 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 1000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    est_ds.append(Init_layer.param_vars[1])
    
plt.figure(figsize = (3,3))
plt.plot(ds, np.abs(est_ds), marker = '+', markersize = 6)
plt.grid()
plt.xlabel('True diffusion length [$\mathrm{\mu m}$]')
plt.xticks([0,0.02,0.04,0.06,0.08,0.1])
plt.ylabel('Estimated diffusion length [$\mathrm{\mu m}$]')
plt.tight_layout()

est_ds = (np.array([163.870909 , 149.046529 , 134.774799 , 120.363845 , 104.937013 , 91.466196 , 75.451213 , 61.644325 , 47.428587 , 33.026675 , 18.089091 ])-17.27261)[::-1]/(163.872384 - 127.222443 )*0.025


track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

vs = np.arange(0, 0.051,0.005)
est_vs = []

86.958017 - 106.876694
106.876694 - 126.795372

150.874088 - 135.871341
135.871341  -  121.257834 


est_vs = ((np.array([163.962577 ,150.874088,
135.871341 ,
121.257834 ,
106.866368 ,
92.420921 ,
76.397455 ,
64.670184 ,
49.278759 ,
31.037366 ,
20.01412])-17.666799 )/(
74.402035  - 44.534417 ))[::-1]*0.01


for v in vs:
    
    #Brownian
    # Directed
    all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                               nb_tracks = nb_tracks,
                               LocErr=0.02, # localization error in x, y and z (even if not used)
                               Fs = np.array([1.]),
                               Ds = np.array([0.0]),
                               nb_dims = 2,
                               velocities = np.array([v]),
                               angular_Ds = np.array([0.5]),
                               conf_forces = np.array([0.]),
                               conf_Ds = np.array([0.0]),
                               conf_dists = np.array([0.0]),
                               LocErr_std = 0,
                               dt = 0.02,
                               nb_sub_steps = 10,
                               field_of_view = np.array([10,10]))
    
    all_tracks = np.array(all_tracks)
    tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)

        
    #inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])
    
    Init_layer = Initial_layer_constraints(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
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
    
    preds = model.predict(tracks, batch_size = batch_size)
    
    np.mean(preds)
    model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        
        max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
        reduced_LP = y_pred - max_LP
        pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
        
        return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis
    
    nb_epochs = 2000
    nb_data_points = batch_size*(track_len-1)
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history0 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history1 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 1000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    est_vs.append(-Init_layer.param_vars[2]*2**0.5)
    
plt.figure(figsize = (3,3))
plt.plot(vs, np.abs(est_vs), marker = '+', markersize = 6)
plt.grid()
plt.xlabel('True velocity [$\mathrm{\mu m / step}$]')
plt.ylabel('Estimated velocity [$\mathrm{\mu m / step}$]')
plt.xticks([0,0.01,0.02,0.03,0.04,0.05])
plt.tight_layout()


track_len = 40
nb_tracks = 1000
batch_size = nb_tracks

ls = np.arange(0, 0.71, 0.05)
est_ls = []

from svgpathtools import svg2paths


for l in ls:
    
    cont_l = - np.log(1-l)

    # Confined
    all_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                                                   nb_tracks = nb_tracks,
                                                   LocErr=0.02, # localization error in x, y and z (even if not used)
                                                   Fs = np.array([1.]),
                                                   Ds = np.array([0.25]),
                                                   nb_dims = 2,
                                                   velocities = np.array([0.0]),
                                                   angular_Ds = np.array([0.0]),
                                                   conf_forces = np.array([cont_l]),
                                                   conf_Ds = np.array([0.0]),
                                                   conf_dists = np.array([0.0]),
                                                   LocErr_std = 0,
                                                   dt = 0.02,
                                                   nb_sub_steps = 10,
                                                   field_of_view = np.array([10,10]))
    
    all_tracks = np.array(all_tracks)
    tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)
    
    #inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, nb_obs_vars), dtype = dtype)
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])
    
    Init_layer = Initial_layer_constraints(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
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
    
    preds = model.predict(tracks, batch_size = batch_size)
    
    np.mean(preds)
    model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        
        max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
        reduced_LP = y_pred - max_LP
        pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
        
        return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis
    
    nb_epochs = 2000
    nb_data_points = batch_size*(track_len-1)
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history0 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history1 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=1/8000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
    
    with tf.device('/CPU:0'):
        history2 = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
    
    est_ls.append(Init_layer.param_vars[2])


plt.figure(figsize = (3,3))
plt.plot(ls[1:], np.abs(est_ls[1:]), marker = '+', markersize = 6)
plt.grid()
plt.xlabel('True confinement factor')
plt.ylabel('Estimated confinement factor')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
plt.tight_layout()


Init_layer.param_vars
Init_layer.initial_param_vars

nb_rows = 6
lim = 2

plt.figure(figsize = (10, 10))
for i in range(nb_rows):
    for j in range(nb_rows):
        track = all_tracks[i*nb_rows+j]
        track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], alpha = 1)
plt.gca().set_aspect('equal', adjustable='box')

est_ls = np.array([0.47442848, 0.04863508, 0.10391907, 0.1557708 , 0.2082419 ,
       0.27376067, 0.32712208, 0.3820418 , 0.4326196 , 0.49563431,
       0.54463087, 0.60434131, 0.65248234, 0.70157102, 0.75570055])

MSDs2 = []
STDs2 = []
for k in range(1,21):
    MSDs2.append(np.mean(np.sum((all_tracks[:, k:] - all_tracks[:, :-k])**2, -1)))
    STDs2.append(np.std(np.sum((all_tracks[:, k:] - all_tracks[:, :-k])**2, -1)))

plt.figure(figsize = (3.5,3.5))
plt.errorbar(np.arange(1,21), MSDs2, yerr = STDs2, ecolor='k')
plt.xlabel('Time step')
plt.ylabel('MSD')
plt.legend(['Original tracks', 'Generated by IRGP'])
plt.tight_layout()


- np.log(1-0.4)
























