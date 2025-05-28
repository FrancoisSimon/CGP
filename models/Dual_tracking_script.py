# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:41:04 2024

@author: franc
"""
import numpy as np
import tensorflow as tf
from CGP import Custom_RNN_layer, Final_layer, transpose_layer, Initial_layer_constraints
from matplotlib import pyplot as plt

def dual_tracking(track_len=100,
                  nb_tracks = 100,
                  LocErr=0.02, # localization error in x, y and z (even if not used)
                  Fs = np.array([1.]),
                  Ds = np.array([0.25]),
                  nb_dims = 2,
                  velocities = np.array([0.1]),
                  angular_Ds = np.array([0.1]),
                  conf_forces = np.array([0.2]),
                  conf_Ds = np.array([0.0]),
                  LocErr_std = 0,
                  dt = 0.02,
                  nb_sub_steps = 10,
                  field_of_view = np.array([10,10])):
    
    '''
    Function to simulate the interactions between a particle in directed motion and
    another particle confined around the first one.
    '''
    nb_states = len(velocities)
    if not np.all(np.array([len(Fs), len(Ds), len(velocities), len(angular_Ds), len(conf_forces), len(conf_Ds)]) == nb_states):
        raise ValueError('Fs, Ds, velocities, angular_Ds, conf_forces, conf_Ds and conf_dists must all be 1D arrays of the same number of element (one element per state)')
    # diff + persistent motion + elastic confinement
    conf_sub_forces = conf_forces / nb_sub_steps
    sub_dt = dt / nb_sub_steps
   
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
    
    tracks = []
    anchors = []
    
    all_states = np.zeros(nb_tracks)
    
    for kkk in range(nb_tracks):
        
        state = np.argmin(np.random.rand()>cum_Fs)
        all_states[kkk] = state
        D, velocity, angular_D, conf_sub_force, conf_D = (Ds[state], velocities[state], angular_Ds[state], conf_sub_forces[state], conf_Ds[state])
       
        positions = np.zeros((track_len * nb_sub_steps, nb_dims))
        
        positions[0] = np.random.rand(nb_dims)*field_of_view
        disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
        
        d_angles = np.random.normal(0, 1, ((track_len) * nb_sub_steps)-1) * (2*angular_D*sub_dt)**0.5
        angles = np.zeros((track_len * nb_sub_steps-1))
        angles[0] = np.random.rand()*2*np.pi
        for i in range(1, len(d_angles)):
            angles[i] = angles[i-1] + d_angles[i]
       
        if conf_sub_force>0:
            conf_dist = np.sqrt(2*D*sub_dt)/(2*conf_sub_force)**0.5
        else:
            conf_dist = 0.05
            
        anchor_positions = np.random.normal(0, np.sqrt(2*conf_D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
        anchor_positions[0] = positions[0] + np.random.normal(0,conf_dist, nb_dims)
       
        for i in range(1, len(anchor_positions)):
            angle = angles[i-1]
            pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
            anchor_positions[i] += anchor_positions[i-1] + pesistent_disp
       
        for i in range(len(positions)-1):
            positions[i+1] = positions[i] + disps[i]
            positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
       
        final_track = np.zeros((track_len, nb_dims))
        final_anchor_positions = np.zeros((track_len, nb_dims))
        for i in range(track_len):
            final_track[i] = positions[i*nb_sub_steps]
            final_anchor_positions[i] = anchor_positions[i*nb_sub_steps]
            
        final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
        final_anchor_positions += np.random.normal(0, LocErr, (track_len, nb_dims))
       
        tracks.append(final_track)
        anchors.append(final_anchor_positions)
        
    return np.array(tracks), np.array(anchors), all_states

dtype = 'float64'


'''
Constrainted dual tracking to predict the confinement factor
'''

nb_states = 1
nb_obs_vars = 2
nb_hidden_vars = 3
nb_gaussians = nb_hidden_vars + nb_obs_vars

loc_err = 0.025 # localization error
d = 0.15 # diffusion length
l = 0.02 # confinement factor
velocity = 0.07
q = 0.01 # change of the anomalous factor
d_well = 0.001

params = tf.constant([loc_err, d, l, q, d_well], dtype = dtype)
initial_params = tf.constant([10, 0.05, velocity], dtype = dtype)

def constraint_function(params, initial_params, dtype):
    
    #hidden variables:  position of the particle, position of the well, drift of the well
    hidden_vars = tf.stack([[[[1/params[0], 0, 0,  0, 0, 0]]],
                            [[[0, 1/params[0], 0,  0, 0, 0]]],
                            [[[0, 0, 1/params[3], 0, 0, -1/params[3]]]],
                            [[[0, 1/params[4], 1/params[4],  0, -1/params[4], 0]]],
                            [[[(1-params[2])/params[1], params[2]/params[1], 0,  -1/params[1], 0, 0]]]])    # Localization error     
    print('test2')

    obs_vars = tf.stack([[[[-1/params[0],           0]]],
                         [[[          0.,-1/params[0]]]],
                         [[[           0,           0]]],
                         [[[           0,           0]]],
                         [[[           0,           0]]]])
    
    Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states), dtype = dtype)
    
    # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
    #hidden variables:  position of the particle, position of the well, drift of the well    
    initial_hidden_vars = tf.stack([[[[1/initial_params[0],       0,    0]]],
                                    [[[1/initial_params[1], -1/initial_params[1],    0]]],
                                    [[[0, 0, 1/initial_params[2]]]]])
    
    initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
    initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    initial_biases =  tf.zeros((nb_hidden_vars, 1, nb_states), dtype = dtype)

    return hidden_vars, obs_vars, Gaussian_stds, biases, initial_hidden_vars, initial_obs_vars, initial_Gaussian_stds, initial_biases

track_len = 30
nb_tracks = 1000
batch_size = nb_tracks
nb_dims = 2

ls = np.arange(0, 1.01,0.1)

est_ls = []

for l in ls:
    #D = d**2/(2*0.02)
        #Brownian
    
    tracks1, anchors, all_states = dual_tracking(track_len = track_len,
                                                  nb_tracks = nb_tracks,
                                                  LocErr = 0.02, # localization error in x, y and z (even if not used)
                                                  Fs = np.array([1.]),
                                                  Ds = np.array([0.25]),
                                                  nb_dims = nb_dims,
                                                  velocities = np.array([0.05]),
                                                  angular_Ds = np.array([0.1]),
                                                  conf_forces = np.array([l]),
                                                  conf_Ds = np.array([0.0]),
                                                  LocErr_std = 0,
                                                  dt = 0.02,
                                                  nb_sub_steps = 10,
                                                  field_of_view = np.array([10,10]))
    
    tracks1 = tracks1.transpose([0, 2, 1]).reshape(nb_tracks*nb_dims, track_len)
    anchors = anchors.transpose([0, 2, 1]).reshape(nb_tracks*nb_dims, track_len)
    tracks = tf.stack([tracks1[:,None, :, None], anchors[:,None, :, None]], axis=4)
    
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
    
    est_ls.append(Init_layer.param_vars[2])

est_ls = np.array(est_ls)

plt.figure(figsize = (3,3))
plt.plot(ls, - np.log(1-est_ls)) 
plt.grid()
plt.xlabel('True confinement factor')
plt.ylabel('Estimated confinement factor')
plt.tight_layout()







