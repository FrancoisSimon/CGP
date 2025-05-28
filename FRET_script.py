# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:43:50 2025

@author: Franc
"""

import numpy as np
import tensorflow as tf
from CGP import Initial_layer_constraints, Custom_RNN_layer, Final_layer, anomalous_diff_mixture, transpose_layer
from matplotlib import pyplot as plt

def FRET_simulation(nb_tracks = 100,
                    track_length = 50,
                    mean_emitted_photon_donors = 200,
                    Foster_radius = 0.006,
                    equilibrium_distance = 0.007,
                    AD_diffusion_length = 0.001,
                    confinement_factor = 0.5, 
                    nb_sub_steps = 4,
                    measurement_error = 10,
                    camera_yield_donor = 0.7,
                    camera_yield_acceptor = 0.75):
    
    print('Confinement standard deviation:', AD_diffusion_length/(2*confinement_factor)**0.5)
    
    sub_confinement_factor = confinement_factor / nb_sub_steps
    
    dt = 1
    D = AD_diffusion_length**2/2
    sub_dt = dt / nb_sub_steps
   
    all_tracks = np.zeros((nb_tracks, track_length, 6)) # dim 3: FD, FA, dist, efficiency, donor_photons, emitor_photons
    
    for kkk in range(nb_tracks):
        
        distances = np.zeros((track_length * nb_sub_steps))
        distances[0] = np.max([np.random.normal(equilibrium_distance, AD_diffusion_length/(2*confinement_factor)**0.5), 0])
        disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_length) * nb_sub_steps - 1))

        for i in range(len(distances)-1):
            distances[i+1] = distances[i] + disps[i]
            distances[i+1] = (1-sub_confinement_factor) *  distances[i+1] + sub_confinement_factor * equilibrium_distance
       
        distances = distances[::nb_sub_steps]
        efficiencies = 1 / (1 + (distances/Foster_radius)**6)

        '''
        plt.figure()
        plt.plot(distances)
         
        
        plt.figure()
        plt.plot(efficiencies)
        
        plt.figure()
        plt.plot(np.linspace(0, 0.02, 50), 1 / (1 + (np.linspace(0, 0.02, 50)/Foster_radius)**6))
        '''
        emitted_photon_donors = np.random.poisson(mean_emitted_photon_donors, track_length)
        emitted_photon_acceptors = np.random.binomial(emitted_photon_donors, efficiencies)
        
        FD = camera_yield_donor * (emitted_photon_donors - 0.8*emitted_photon_acceptors) + np.random.chisquare(measurement_error, track_length) #   np.random.poisson(measurement_error, track_length)* np.random.normal(1, 0.05, track_length)
        FA = camera_yield_acceptor * emitted_photon_acceptors + np.random.chisquare(measurement_error, track_length) #   np.random.poisson(measurement_error, track_length)* np.random.normal(1, 0.05, track_length)
        
        all_tracks[kkk, :, 0] = FD
        all_tracks[kkk, :, 1] = FA
        all_tracks[kkk, :, 2] = distances
        all_tracks[kkk, :, 3] = efficiencies
        all_tracks[kkk, :, 4] = emitted_photon_donors
        all_tracks[kkk, :, 5] = emitted_photon_acceptors
    
        '''
        plt.figure()
        plt.plot(FD)
        plt.plot(FA)
        
        plt.figure()
        plt.scatter(FD, FA)
        '''
    return all_tracks


nb_tracks = 1000    
track_len = 100

pred_ls = []
pred_states = []

for l in 2**np.arange(-6.,7):
    for k in range(10):
        try: 
            all_tracks = FRET_simulation(nb_tracks = nb_tracks,
                                track_length = track_len,
                                mean_emitted_photon_donors = 200,
                                Foster_radius = 0.006,
                                equilibrium_distance = 0.0065,
                                AD_diffusion_length = 0.001,
                                confinement_factor = l, 
                                nb_sub_steps = 200,
                                measurement_error = 10,
                                camera_yield_donor = 0.7,
                                camera_yield_acceptor = 0.75)
                    
                    
            nb_gaussians = 3
            nb_states = 1
            nb_obs_vars = 2
            nb_hidden_vars = 1
            batch_size = nb_tracks
            
            dtype = 'float64'
            
            hidden_vars = np.array([[[[0.5, -1]]],
                                    [[[1,     0]]],
                                    [[[-1,     0]]]], dtype = dtype)*(0.95+0.1*np.random.rand(nb_gaussians, 1, 1, nb_hidden_vars*2))
            
            obs_vars = np.array([[[[0, 0]]],
                                 [[[1, 0]]],
                                 [[[0, 1]]]], dtype = dtype)
            
            indices_hidden_vars = np.array(np.where(hidden_vars != 0)).T
            indices_obs_vars = np.array(np.where(obs_vars != 0)).T
            
            hidden_var_coef_values = hidden_vars[np.where(hidden_vars != 0)]
            obs_var_coef_values = obs_vars[np.where(obs_vars != 0)]
            
            Gaussian_stds = np.array([[[[1]]],
                                      [[[1]]],
                                      [[[1]]]], dtype = dtype)
            
            biases = np.array([[[  1.]],
                               [[  1]],
                               [[  1.]]], dtype = dtype)
            
            # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
            initial_hidden_vars = np.array([[[[1]]]], dtype = dtype)
            
            initial_obs_vars = np.array([[[[0, 0]]]], dtype = dtype)
            
            indices_initial_hidden_vars = np.array(np.where(initial_hidden_vars != 0)).T
            indices_initial_obs_vars = np.array(np.where(initial_obs_vars != 0)).T
            
            initial_obs_var_coef_values = indices_initial_obs_vars[np.where(indices_initial_obs_vars != 0)].astype(dtype)
            
            initial_hidden_var_coef_values = initial_hidden_vars[np.where(initial_hidden_vars != 0)]
            
            initial_Gaussian_stds = np.array([[[[1]]]], dtype = dtype)
            initial_biases = np.array([[[  0.]]], dtype = dtype)
            
            
            tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)
            
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
                history = model.fit(tracks, tracks, epochs = 1500, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
            model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
            
            with tf.device('/CPU:0'):
                history = model.fit(tracks, tracks, epochs = 500, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
            model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
            
            with tf.device('/CPU:0'):
                history = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
            model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
            
            with tf.device('/CPU:0'):
                history = model.fit(tracks, tracks, epochs = 2000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            844.2018
            est_obs_var_coef_values, est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_stds, est_initial_obs_var_coefs_values, est_initial_stds = model.layers[3].weights
            #est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_obs_var_coef_values, est_stds, est_initial_obs_var_coefs_values, est_initial_stds = model.layers[3].weights
            
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
            
            conf = np.array([est_hidden_var_coefs[0,0,0,0],  est_hidden_var_coefs[0,0,0,1], est_biases[0,0,0]])
            d = 1/np.abs(conf[1])
            conf = conf/conf[1]
            est_l = conf[0]+1
            corr_l = - np.log(1 - est_l)
            h = conf[2]/est_l
            
            radius = d/(2*est_l)**0.5
            
            pred_ls.append(est_l)
            pred_states.append([est_obs_var_coef_values, est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_stds, est_initial_obs_var_coefs_values, est_initial_stds])
            
        except:
                    
            all_tracks = FRET_simulation(nb_tracks = nb_tracks,
                                track_length = track_len,
                                mean_emitted_photon_donors = 200,
                                Foster_radius = 0.006,
                                equilibrium_distance = 0.0065,
                                AD_diffusion_length = 0.001,
                                confinement_factor = l, 
                                nb_sub_steps = 200,
                                measurement_error = 10,
                                camera_yield_donor = 0.7,
                                camera_yield_acceptor = 0.75)
                    
                    
            nb_gaussians = 3
            nb_states = 1
            nb_obs_vars = 2
            nb_hidden_vars = 1
            batch_size = nb_tracks
            
            dtype = 'float64'
            
            hidden_vars = np.array([[[[0.5, -1]]],
                                    [[[1,     0]]],
                                    [[[-1,     0]]]], dtype = dtype)*(0.95+0.1*np.random.rand(nb_gaussians, 1, 1, nb_hidden_vars*2))
            
            obs_vars = np.array([[[[0, 0]]],
                                 [[[1, 0]]],
                                 [[[0, 1]]]], dtype = dtype)
            
            indices_hidden_vars = np.array(np.where(hidden_vars != 0)).T
            indices_obs_vars = np.array(np.where(obs_vars != 0)).T
            
            hidden_var_coef_values = hidden_vars[np.where(hidden_vars != 0)]
            obs_var_coef_values = obs_vars[np.where(obs_vars != 0)]
            
            Gaussian_stds = np.array([[[[1]]],
                                      [[[1]]],
                                      [[[1]]]], dtype = dtype)
            
            biases = np.array([[[  1.]],
                               [[  1]],
                               [[  1.]]], dtype = dtype)
            
            # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
            initial_hidden_vars = np.array([[[[1]]]], dtype = dtype)
            
            initial_obs_vars = np.array([[[[0, 0]]]], dtype = dtype)
            
            indices_initial_hidden_vars = np.array(np.where(initial_hidden_vars != 0)).T
            indices_initial_obs_vars = np.array(np.where(initial_obs_vars != 0)).T
            
            initial_obs_var_coef_values = indices_initial_obs_vars[np.where(indices_initial_obs_vars != 0)].astype(dtype)
            
            initial_hidden_var_coef_values = initial_hidden_vars[np.where(initial_hidden_vars != 0)]
            
            initial_Gaussian_stds = np.array([[[[1]]]], dtype = dtype)
            initial_biases = np.array([[[  0.]]], dtype = dtype)
            
            
            tracks = tf.constant(all_tracks[:,None, :, None, :nb_obs_vars], dtype)
            
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
                history = model.fit(tracks, tracks, epochs = 1500, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            adam = tf.keras.optimizers.Adam(learning_rate=1/100, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
            model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
            
            with tf.device('/CPU:0'):
                history = model.fit(tracks, tracks, epochs = 500, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            adam = tf.keras.optimizers.Adam(learning_rate=1/500, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
            model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
            
            with tf.device('/CPU:0'):
                history = model.fit(tracks, tracks, epochs = 3000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            adam = tf.keras.optimizers.Adam(learning_rate=1/2000, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
            model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)
            
            with tf.device('/CPU:0'):
                history = model.fit(tracks, tracks, epochs = 2000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])
            
            844.2018
            est_obs_var_coef_values, est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_stds, est_initial_obs_var_coefs_values, est_initial_stds = model.layers[3].weights
            #est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_obs_var_coef_values, est_stds, est_initial_obs_var_coefs_values, est_initial_stds = model.layers[3].weights
            
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
            
            conf = np.array([est_hidden_var_coefs[0,0,0,0],  est_hidden_var_coefs[0,0,0,1], est_biases[0,0,0]])
            d = 1/np.abs(conf[1])
            conf = conf/conf[1]
            est_l = conf[0]+1
            corr_l = - np.log(1 - est_l)
            h = conf[2]/est_l
            
            radius = d/(2*est_l)**0.5
            
            pred_ls.append(est_l)
            pred_states.append([est_obs_var_coef_values, est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_stds, est_initial_obs_var_coefs_values, est_initial_stds])
            
'''
Obtained results:
pred_ls = [0.032286241435317775,
     0.03030749484745765,
     0.03102003075440518,
     0.03290709836942274,
     0.031047232628690802,
     0.03166951296405307,
     0.03271809552301164,
     0.030962475077165097,
     0.03307037573688332,
     0.0334576381296704,
     0.044442893949207085,
     0.045418064731962304,
     0.04603665852738248,
     0.04714978871502851,
     0.047923535807662065,
     0.04627999339173261,
     0.04453611373172606,
     0.04574984373202262,
     0.04682665637846639,
     0.047087856772473735,
     0.07715200979326331,
     0.07406495465038476,
     0.07786649650131738,
     0.07726277254726455,
     0.07556762314189769,
     0.07264309501654886,
     0.07661603680837026,
     0.07519876155630723,
     0.07597317809945892,
     0.0754326737334251,
     0.13202617998878075,
     0.13037390589088305,
     0.12717060383486545,
     0.12888705937111544,
     0.13224283674497284,
     0.13152760633094962,
     0.13193060395468637,
     0.13021716541209505,
     0.12875870611063311,
     0.12946562963706454,
     0.22976911889275298,
     0.22822404348605319,
     0.22975429497029087,
     0.22583630188773052,
     0.23235363850063961,
     0.22847998881236653,
     0.22875284733329437,
     0.22937189400318325,
     0.22787270259850312,
     0.22918568239019643,
     0.3943161622556792,
     0.39339261068851317,
     0.3932979261027464,
     0.39949601224372655,
     0.39814804288723116,
     0.3908427873865603,
     0.39398223691903556,
     0.39192920338516835,
     0.39685307672388614,
     0.3893036126716638,
     0.6544386202320001,
     0.6659138304282661,
     0.6553034386728618,
     0.6639619081475214,
     0.6659487235375934,
     0.6584055814649827,
     0.6505764009368242,
     0.6584327836287722,
     0.6583906060111573,
     0.6677955165558549,
     0.913043633593972,
     0.913866950418282,
     0.9149396267502014,
     0.9075824874394353,
     0.9162699493610167,
     0.9103727676504244,
     0.9150404247655748,
     0.9114351250405559,
     0.9130033144324152,
     0.9109728703628303,
     0.9881238124134676,
     0.9868296547402332,
     0.9962185143016484,
     0.9906821617628457,
     0.9874756401727574,
     0.9938051339546083,
     0.9890318002611613,
     0.9953778575848962,
     0.9934354512493667,
     0.9869256664996772,
     1.0683739459711485,
     1.0568316119816632,
     1.0640005785311772,
     1.0406645862087918,
     1.0604236704735148,
     1.0683739459711485,
     1.0568316119816632,
     1.0640005785311772,
     1.0406645862087918,
     1.0604236704735148]
'''

plt.figure()
plt.plot(np.mean(pred_ls, 1), 1-np.exp(-2**np.arange(-6.,7)[:]))
plt.ylabel('Predicted confinement factor')
plt.xlabel('True confinement factor')
plt.grid()
plt.tight_layout()














