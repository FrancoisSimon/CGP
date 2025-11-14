# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:21:26 2024

@author: francois
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN
#from tensorflow.python.keras.layers.recurrent import RNN

dtype = 'float64'
pi = tf.constant(np.pi, dtype = dtype)
minval = 1e-8

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


@tf.function(jit_compile=True)
def log_gaussian(top, variance=tf.constant(1, dtype = dtype)):
    #if complex_number:
    #    #sum_top = np.sum(top)
    #    return tf.math.real(- 0.5*tf.math.log(2*pi*variance) - top*tf.math.conj(top)/(2*variance))
    #else:
    return - 0.5*tf.math.log(2*pi*variance) - top**2/(2*variance)

@tf.function(jit_compile=True)
def norm_log_gaussian(top):
    #if complex_number:
    #    #sum_top = np.sum(top)
    #    return tf.math.real(- 0.5*(tf.math.log(2*pi) + top*tf.math.conj(top)))
    #else:
    return - 0.5*(tf.math.log(2*pi) + top**2)

#current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2, coef_index
#@tf.function(jit_compile=True)

#np.any(np.array([100] + list(np.random.normal(0, 1e-100, 10000000)))==0)

@tf.function(jit_compile=True)
def RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2, coef_index):
    C1 = current_hidden_var_coefs_1[:,:, coef_index:coef_index+1] + tf.random.normal([1,1,1], 0, 1e-20, dtype = dtype)
    C2 = current_hidden_var_coefs_2[:,:, coef_index:coef_index+1] + tf.random.normal([1,1,1], 0, 1e-20, dtype = dtype)
    '''
    C1 = tf.math.sign(C1) * tf.clip_by_value(tf.math.abs(C1), clip_value_min=1e-50, clip_value_max=np.inf)
    C2 = tf.sign(C2) * tf.clip_by_value(tf.math.abs(C2), clip_value_min=1e-20, clip_value_max=np.inf)
    C2[0]
    C1 = tf.math.divide_no_nan(C1**2, C1)
    '''
    
    current_coefs1 = tf.math.divide_no_nan(current_hidden_var_coefs_1, C1)
    current_coefs2 = tf.math.divide_no_nan(current_hidden_var_coefs_2, C2)
    
    next_coefs1 = tf.math.divide_no_nan(next_hidden_var_coefs_1, C1)
    next_coefs2 = tf.math.divide_no_nan(next_hidden_var_coefs_2, C2)
    biases1 = tf.math.divide_no_nan(biases_1, C1)
    biases2 = tf.math.divide_no_nan(biases_2, C2)
    
    var1 = 1./(C1**2 + tf.random.normal([1,1,1], 0, 1e-100, dtype = dtype))
    var2 = 1./(C2**2 + tf.random.normal([1,1,1], 0, 1e-100, dtype = dtype))
    
    var3 = var1 + var2
    std3 = var3**0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3
    biases3 = (biases1 - biases2)/std3
    
    var4 = var1 * var2 / var3
    std4 = var4**0.5
    current_coefs4 = (current_coefs1*var2 + current_coefs2*var1)/(var3*std4)
    next_coefs4 = (next_coefs1*var2 + next_coefs2*var1)/(var3*std4)
    
    biases4 = (biases1*var2 + biases2*var1)/(var3*std4)
    
    LogConstant = -tf.math.log(tf.math.abs(C1*C2*std4*std3))[:,:,0]
    return LogConstant, current_coefs3, current_coefs4, next_coefs3, next_coefs4, biases3, biases4

"""
@tf.function(jit_compile=False)
def RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2, coef_index=0,  Complex = False, dtype = 'float64'):
    '''
    product of 2 gaussians 1 to simplify the expression depending on the current hidden variable
    defined by its index coef_index:
    
    the new first gaussian will be independent of the current hidden variable and the second gaussian 
    will be a weighted sum of the 2 initial gaussians
    '''
    
    C1 = current_hidden_var_coefs_1[:,:, coef_index:coef_index+1]
    C2 = current_hidden_var_coefs_2[:,:, coef_index:coef_index+1]
    if tf.math.equal(C1[0,0,0], tf.constant(0, dtype = dtype, shape = ())): # if C1 = 0, the goal is already acheived so we don't need to change anything
        # in principle, if some of the C1 or C2 values = 0, we should pass the iteration but it can't be done from a tensor point-of-view 
        return tf.math.real(tf.constant(0, dtype = dtype, shape = C1.shape[:2])), current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2
    elif tf.math.equal(C2[0,0,0], tf.constant(0, dtype = dtype, shape = ())): # if C1 is not 0 but C2 is, we just swap the gaussians and the goal is acheived
        #current_hidden_var_coefs[Gaussian_ID_1], current_hidden_var_coefs[Gaussian_ID_2] = current_hidden_var_coefs[Gaussian_ID_2], current_hidden_var_coefs[Gaussian_ID_1]
        #next_hidden_var_coefs[Gaussian_ID_1], next_hidden_var_coefs[Gaussian_ID_2] = next_hidden_var_coefs[Gaussian_ID_2], next_hidden_var_coefs[Gaussian_ID_1]
        #biases[Gaussian_ID_1], biases[Gaussian_ID_2] = biases[Gaussian_ID_2], biases[Gaussian_ID_1]
        return tf.math.real(tf.constant(0, dtype = dtype, shape = C1.shape[:2])), current_hidden_var_coefs_2, current_hidden_var_coefs_1, next_hidden_var_coefs_2, next_hidden_var_coefs_1, biases_2, biases_1
    else:
        current_coefs1 = tf.math.divide_no_nan(current_hidden_var_coefs_1, C1)
        current_coefs2 = tf.math.divide_no_nan(current_hidden_var_coefs_2, C2)
        next_coefs1 = tf.math.divide_no_nan(next_hidden_var_coefs_1, C1)
        next_coefs2 = tf.math.divide_no_nan(next_hidden_var_coefs_2, C2)
        biases1 = tf.math.divide_no_nan(biases_1, C1[:,:,0])
        biases2 = tf.math.divide_no_nan(biases_2, C2[:,:,0])
        
        if Complex:
            var1 = 1./(C1*tf.math.conj(C1))  # for complex numbers, else C1**2
            var2 = 1./(C2*tf.math.conj(C2))  # C2**2
        else:
            var1 = 1./C1**2
            var2 = 1./C2**2
        
        var3 = var1 + var2
        std3 = var3**0.5
        current_coefs3 = (current_coefs1 - current_coefs2) / std3
        next_coefs3 = (next_coefs1 - next_coefs2) / std3
        biases3 = (biases1 - biases2)/std3[:,:,0]
        
        var4 = var1 * var2 / var3
        std4 = var4**0.5
        current_coefs4 = (current_coefs1*var2 + current_coefs2*var1)/(var3*std4)
        next_coefs4 = (next_coefs1*var2 + next_coefs2*var1)/(var3*std4)
    
        biases4 = (biases1*var2[:,:,0] + biases2*var1[:,:,0])/(var3*std4)[:,:,0]
        
        LogConstant = -tf.math.log(tf.math.abs(C1*C2*std4*std3))[:,:,0]
        return LogConstant, current_coefs3, current_coefs4, next_coefs3, next_coefs4, biases3, biases4
"""
#kept_next_hidden_var_coefs = kept_next_hidden_var_coefs_cp
#kept_biases = kept_biases_cp

@tf.function(jit_compile=True)
def intermediate_RNN_function(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases):
    
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)
    
    current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2 = current_hidden_var_coefs_cp[ID_1], current_hidden_var_coefs_cp[ID_2], next_hidden_var_coefs_cp[ID_1], next_hidden_var_coefs_cp[ID_2], biases_cp[ID_1], biases_cp[ID_2]
    LogConstant, current_coefs3, current_coefs4, next_coefs3, next_coefs4, biases3, biases4 = RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2, coef_index)
    
    current_hidden_var_coefs_cp[ID_1] = tf.identity(current_coefs3)
    current_hidden_var_coefs_cp[ID_2] = tf.identity(current_coefs4)
    next_hidden_var_coefs_cp[ID_1] = tf.identity(next_coefs3)
    next_hidden_var_coefs_cp[ID_2] = tf.identity(next_coefs4)
    biases_cp[ID_1] = tf.identity(biases3)
    biases_cp[ID_2] = tf.identity(biases4)
    #if not get_scaling_factor_only:
    #print(Complex, LC, LogConstant)
    LC +=  LogConstant
    
    return tf.stack(current_hidden_var_coefs_cp), tf.stack(next_hidden_var_coefs_cp), tf.stack(biases_cp), LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases

@tf.function(jit_compile=True, reduce_retracing=True)
def final_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases):
    
    current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases = intermediate_RNN_function(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases)
    
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs_cp)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs_cp)
    biases_cp = tf.unstack(biases_cp)
    
    LC += - tf.math.log(tf.abs(current_hidden_var_coefs_cp[ID_2][:,:,coef_index])) # we must first normalize the integrated variable, log_gaussian(xs*coefs_matrix[2], 1) == log_gaussian(xs*coefs_matrix[2]/a, 1/a**2) - np.log(a)
    
    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    
    nb_gaussians += -1
    
    return tf.stack(current_hidden_var_coefs_cp), tf.stack(next_hidden_var_coefs_cp), tf.stack(biases_cp), LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases

@tf.function(jit_compile=False, reduce_retracing=True)
def no_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases):
    #kept_next_hidden_var_coefs = kept_next_hidden_var_coefs_cp
    #kept_biases = kept_biases_cp
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)
    
    LC += - tf.math.log(tf.abs(current_hidden_var_coefs_cp[ID_2][:,:,coef_index])) # we must first normalize the integrated variable, log_gaussian(xs*coefs_matrix[2], 1) == log_gaussian(xs*coefs_matrix[2]/a, 1/a**2) - np.log(a)
    
    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    
    nb_gaussians += -1
    
    return tf.stack(current_hidden_var_coefs_cp), tf.stack(next_hidden_var_coefs_cp), tf.reshape(tf.stack(biases_cp), (len(biases_cp), biases.shape[1], biases.shape[2], biases.shape[3])), LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases

@tf.function(jit_compile=True, reduce_retracing=True)
def final_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases):
    
    next_hidden_var_coefs_cp, current_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases = intermediate_RNN_function(next_hidden_var_coefs, current_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases)
    
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs_cp)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs_cp)
    biases_cp = tf.unstack(biases_cp)
    
    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)
    
    kept_next_hidden_var_coefs_cp = tf.unstack(kept_next_hidden_var_coefs)
    kept_biases_cp = tf.unstack(kept_biases)
    
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    
    nb_gaussians += -1
    
    return tf.stack(next_hidden_var_coefs_cp), tf.stack(current_hidden_var_coefs_cp), tf.stack(biases_cp), LC, nb_gaussians, tf.stack(kept_next_hidden_var_coefs_cp), tf.stack(kept_biases_cp)
#kept_next_hidden_var_coefs = kept_next_hidden_var_coefs_cp
#kept_biases = kept_biases_cp

#np.max(current_hidden_var_coefs_cp)

@tf.function(jit_compile=True, reduce_retracing=True)
def no_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases):
    
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)
    
    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)
    
    kept_next_hidden_var_coefs_cp = tf.unstack(kept_next_hidden_var_coefs)
    kept_biases_cp = tf.unstack(kept_biases)
    
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    
    nb_gaussians += -1
    
    return tf.stack(next_hidden_var_coefs_cp), current_hidden_var_coefs, tf.stack(biases_cp), LC, nb_gaussians, tf.stack(kept_next_hidden_var_coefs_cp), tf.stack(kept_biases_cp)


@tf.function(jit_compile=False, reduce_retracing=True)
def RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                           next_hidden_var_coefs,
                           biases,
                           sequence_phase_1,
                           sequence_phase_2,
                           dtype = 'float64'): # False by default, set to true when aiming to compute the scaling factor
    '''
    We first integrate over the current hidden variables. To do so, we use RNN_gaussian_product
    to reduce the number of gaussians that depend on the current hidden variable to 1. Once this
    is done, we can simply remove the last gaussian.
    '''
    current_hidden_var_coefs_cp = tf.identity(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.identity(next_hidden_var_coefs)
    biases_cp = tf.identity(biases)
    
    kept_next_hidden_var_coefs_cp, kept_biases_cp = [[],[]]
    
    nb_gaussians = biases_cp.shape[0]
    
    nb_hidden_variables = current_hidden_var_coefs_cp[0].shape[-1]
    
    LC = tf.constant(0, shape = current_hidden_var_coefs_cp[0].shape[:2], dtype = dtype)
    
    #print('LC1',LC)
    for f, s in zip(sequence_phase_1[0], sequence_phase_1[1]):
        print('1...')
        
        coef_index, ID_1, ID_2 = s
        current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp = f(current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp)
        #tf.debugging.check_numerics(current_hidden_var_coefs_cp, "current_hidden_var_coefs_cp has NaN or Inf")
        #tf.debugging.check_numerics(next_hidden_var_coefs_cp, "next_hidden_var_coefs_cp has NaN or Inf")
        #tf.debugging.check_numerics(biases_cp, "biases_cp has NaN or Inf")
        #tf.debugging.check_numerics(LC, "LC has NaN or Inf")
        #tf.debugging.check_numerics(kept_next_hidden_var_coefs_cp, "kept_next_hidden_var_coefs_cp has NaN or Inf")
        #tf.debugging.check_numerics(kept_biases_cp, "kept_biases_cp has NaN or Inf")

    #print('remove LC', LC)
    #print('LC2',LC)
    
    '''
    Once the integration is done, all the current_hidden_var_coefs_cp are 0 and we 
    have nb_gaussians - nb_hidden_variables variables left. If that number is higher than 
    nb_hidden_variables, we have redundancies that we can eliminate. To eliminate them 
    we can perform RNN_gaussian_product inverting current_hidden_var_coefs_cp and next_hidden_var_coefs_cp
    on the nb_remaining_gaussians - nb_hidden_variables + 1 first gaussians to set the
    nb_remaining_gaussians - nb_hidden_variables next hidden variables to 0 and obtain 
    a final number of gaussians equal to nb_hidden_variables
    '''
    # f = sequence_phase_2[0][6]
    # s = sequence_phase_2[1][6]
    for f, s in zip(sequence_phase_2[0][:], sequence_phase_2[1][:]):
        print('2...')
        coef_index, ID_1, ID_2 = s
        next_hidden_var_coefs_cp, current_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp = f(next_hidden_var_coefs_cp, current_hidden_var_coefs_cp, biases_cp, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp)
        #tf.debugging.check_numerics(current_hidden_var_coefs_cp, "current_hidden_var_coefs_cp has NaN or Inf")
        #tf.debugging.check_numerics(next_hidden_var_coefs_cp, "next_hidden_var_coefs_cp has NaN or Inf")
        #tf.debugging.check_numerics(biases_cp, "biases_cp has NaN or Inf")
        #tf.debugging.check_numerics(LC, "LC has NaN or Inf")
        #tf.debugging.check_numerics(kept_next_hidden_var_coefs_cp, "kept_next_hidden_var_coefs_cp has NaN or Inf")
        #tf.debugging.check_numerics(kept_biases_cp, "kept_biases_cp has NaN or Inf")
    
    new_LCs = norm_log_gaussian(tf.cast(tf.stack(biases_cp), dtype = dtype))
    
    LC += tf.math.reduce_sum(new_LCs, (0, 3))
    LC.shape
    #Next_coefs = tf.reshape(Next_coefs, [next_nb_gaussians] + shape)
    #Next_biases = tf.reshape(Next_biases, [next_nb_gaussians] + shape[:-1])
    Next_coefs = tf.stack(kept_next_hidden_var_coefs_cp)
    Next_biases = tf.stack(kept_biases_cp)
    
    return Next_coefs, Next_biases, LC
#current_hidden_var_coefs_cp[4].shape


class Initial_layer(tf.keras.layers.Layer):
    def __init__(
        self,
        obs_var_coef_values,
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
        trainable_params = {'obs': True, 'hidden': True, 'stds': True, 'biases': True},
        trainable_initial_params = {'obs': False, 'hidden': False, 'stds': False, 'biases': False},
        **kwargs):
        
        self.obs_var_coef_values = obs_var_coef_values
        self.indices_obs_vars = indices_obs_vars
        self.hidden_var_coef_values = hidden_var_coef_values
        self.indices_hidden_vars = indices_hidden_vars
        self.Gaussian_stds = Gaussian_stds
        self.biases = biases
        self.nb_states = nb_states
        self.nb_gaussians = nb_gaussians
        self.nb_obs_vars = nb_obs_vars
        self.nb_hidden_vars = nb_hidden_vars
        self.trainable_params = trainable_params
        self.initial_obs_var_coef_values = initial_obs_var_coef_values
        self.indices_initial_obs_vars = indices_initial_obs_vars
        self.initial_hidden_var_coef_values = initial_hidden_var_coef_values
        self.indices_initial_hidden_vars = indices_initial_hidden_vars
        self.initial_Gaussian_stds = initial_Gaussian_stds
        self.initial_biases = initial_biases
        self.trainable_initial_params = trainable_initial_params
        
        initial_sequence_phase_1, initial_sequence_phase_2, recurrent_sequence_phase_1, recurrent_sequence_phase_2, final_sequence_phase_1 = get_sequences(hidden_var_coef_values,
                              indices_hidden_vars,
                              nb_gaussians,
                              nb_hidden_vars,
                              nb_states,
                              initial_hidden_var_coef_values,
                              indices_initial_hidden_vars)
        
        self.initial_sequence_phase_1 = initial_sequence_phase_1
        self.initial_sequence_phase_2 = initial_sequence_phase_2
        self.recurrent_sequence_phase_1 = recurrent_sequence_phase_1
        self.recurrent_sequence_phase_2 = recurrent_sequence_phase_2
        self.final_sequence_phase_1 = final_sequence_phase_1
        
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        #print(input_shape)
        # we add constraints to the variables so it never equal 
        
        dtype = self.dtype
        nb_states = self.nb_states
        
        self.obs_var_coef_values = self.add_weight(name="obs_var_coefs",
                                                   shape=self.obs_var_coef_values.shape,
                                                   dtype=dtype,
                                                   trainable=self.trainable_params["obs"],
                                                   initializer=tf.constant_initializer(self.obs_var_coef_values),
                                                   constraint=lambda w: 
                                                       tf.where(tf.equal(w, 0), tf.cast(tf.random.uniform(shape=w.shape, minval=0, maxval=2, dtype=tf.int32) * 2 - 1,
                                                        dtype=dtype) * tf.random.uniform(tf.shape(w), minval=minval, maxval=10 * minval, dtype=dtype), w))
        
        #self.obs_var_coef_values = tf.Variable(self.obs_var_coef_values, trainable = self.trainable_params['obs'],  dtype = dtype, name = 'obs_var_coefs', constraint=lambda w: tf.where(tf.equal(w, 0), tf.cast(tf.random.uniform(shape = w.shape, minval=0, maxval=2, dtype=tf.int32)*2-1, dtype=dtype)*tf.random.uniform(tf.shape(w), minval=minval, maxval=10*minval, dtype = dtype), w))
        self.hidden_var_coef_values = tf.Variable(self.hidden_var_coef_values, trainable = self.trainable_params['hidden'], dtype = dtype, name = 'hidden_var_coefs', constraint=lambda w: tf.where(tf.equal(w, 0), tf.cast(tf.random.uniform(shape = w.shape, minval=0, maxval=2, dtype=tf.int32)*2-1, dtype=dtype)*tf.random.uniform(tf.shape(w), minval=minval, maxval=10*minval, dtype = dtype), w))
        self.Gaussian_stds = tf.Variable(self.Gaussian_stds, dtype = dtype, trainable = self.trainable_params['stds'], name = 'stds', constraint=lambda w: tf.clip_by_value(w, minval, np.inf))
        self.biases = tf.Variable(self.biases, dtype = dtype, name = 'biases', trainable = self.trainable_params['biases'])
        
        # same for the initial Gaussians:
        self.initial_obs_var_coef_values = tf.Variable(self.initial_obs_var_coef_values, trainable = self.trainable_initial_params['obs'],  dtype = dtype, name = 'initial_obs_var_coefs', constraint=lambda w: tf.where(tf.equal(w, 0), tf.cast(tf.random.uniform(shape = w.shape, minval=0, maxval=2, dtype=tf.int32)*2-1, dtype=dtype)*tf.random.uniform(tf.shape(w), minval=minval, maxval=10*minval, dtype = dtype), w))
        self.initial_hidden_var_coef_values = tf.Variable(self.initial_hidden_var_coef_values, trainable = self.trainable_initial_params['hidden'], dtype = dtype, name = 'initial_hidden_var_coefs', constraint=lambda w: tf.where(tf.equal(w, 0), tf.cast(tf.random.uniform(shape = w.shape, minval=0, maxval=2, dtype=tf.int32)*2-1, dtype=dtype)*tf.random.uniform(tf.shape(w), minval=minval, maxval=10*minval, dtype = dtype), w))
        self.initial_Gaussian_stds = tf.Variable(self.initial_Gaussian_stds, dtype = dtype, trainable = self.trainable_initial_params['stds'], name = 'initial_stds', constraint=lambda w: tf.clip_by_value(w, minval, np.inf))
        self.initial_biases = tf.Variable(self.initial_biases, dtype = dtype, name = 'initial_biases', trainable =  self.trainable_initial_params['biases'])
        
        self.initial_fractions = tf.Variable((np.random.rand(1, nb_states)+0.5)/nb_states, dtype = dtype, name = 'Fractions', trainable = True)
        
    def call(self, inputs):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        '''

        nb_tracks = inputs.shape[2]
        sparse_obs_var_coefs = tf.SparseTensor(indices = self.indices_obs_vars,
                                               values = self.obs_var_coef_values,
                                               dense_shape = (self.nb_gaussians, 1, self.nb_states, self.nb_obs_vars))
        
        obs_var_coefs = tf.sparse.to_dense(sparse_obs_var_coefs)
        
        sparse_hidden_var_coefs = tf.SparseTensor(indices = self.indices_hidden_vars,
                                               values = self.hidden_var_coef_values,
                                               dense_shape = (self.nb_gaussians, 1, self.nb_states, 2*self.nb_hidden_vars))
        
        hidden_var_coefs = tf.sparse.to_dense(sparse_hidden_var_coefs)
        
        #norm_vals = tf.math.reduce_max(tf.math.abs(hidden_var_coefs), -1, keepdims = True)
        
        biases = - self.biases
        Gaussian_stds = self.Gaussian_stds
                

        #LF = - tf.math.reduce_sum(tf.math.log(Gaussian_stds), 0)[:,:,0] + tf.math.log(tf.math.abs(tf.linalg.det(Mat)))#+ tf.math.reduce_sum(tf.math.log(tf.abs(self.hidden_var_coef_values)))
        #Log_factors = - tf.math.reduce_sum(tf.math.log(Gaussian_stds), 0)[:,:,0] #+ tf.math.log(tf.math.abs(tf.linalg.det(Mat)))#+ tf.math.reduce_sum(tf.math.log(tf.abs(self.hidden_var_coef_values)))
        #print('Simple log factor', Log_factors)
        
        #print('hidden_var_coefs', hidden_var_coefs)
        # change of variables to deal with gaussians of variance 1
        hidden_var_coefs = hidden_var_coefs/Gaussian_stds
        obs_var_coefs = obs_var_coefs/Gaussian_stds
        biases = biases/Gaussian_stds[:,:,:,0]
        
        obs_var_coefs = tf.repeat(obs_var_coefs, nb_tracks, 1)
        hidden_var_coefs = tf.repeat(hidden_var_coefs, nb_tracks, 1)
        biases = tf.repeat(biases, nb_tracks, 1)
        
        current_hidden_var_coefs = hidden_var_coefs[:,:,:,:self.nb_hidden_vars]
        next_hidden_var_coefs = hidden_var_coefs[:,:,:,self.nb_hidden_vars:]
        
        reccurent_obs_var_coefs = tf.identity(obs_var_coefs)
        reccurent_hidden_var_coefs = tf.identity(current_hidden_var_coefs)
        reccurent_next_hidden_var_coefs = tf.identity(next_hidden_var_coefs)
        reccurent_biases = tf.identity(biases)
                
        sparse_initial_obs_var_coefs = tf.SparseTensor(indices = self.indices_initial_obs_vars,
                                               values = self.initial_obs_var_coef_values,
                                               dense_shape = (self.nb_hidden_vars, 1, self.nb_states, self.nb_obs_vars))
        
        initial_obs_var_coefs = tf.sparse.to_dense(sparse_initial_obs_var_coefs)
        
        sparse_initial_hidden_var_coefs = tf.SparseTensor(indices = self.indices_initial_hidden_vars,
                                               values = self.initial_hidden_var_coef_values,
                                               dense_shape = (self.nb_hidden_vars, 1, self.nb_states, 2*self.nb_hidden_vars))
        
        initial_hidden_var_coefs = tf.sparse.to_dense(sparse_initial_hidden_var_coefs)
        
        initial_Gaussian_stds = self.initial_Gaussian_stds
        initial_biases = - self.initial_biases
        
        #print('hidden_var_coefs', hidden_var_coefs)
        # change of variables to deal with gaussians of variance 1
        initial_hidden_var_coefs = initial_hidden_var_coefs/initial_Gaussian_stds
        initial_obs_var_coefs = initial_obs_var_coefs/initial_Gaussian_stds
        initial_biases = initial_biases/initial_Gaussian_stds[:,:,:,0]
        #self.nb_gaussians += initial_hidden_var_coefs.shape[0]
        
        initial_obs_var_coefs = tf.repeat(initial_obs_var_coefs, nb_tracks, 1)
        initial_hidden_var_coefs = tf.repeat(initial_hidden_var_coefs, nb_tracks, 1)
        initial_biases = tf.repeat(initial_biases, nb_tracks, 1)
                
        current_initial_hidden_var_coefs = initial_hidden_var_coefs[:,:,:,:self.nb_hidden_vars]
        next_initial_hidden_var_coefs = initial_hidden_var_coefs[:,:,:,self.nb_hidden_vars:]*0 # these coefs must equal 0 as the initial gaussians must only depend on the fist set of hidden states
        
        #print(obs_var_coefs, inputs, biases)
        biases += tf.reduce_sum(obs_var_coefs * inputs[0], -1)
        initial_biases += tf.reduce_sum(initial_obs_var_coefs * inputs[0], -1)
        
        #print('inputs[0]', inputs[0])
        #hidden_var_coefs = list(hidden_var_coefs)
        
        #current_hidden_var_coefs = tf.unstack(current_initial_hidden_var_coefs) + tf.unstack(current_hidden_var_coefs)
        #next_hidden_var_coefs =  tf.unstack(next_initial_hidden_var_coefs) + tf.unstack(next_hidden_var_coefs)
        #biases = tf.unstack(initial_biases) + tf.unstack(biases)
        
        current_hidden_var_coefs = tf.concat((current_initial_hidden_var_coefs, current_hidden_var_coefs), axis = 0)
        next_hidden_var_coefs =  tf.concat((next_initial_hidden_var_coefs, next_hidden_var_coefs), axis = 0)
        biases = tf.concat((initial_biases, biases), axis = 0)
                
        sequence_phase_1 = self.initial_sequence_phase_1
        sequence_phase_2 = self.initial_sequence_phase_2

        Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                             next_hidden_var_coefs,
                                                             biases,
                                                             sequence_phase_1,
                                                             sequence_phase_2,
                                                             dtype = self.dtype)
        
        initial_Log_factors, Log_factors = self.compute_scaling_factors(initial_obs_var_coefs, initial_hidden_var_coefs, obs_var_coefs, hidden_var_coefs)
        
        #print('LP',LP)
        #print('LC',LC)
        #print('Log_factors',Log_factors)
        
        LP = (LC + initial_Log_factors)*1 + tf.math.softmax(self.initial_fractions)
        #print('LP',LP)
        #initial_states = [Next_coefs, Next_biases, LP, Log_factors, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases]
        initial_states = [Next_coefs, Next_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases]
        
        return inputs, initial_states
    
    def compute_scaling_factors(self, initial_obs_var_coefs, initial_hidden_var_coefs, obs_var_coefs, hidden_var_coefs):
        
        nb_hidden_vars = self.nb_hidden_vars
        nb_obs_vars = self.nb_obs_vars
        dtype = self.dtype
        #total_nb_vars = N*(nb_hidden_vars + nb_obs_vars) + nb_hidden_vars
    
        init_obs_vars = initial_obs_var_coefs[:,0]
        init_hidden_vars = initial_hidden_var_coefs[:,0]
        obs_vars = obs_var_coefs[:,0]
        hidden_vars = hidden_var_coefs[:,0]
        init_obs_zeros = tf.zeros(init_obs_vars.shape, dtype = dtype)
        init_hidden_zeros = tf.zeros((init_hidden_vars.shape[0], init_hidden_vars.shape[1], nb_hidden_vars), dtype = dtype)
    
        N = 1
        A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*(N-1)
        A0 = tf.concat(tuple(A0), axis = 2)
        A = [A0]
    
        obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
        hidden_zeros = tf.zeros((hidden_vars.shape[0], hidden_vars.shape[1], nb_hidden_vars), dtype = dtype)
        for i in range(0, N):
            var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1) 
            Ai = tf.concat(tuple(var_list), axis = 2)
            A.append(Ai)
        
        A = tf.concat(A, axis = 0)
        new_A = tf.transpose(A, [1,0,2])
        #new_A[0, -1, -1] = 1
        
        initial_Log_factors = tf.math.log(tf.math.abs(tf.linalg.det(new_A)))
        #print(det_1)
        
        N = 2
        A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*(N-1)
        A0 = tf.concat(tuple(A0), axis = 2)
        A = [A0]
        
        obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
        hidden_zeros = tf.zeros((hidden_vars.shape[0], hidden_vars.shape[1], nb_hidden_vars), dtype = dtype)
        for i in range(0, N):
            var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1) 
            Ai = tf.concat(tuple(var_list), axis = 2)
            A.append(Ai)
        
        A = tf.concat(A, axis = 0)
        new_A = tf.transpose(A, [1,0,2])
        Log_factors = tf.math.log(tf.math.abs(tf.linalg.det(new_A))) - initial_Log_factors
        
        return initial_Log_factors, Log_factors

class NonNegative(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w * tf.keras.ops.cast(tf.keras.ops.greater_equal(w, 0.), dtype=w.dtype)

class Initial_layer_constraints(tf.keras.layers.Layer):
    def __init__(
        self,
        nb_states,
        nb_gaussians,
        nb_obs_vars,
        nb_hidden_vars,
        params,
        initial_params,
        constraint_function,
        **kwargs):
        
        self.nb_states = nb_states
        self.nb_gaussians = nb_gaussians
        self.nb_obs_vars = nb_obs_vars
        self.nb_hidden_vars = nb_hidden_vars
        self.params = params
        self.initial_params = initial_params
        self.constraint_function = constraint_function
        
        super().__init__(**kwargs)
        
        dtype = self.dtype
        initial_sequence_phase_1, initial_sequence_phase_2, recurrent_sequence_phase_1, recurrent_sequence_phase_2, final_sequence_phase_1 = get_sequences(params, initial_params, constraint_function, nb_gaussians, nb_hidden_vars, nb_states, dtype)
        
        self.initial_sequence_phase_1 = initial_sequence_phase_1
        self.initial_sequence_phase_2 = initial_sequence_phase_2
        self.recurrent_sequence_phase_1 = recurrent_sequence_phase_1
        self.recurrent_sequence_phase_2 = recurrent_sequence_phase_2
        self.final_sequence_phase_1 = final_sequence_phase_1
        
    def build(self, input_shape):
        #print(input_shape)
        # we add constraints to the variables so it never equal 
        
        dtype = self.dtype
        nb_states = self.nb_states
        '''
        param_vars = tf.Variable(params,  dtype = dtype, name = 'recurrence_variables', constraint=lambda w: tf.where(tf.greater_equal(w, -1), w, 0.0000001))
        initial_param_vars = tf.Variable(initial_params,  dtype = dtype, name = 'initial_variables', constraint=lambda w: tf.where(tf.greater_equal(w, 0), w, 0.0000001))
        initial_fractions = tf.Variable((np.random.rand(1, nb_states)+0.5)/nb_states, dtype = dtype, name = 'Fractions', trainable = True)
        '''
        
        self.param_vars = self.add_weight(name="param_vars",
                                                   shape=self.params.shape,
                                                   dtype=dtype,
                                                   trainable=True,
                                                   initializer=tf.constant_initializer(self.params))
        
        self.initial_param_vars = self.add_weight(name="initial_param_vars",
                                                   shape=self.initial_params.shape,
                                                   dtype=dtype,
                                                   trainable=True,
                                                   initializer=tf.constant_initializer(self.initial_params))
        
        self.initial_fractions = self.add_weight(name="initial_fractions",
                                                   shape=(1, nb_states),
                                                   dtype=dtype,
                                                   trainable=True,
                                                   initializer=tf.constant_initializer((np.random.rand(1, nb_states)+0.5)/nb_states))
        
        #self.param_vars = tf.Variable(self.params,  dtype = dtype, name = 'recurrence_variables', constraint=lambda w: tf.where(tf.greater_equal(w, -1), w, 0.0000001), trainable = True)
        #self.initial_param_vars = tf.Variable(self.initial_params,  dtype = dtype, name = 'initial_variables', constraint=lambda w: tf.where(tf.greater_equal(w, 0), w, 0.0000001), trainable = True)
        #self.initial_fractions = tf.Variable((np.random.rand(1, nb_states)+0.5)/nb_states, dtype = dtype, name = 'Fractions', trainable = True)
        
    def call(self, inputs):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        '''
        
        nb_tracks = inputs.shape[2]
        nb_hidden_vars = self.nb_hidden_vars
        dtype = self.dtype
        constraint_function = self.constraint_function
        nb_states = self.nb_states
        
        param_vars = self.param_vars
        initial_param_vars = self.initial_param_vars
        #param_vars = tf.clip_by_value(self.param_vars, clip_value_min= 0.00000001, clip_value_max=np.inf)
        #initial_param_vars = tf.clip_by_value(self.initial_param_vars, clip_value_min= 0.00000001, clip_value_max=np.inf)
        
        hidden_var_coefs, obs_var_coefs, Gaussian_stds, biases, initial_hidden_var_coefs, initial_obs_var_coefs, initial_Gaussian_stds, initial_biases = constraint_function(param_vars, initial_param_vars, dtype)
        
        hidden_var_coefs = hidden_var_coefs/Gaussian_stds
        obs_var_coefs = obs_var_coefs/Gaussian_stds[:,:,:,:,None]
        biases = biases/Gaussian_stds
        
        #obs_var_coefs = tf.repeat(obs_var_coefs, nb_tracks, 1)
        #hidden_var_coefs = tf.repeat(hidden_var_coefs, nb_tracks, 1)
        #biases = tf.repeat(biases, nb_tracks, 1)
        
        current_hidden_var_coefs = hidden_var_coefs[:,:,:,:nb_hidden_vars]
        next_hidden_var_coefs = hidden_var_coefs[:,:,:,nb_hidden_vars:]
        
        reccurent_obs_var_coefs = tf.identity(obs_var_coefs)
        reccurent_hidden_var_coefs = tf.identity(current_hidden_var_coefs)
        reccurent_next_hidden_var_coefs = tf.identity(next_hidden_var_coefs)
        reccurent_biases = tf.identity(biases)
        
        #print('hidden_var_coefs', hidden_var_coefs)
        # change of variables to deal with gaussians of variance 1
        initial_hidden_var_coefs = initial_hidden_var_coefs/initial_Gaussian_stds
        initial_obs_var_coefs = initial_obs_var_coefs/initial_Gaussian_stds[:,:,:,:,None]
        initial_biases = initial_biases/initial_Gaussian_stds
        #self.nb_gaussians += initial_hidden_var_coefs.shape[0]
        
        #initial_obs_var_coefs = tf.repeat(initial_obs_var_coefs, nb_tracks, 1)
        #initial_hidden_var_coefs = tf.repeat(initial_hidden_var_coefs, nb_tracks, 1)
        #initial_biases = tf.repeat(initial_biases, nb_tracks, 1)
        
        current_initial_hidden_var_coefs = initial_hidden_var_coefs[:,:,:,:nb_hidden_vars]
        next_initial_hidden_var_coefs = tf.zeros((nb_hidden_vars, 1, nb_states, nb_hidden_vars), dtype = dtype)  # these coefs must equal 0 as the initial gaussians must only depend on the fist set of hidden states
        
        #print(obs_var_coefs, inputs, biases)
        biases += tf.reduce_sum(obs_var_coefs * inputs[0], -2)
        initial_biases += tf.reduce_sum(initial_obs_var_coefs * inputs[0], -2)
        
        #print('inputs[0]', inputs[0])
        #hidden_var_coefs = list(hidden_var_coefs)
        
        #current_hidden_var_coefs = tf.unstack(current_initial_hidden_var_coefs) + tf.unstack(current_hidden_var_coefs)
        #next_hidden_var_coefs =  tf.unstack(next_initial_hidden_var_coefs) + tf.unstack(next_hidden_var_coefs)
        #biases = tf.unstack(initial_biases) + tf.unstack(biases)
        
        current_hidden_var_coefs = tf.concat((current_initial_hidden_var_coefs, current_hidden_var_coefs), axis = 0)
        next_hidden_var_coefs =  tf.concat((next_initial_hidden_var_coefs, next_hidden_var_coefs), axis = 0)
        biases = tf.concat((initial_biases, biases), axis = 0)
        
        sequence_phase_1 = self.initial_sequence_phase_1
        sequence_phase_2 = self.initial_sequence_phase_2
        
        Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                             next_hidden_var_coefs,
                                                             biases,
                                                             sequence_phase_1,
                                                             sequence_phase_2,
                                                             dtype = dtype)
        
        initial_Log_factors, Log_factors = self.compute_scaling_factors(initial_obs_var_coefs, initial_hidden_var_coefs, obs_var_coefs, hidden_var_coefs)
        
        #print('LP',LP)
        #print('LC',LC)
        #print('Log_factors',Log_factors)
        
        LP = (LC + initial_Log_factors)*1 + tf.math.softmax(self.initial_fractions)
        #print('LP',LP)
        #initial_states = [Next_coefs, Next_biases, LP, Log_factors, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases]
        initial_states = [Next_coefs, Next_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases]
        
        return inputs, initial_states
    '''
    def compute_scaling_factors(self, initial_obs_var_coefs, initial_hidden_var_coefs, obs_var_coefs, hidden_var_coefs):
        
        nb_hidden_vars = self.nb_hidden_vars
        nb_obs_vars = self.nb_obs_vars
        dtype = self.dtype
        #total_nb_vars = N*(nb_hidden_vars + nb_obs_vars) + nb_hidden_vars
        Log_factors = []
        initial_Log_factors = []
        
        for state in range(self.nb_states):
            init_obs_vars = initial_obs_var_coefs[:,0,state]
            init_hidden_vars = initial_hidden_var_coefs[:,0,state]
            obs_vars = obs_var_coefs[:,0,state]
            hidden_vars = hidden_var_coefs[:,0,state]
            init_obs_zeros = tf.zeros(init_obs_vars.shape, dtype = dtype)
            init_hidden_zeros = tf.zeros((init_hidden_vars.shape[0], nb_hidden_vars), dtype = dtype)

            N = 1
            A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*(N-1)
            A0 = tf.concat(tuple(A0), axis = 1)
            A = [A0]
            for i in range(0, N):
                obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
                hidden_zeros = tf.zeros((hidden_vars.shape[0], nb_hidden_vars), dtype = dtype)
                
                var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1) 
                Ai = tf.concat(tuple(var_list), axis = 1)
                
                A.append(Ai)
            
            A = tf.concat(A, axis = 0)
            #print('A', A)

            det_1 =  tf.math.log(tf.math.abs(tf.linalg.det(A)))
            #print(det_1)
            initial_Log_factors.append(det_1)
            
            N = 2
            A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*(N-1)
            A0 = tf.concat(tuple(A0), axis = 1)
            A = [A0]
            for i in range(0, N):
                obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
                hidden_zeros = tf.zeros((hidden_vars.shape[0], nb_hidden_vars), dtype = dtype)
                
                var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1) 
                Ai = tf.concat(tuple(var_list), axis = 1)
                A.append(Ai)
            
            A = tf.concat(A, axis = 0)
            
            det_2 = tf.math.log(tf.math.abs(tf.linalg.det(A)))
            Log_factors.append(det_2 - det_1)
            
        Log_factors = tf.stack([Log_factors])
        initial_Log_factors = tf.stack([initial_Log_factors])
        
        return initial_Log_factors, Log_factors
    '''
    def compute_scaling_factors(self, initial_obs_var_coefs, initial_hidden_var_coefs, obs_var_coefs, hidden_var_coefs):
        
        nb_hidden_vars = self.nb_hidden_vars
        nb_obs_vars = self.nb_obs_vars
        dtype = self.dtype
        #total_nb_vars = N*(nb_hidden_vars + nb_obs_vars) + nb_hidden_vars
        
        init_obs_vars = initial_obs_var_coefs[:,0, :,:,0]
        init_hidden_vars = initial_hidden_var_coefs[:,0]
        obs_vars = obs_var_coefs[:,0, :,:,0]
        hidden_vars = hidden_var_coefs[:,0]
        init_obs_zeros = tf.zeros(init_obs_vars.shape, dtype = dtype)
        init_hidden_zeros = tf.zeros((init_hidden_vars.shape[0], init_hidden_vars.shape[1], nb_hidden_vars), dtype = dtype)
    
        N = 1
        A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*N
        A0 = tf.concat(tuple(A0), axis = 2)
        A = [A0]
    
        obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
        hidden_zeros = tf.zeros((hidden_vars.shape[0], hidden_vars.shape[1], nb_hidden_vars), dtype = dtype)
        for i in range(0, N):
            var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1) 
            Ai = tf.concat(tuple(var_list), axis = 2)
            A.append(Ai)
        
        A = tf.concat(A, axis = 0)
        new_A = tf.transpose(A, [1,0,2])
        #new_A[0, -1, -1] = 1
        
        initial_Log_factors = tf.math.log(tf.math.abs(tf.linalg.det(new_A)))
        #print(det_1)
        
        N = 2
        A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*N
        A0 = tf.concat(tuple(A0), axis = 2)
        A = [A0]
        
        obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
        hidden_zeros = tf.zeros((hidden_vars.shape[0], hidden_vars.shape[1], nb_hidden_vars), dtype = dtype)
        for i in range(0, N):
            var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1) 
            Ai = tf.concat(tuple(var_list), axis = 2)
            A.append(Ai)
        
        A = tf.concat(A, axis = 0)
        new_A = tf.transpose(A, [1,0,2])
        Log_factors = tf.math.log(tf.math.abs(tf.linalg.det(new_A))) - initial_Log_factors
        
        return initial_Log_factors, Log_factors


@tf.function(jit_compile=True, reduce_retracing=True)
def RNN_cell(input_i, Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, sequence_phase_1, sequence_phase_2):
    
    current_hidden_var_coefs = tf.concat((Prev_coefs, tf.identity(reccurent_hidden_var_coefs)), axis = 0)
    zero_tensor = tf.constant(0, dtype = dtype, shape = Prev_coefs.shape)
    next_hidden_var_coefs = tf.concat((zero_tensor, tf.identity(reccurent_next_hidden_var_coefs)), axis = 0)
    #biases = list(Prev_biases) + list(tf.identity(reccurent_biases))
    
    current_biases = tf.identity(reccurent_biases)
    current_biases += tf.reduce_sum(reccurent_obs_var_coefs * input_i, -2)
    biases = tf.concat((Prev_biases, current_biases), axis =  0)
    
    #nb_gaussians = len(biases)
    
    #print('current_hidden_var_coefs', current_hidden_var_coefs)
    #print('next_hidden_var_coefs', next_hidden_var_coefs)
    #print('biases', biases)
    
    Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                         next_hidden_var_coefs,
                                                         biases,
                                                         sequence_phase_1,
                                                         sequence_phase_2,
                                                         #dependent_variables, # bool array that memorized if coefficients are non-nul 
                                                         #nb_hidden_vars, # number of hidden variables to integrate during this step
                                                         #nb_gaussians, # number of gaussians, must equal hidden_vars_coefs.shape[0]
                                                         dtype = dtype)
    
    LP += LC + Log_factors
    
    return Next_coefs, Next_biases, LP

class Custom_RNN_layer(tf.keras.layers.Layer):
    
    def __init__(self, sequence_phase_1, sequence_phase_2, **kwargs):
        self.sequence_phase_1 = sequence_phase_1
        self.sequence_phase_2 = sequence_phase_2
        
        #self.parent = parent
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        self.built = True
        
    @tf.function(jit_compile=False)
    def call(self, inputs, Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        
        sequence_phase_1 = self.sequence_phase_1
        sequence_phase_2 = self.sequence_phase_2
        
        for input_i in inputs:
            #Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases = states
            Prev_coefs, Prev_biases, LP = RNN_cell(input_i, Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, sequence_phase_1, sequence_phase_2)
            Prev_coefs
            #print(LP)
            #output = [Next_coefs, Next_biases, LP, Log_factors, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases]
            #print('Next_coefs', tf.stack(Next_coefs))
            
            
        #states = [Next_coefs, Next_biases, LP]
        
        #print('inputs', states)
        #print('outputs', output)
        
        return Prev_coefs, Prev_biases, LP #states
  

def check_error(reccurent_obs_var_coefs, input_i):
    return tf.reduce_sum(reccurent_obs_var_coefs * input_i, -1)


class Custom_RNNCell(tf.keras.layers.Layer):
   
    def __init__(self, state_size, sequence_phase_1, sequence_phase_2, **kwargs):
        self.state_size = state_size # [nb_dims, 1,1]
        self.sequence_phase_1 = sequence_phase_1
        self.sequence_phase_2 = sequence_phase_2
        #self.parent = parent
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        self.built = True
    
    @tf.function(jit_compile=True)
    def call(self, input_i, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        
        sequence_phase_1 = self.sequence_phase_1
        sequence_phase_2 = self.sequence_phase_2
        dtype = self.dtype
        
        Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases = states
        
        current_hidden_var_coefs = tf.concat((Prev_coefs, tf.identity(reccurent_hidden_var_coefs)), axis = 0)
        zero_tensor = tf.constant(0, dtype = dtype, shape = Prev_coefs.shape)
        next_hidden_var_coefs = tf.concat((zero_tensor, tf.identity(reccurent_next_hidden_var_coefs)), axis = 0)
        #biases = list(Prev_biases) + list(tf.identity(reccurent_biases))
        
        current_biases = tf.identity(reccurent_biases)
        #current_biases += tf.reduce_sum(reccurent_obs_var_coefs * input_i, -1)
        current_biases += check_error(reccurent_obs_var_coefs, input_i)
        
        biases = tf.concat((Prev_biases, current_biases), axis =  0)
        #nb_gaussians = len(biases)
        
        Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                             next_hidden_var_coefs,
                                                             biases,
                                                             sequence_phase_1,
                                                             sequence_phase_2,
                                                             #dependent_variables, # bool array that memorized if coefficients are non-nul 
                                                             #nb_hidden_vars, # number of hidden variables to integrate during this step
                                                             #nb_gaussians, # number of gaussians, must equal hidden_vars_coefs.shape[0]
                                                             dtype = dtype)
        
        print(LC)

        LP += LC + Log_factors
        output = [tf.stack(Next_coefs), tf.stack(Next_biases), LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases]
        
        return tf.transpose(output[2], perm = [1,0]), output


class Final_layer(tf.keras.layers.Layer):
    def __init__(self, sequence_phase_1, **kwargs):
        self.sequence_phase_1 = sequence_phase_1
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
   
    @tf.function(jit_compile=True)
    def call(self, states):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        '''
        
        Prev_coefs, Prev_biases, LP = states
        """
        def final_function(Prev_coefs, Prev_biases, LP):
            
            current_hidden_var_coefs = Prev_coefs
            zero_tensor = tf.constant(0, dtype = dtype, shape =  Prev_coefs.shape)
            next_hidden_var_coefs = zero_tensor
            
            biases = Prev_biases
            '''
            sequence_phase_1 = self.sequence_phase_1
            sequence_phase_2 = [[], []]
            '''
            Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                                 next_hidden_var_coefs,
                                                                 biases,
                                                                 self.sequence_phase_1,
                                                                 [[], []],
                                                                 #dependent_variables, # bool array that memorized if coefficients are non-nul 
                                                                 #nb_hidden_vars, # number of hidden variables to integrate during this step
                                                                 #nb_gaussians, # number of gaussians, must equal hidden_vars_coefs.shape[0]
                                                                 dtype = self.dtype)
            
            LP += LC
            return LP
        
        def no_final_function(Prev_coefs, Prev_biases, LP):
            return LP
        
        LP = tf.cond(tf.less(0.5, Prev_coefs.shape[0]), 
                         lambda: final_function(Prev_coefs, Prev_biases, LP), 
                         lambda: no_final_function(Prev_coefs, Prev_biases, LP))
        """
        #print('LP', LP)
        if Prev_coefs.shape[0]>0:
            
            current_hidden_var_coefs = Prev_coefs
            zero_tensor = tf.constant(0, dtype = dtype, shape =  Prev_coefs.shape)
            next_hidden_var_coefs = zero_tensor
            
            biases = Prev_biases
            '''
            sequence_phase_1 = self.sequence_phase_1
            sequence_phase_2 = [[], []]
            '''
            Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                                 next_hidden_var_coefs,
                                                                 biases,
                                                                 self.sequence_phase_1,
                                                                 [[], []],
                                                                 #dependent_variables, # bool array that memorized if coefficients are non-nul 
                                                                 #nb_hidden_vars, # number of hidden variables to integrate during this step
                                                                 #nb_gaussians, # number of gaussians, must equal hidden_vars_coefs.shape[0]
                                                                 dtype = self.dtype)
            
            LP += LC
        #"""
        output = LP        
        return output

class Fractions_Layer(tf.keras.layers.Layer):
    def __init__(self, nb_states, **kwargs):
        self.nb_states = nb_states
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        if not self.built:
            self.Fractions = tf.Variable(initial_value=np.ones(self.nb_states)[None]/self.nb_states, trainable=True, name="Fractions", dtype = dtype)
        self.built = True

    def call(self, x):
        F_P =  tf.math.log(tf.math.softmax(self.Fractions))
        return x + F_P

class transpose_layer(tf.keras.layers.Layer):
    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
   
    def call(self, x, perm):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        '''
        return tf.transpose(x, perm = perm)

def simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2):
    '''
    simplification of RNN_gaussian_product for the function get_sequences
    '''
    
    current_coefs1 = current_hidden_var_coefs_1 / C1
    current_coefs2 =  current_hidden_var_coefs_2 / C2
    next_coefs1 = next_hidden_var_coefs_1 / C1
    next_coefs2 = next_hidden_var_coefs_2 / C2
    
    var1 = 1./C1**2
    var2 = 1./C2**2
    
    var3 = var1 + var2
    std3 = var3**0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3
    
    var4 = var1 * var2 / var3
    std4 = var4**0.5
    current_coefs4 = (current_coefs1*var2 + current_coefs2*var1)/(var3*std4)
    next_coefs4 = (next_coefs1*var2 + next_coefs2*var1)/(var3*std4)
    
    return current_coefs3, current_coefs4, next_coefs3, next_coefs4

def get_sequences(all_params, all_initial_params, constraint_function, nb_gaussians, nb_hidden_vars, nb_states, dtype):
    '''
    Function that get the sequences of indexes to eliminate the coefficents and perform the recursive integration process
    
    The integration process for one time step is composed of 2 phases, phase 1: integration over the current hidden variables, phase 2 rearangement of the matrix of the remaining next hidden variables to minimize the number of gaussians that are dependent on the next hidden variables
    
    In the process, we need to get 2 sequences (for phases 1 and 2) that specify the operations for the initial step, 2 additional sequences (phases 1 and 2) for the recurrence step and 1 final sequence (phase 1) for the last step.
    Each sequence must inform about the coefficient to integrate, the gaussian IDs and the function to use.
    
    The function then needs to compute and return 6 lists : [initial_functions_phase_1, np.array(initial_sequence_phase_1, dtype = 'int32')], [initial_functions_phase_2, np.array(initial_sequence_phase_2, dtype = 'int32')], [recurrent_functions_phase_1, np.array(recurrent_sequence_phase_1, dtype = 'int32')], [recurrent_functions_phase_2, np.array(recurrent_sequence_phase_2, dtype = 'int32')], [final_functions_phase_1, np.array(final_sequence_phase_1, dtype = 'int32')]
    [initial_functions_phase_1, np.array(initial_sequence_phase_1, dtype = 'int32')] : sequence to apply for the phase 1 of the inital step
    [initial_functions_phase_2, np.array(initial_sequence_phase_2, dtype = 'int32')] : sequence to apply for the phase 2 of the inital step
    [recurrent_functions_phase_1, np.array(recurrent_sequence_phase_1, dtype = 'int32')] : sequence to apply for the phase 1 of the recurrence step
    [recurrent_functions_phase_2, np.array(recurrent_sequence_phase_2, dtype = 'int32')] : sequence to apply for the phase 2 of the recurrence step
    [final_functions_phase_1, np.array(final_sequence_phase_1, dtype = 'int32')] : sequence to apply for the phase 1 of the recurrence step, the last step has no phase 2
    '''
    
    hidden_var_coefs, _, _, _, initial_hidden_var_coefs, _, _, _ = constraint_function(all_params, all_initial_params, dtype)
    
    recurrent_current_hidden_var_coefs = np.copy(hidden_var_coefs[:,0,0,:nb_hidden_vars])
    recurrent_next_hidden_var_coefs = np.copy(hidden_var_coefs[:,0,0,nb_hidden_vars:])
    
    current_hidden_var_coefs = hidden_var_coefs[:,0,0,:nb_hidden_vars]
    next_hidden_var_coefs = hidden_var_coefs[:,0,0,nb_hidden_vars:]
        
    current_initial_hidden_var_coefs = initial_hidden_var_coefs[:,0,0,:nb_hidden_vars]
    next_initial_hidden_var_coefs = tf.zeros((nb_hidden_vars, nb_hidden_vars), dtype = dtype) # these coefs must equal 0 as the initial gaussians must only depend on the fist set of hidden states
    
    current_hidden_var_coefs = np.concatenate((current_initial_hidden_var_coefs, current_hidden_var_coefs), axis = 0)
    next_hidden_var_coefs = np.concatenate((next_initial_hidden_var_coefs, next_hidden_var_coefs), axis = 0)
    
    current_nb_gaussians = len(current_hidden_var_coefs)
    
    '''
    Initial step:
    '''
    
    initial_sequence_phase_1 = [] # list of lists containing the sequence of coef_index and gaussian IDs to
    initial_functions_phase_1 = []
    
    #print('LC1',LC)
    for coef_index in range(nb_hidden_vars):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            initial_sequence_phase_1.append([coef_index, ID_1, ID_2])
            initial_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = current_hidden_var_coefs[ID_1]
            current_hidden_var_coefs_2 = current_hidden_var_coefs[ID_2]
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]
            
            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2)
            
            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4
            
        if len(non_zero_gaussian_IDs)>1:
            initial_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs)==1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            initial_sequence_phase_1.append([coef_index, ID_1, ID_2])
            initial_functions_phase_1.append(no_RNN_function_phase_1)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass
        
        if len(non_zero_gaussian_IDs)>=1:
            current_hidden_var_coefs = np.delete(current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1
    
    initial_sequence_phase_2 = []
    initial_functions_phase_2 = []
    
    saved_Gaussians = np.zeros((nb_hidden_vars, nb_hidden_vars))
    # contrary to the integration step, we cannot remove Gaussians. Instead, we will save them to solve the linear problem
    for coef_index in range(nb_hidden_vars):
        
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = next_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            initial_sequence_phase_2.append([coef_index, ID_1, ID_2])
            initial_functions_phase_2.append(intermediate_RNN_function)
            
            C1 = next_hidden_var_coefs[ID_1, coef_index]
            C2 = next_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]*0
            current_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]*0
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]
            
            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, current_hidden_var_coefs_1, current_hidden_var_coefs_2)
            
            next_hidden_var_coefs[ID_1] = current_coefs3
            next_hidden_var_coefs[ID_2] = current_coefs4
            
        if len(non_zero_gaussian_IDs)>1:
            initial_functions_phase_2[-1] = final_RNN_function_phase_2
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            initial_sequence_phase_2.append([coef_index, ID_1, ID_2])
            initial_functions_phase_2.append(no_RNN_function_phase_2)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: 
            saved_Gaussians[coef_index] = next_hidden_var_coefs[ID_2]
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, ID_2, 0)
            current_nb_gaussians += -1
    
    initial_saved_Gaussians = saved_Gaussians
    
    # Recurrence step:
    
    current_hidden_var_coefs = np.concatenate((saved_Gaussians, recurrent_current_hidden_var_coefs), 0)
    next_hidden_var_coefs = np.concatenate((saved_Gaussians*0, recurrent_next_hidden_var_coefs), 0)
    
    current_nb_gaussians = len(current_hidden_var_coefs)
    
    '''
    recurrence step:
    '''
    
    recurrent_sequence_phase_1 = [] # list of lists containing the sequence of coef_index and gaussian IDs to 
    recurrent_functions_phase_1 = []

    #print('LC1',LC)
    for coef_index in range(nb_hidden_vars):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            recurrent_sequence_phase_1.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = current_hidden_var_coefs[ID_1]
            current_hidden_var_coefs_2 = current_hidden_var_coefs[ID_2]
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]

            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2)

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4

            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            recurrent_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            recurrent_sequence_phase_1.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_1.append(no_RNN_function_phase_1)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: 
            current_hidden_var_coefs = np.delete(current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1
    
    recurrent_sequence_phase_2 = []
    recurrent_functions_phase_2 = []

    saved_Gaussians = np.zeros((nb_hidden_vars, nb_hidden_vars))
    # contrary to the integration step, we cannot remove Gaussians. Instead, we will save them to solve the linear problem
    for coef_index in range(nb_hidden_vars):
        
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = next_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            recurrent_sequence_phase_2.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_2.append(intermediate_RNN_function)

            C1 = next_hidden_var_coefs[ID_1, coef_index]
            C2 = next_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]*0
            current_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]*0
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]

            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, current_hidden_var_coefs_1, current_hidden_var_coefs_2)

            next_hidden_var_coefs[ID_1] = current_coefs3
            next_hidden_var_coefs[ID_2] = current_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            recurrent_functions_phase_2[-1] = final_RNN_function_phase_2
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            recurrent_sequence_phase_2.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_2.append(no_RNN_function_phase_2)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: # we remove the last gaussian that depend on the coefficient on index coef_index (only valid if at least on gaussian has a non 0 coefficient)
            saved_Gaussians[coef_index] = next_hidden_var_coefs[ID_2]
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, ID_2, 0)
            current_nb_gaussians += -1
            
    print('Checking that the recurrent next Gaussians have the same form than the initial next gaussians:', np.all((initial_saved_Gaussians == 0) == (saved_Gaussians == 0)))
    '''
    Final step
    Contrary to the previous steps, the final step does not introduce new gaussians that depend on the next hidden variables. 
    Therefore, we only need to perform the phase 1 on the gaussians that remain from the previous step
    '''
    
    
    current_hidden_var_coefs = saved_Gaussians
    current_nb_gaussians = len(current_hidden_var_coefs)

    next_hidden_var_coefs = np.zeros(current_hidden_var_coefs.shape)
    
    final_sequence_phase_1 = [] # list of lists containing the sequence of coef_index and gaussian IDs to 
    final_functions_phase_1 = []
    
    #print('LC1',LC)
    for coef_index in range(nb_hidden_vars):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            final_sequence_phase_1.append([coef_index, ID_1, ID_2])
            final_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = current_hidden_var_coefs[ID_1]
            current_hidden_var_coefs_2 = current_hidden_var_coefs[ID_2]
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]
            
            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2)

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4

            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            recurrent_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            final_sequence_phase_1.append([coef_index, ID_1, ID_2])
            final_functions_phase_1.append(no_RNN_function_phase_1)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: 
            current_hidden_var_coefs = np.delete(current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1
    
    #return [initial_functions_phase_1, np.array(initial_sequence_phase_1, dtype = 'int32')], [initial_functions_phase_2, np.array(initial_sequence_phase_2, dtype = 'int32')], [recurrent_functions_phase_1, np.array(recurrent_sequence_phase_1, dtype = 'int32')], [recurrent_functions_phase_2, np.array(recurrent_sequence_phase_2, dtype = 'int32')], [final_functions_phase_1, np.array(final_sequence_phase_1, dtype = 'int32')]
    return [initial_functions_phase_1, initial_sequence_phase_1], [initial_functions_phase_2, initial_sequence_phase_2], [recurrent_functions_phase_1, recurrent_sequence_phase_1], [recurrent_functions_phase_2, recurrent_sequence_phase_2], [final_functions_phase_1, final_sequence_phase_1]


