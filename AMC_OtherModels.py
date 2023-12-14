#!/usr/bin/env python
# coding: utf-8

import numpy as np

from scipy.fft import fft, fftfreq, fftshift

from tensorflow.random import set_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ## “Effect of training algorithms on performance of a developed automatic modulation classification using artificial neural network”
# ### J. Popoola and R. van Olst

# In[ ]:



def calcFeatures_Popoola (X, idx, mods, X_mods, lbl, usedSnr):
      
    n_train = len(idx)
    n_snr = len(usedSnr)
    
    v20_values = np.zeros((len(mods) * n_snr, n_train))
    beta_values = np.zeros((len(mods) * n_snr, n_train))
    X_values = np.zeros((len(mods) * n_snr, n_train))
    gamma_max_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_ap_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_dp_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_aa_values = np.zeros((len(mods) * n_snr, n_train))
      
    for i,mod in enumerate (mods): 
        idx_sameMod = np.where(np.array(X_mods) == mod)[0]
        X_sameMod = X[idx_sameMod]  
        
        for j,snr in enumerate (usedSnr):  
            snr_sameMod = np.array(list(map(lambda x: lbl[x][1], idx_sameMod)))
            idx_sameMod_snr = np.where(np.array(snr_sameMod) == snr)[0]
            X_sameMod_snr = X_sameMod[idx_sameMod_snr]   
            
            X_idx = X_sameMod_snr[idx]

            for ex in range(0, X_idx.shape[0]):
                
                X_0 = X_idx[ex][0]
                X_1 = X_idx[ex][1]
                

                signal = X_0 + X_1 * 1j
                
                # The next normalization is the one implemented on the public dataset.
                # We implement it in the same way to normalize the cases when the signals are interfered
                ener = sum(abs(signal))
                signal = signal / np.sqrt(ener)
                
                ampli = abs(signal) 
                
                X_0 = np.real(signal)
                X_1 = np.imag(signal)
                
                ma = ampli.mean()             
                
                #-------------- 
                
                angle = np.angle(signal)
                
                sigma_dp_values[j + i*n_snr][ex] = angle.std()
                
                #---------------
                
                angle = abs(angle)
                
                sigma_ap_values[j + i*n_snr][ex] = angle.std()
                
                #---------------------- 
                 
                ma = ampli.mean()
                
                acn = ampli / ma - 1
                
                acn2 = abs(acn)
                
                sigma_aa_values[j + i*n_snr][ex] = acn2.std()
                #------------------
         
                N = len(ampli)
            
                acn = ampli / ma - 1
                
                dft = fft(acn)
                
                gamma_max_values[j + i*n_snr][ex] = np.max(np.abs(dft)**2) / N
            
                #------------------------------
                
                M21 = (signal * np.conj(signal)).mean()

                M42 = (signal**2 * np.conj(signal)**2).mean()
                
                v20_values[j + i*n_snr][ex] = np.real(M42 / M21**2)
                
                #-----------------
                
                ai_2 = X_0**2
                aq_2 = X_1**2
                
                beta_values[j + i*n_snr][ex] = sum(aq_2) / sum(ai_2)
                
                #----------------
      
                X_values[j + i*n_snr][ex] = ma
                
    return gamma_max_values, sigma_ap_values,            sigma_dp_values, sigma_aa_values,             X_values, v20_values,            beta_values


# In[ ]:


def generateAllDataArray_Popoola(X, idx, mods, X_mods, lbl, usedSnr):
 

    gamma_max_values, sigma_ap_values,    sigma_dp_values, sigma_aa_values,    X_values, v20_values,    beta_values = calcFeatures_Popoola (X, idx, mods, X_mods, lbl, usedSnr)  
    
    X_all = np.hstack((                         gamma_max_values[0].reshape(-1, 1),                         sigma_ap_values[0].reshape(-1, 1),                         sigma_dp_values[0].reshape(-1, 1),                         sigma_aa_values[0].reshape(-1, 1),                         X_values[0].reshape(-1, 1),                         v20_values[0].reshape(-1, 1),                         beta_values[0].reshape(-1, 1)                      ))


    for p in range(1, len(usedSnr) * len(mods)):
        X_data = np.hstack((                            gamma_max_values[p].reshape(-1, 1),                             sigma_ap_values[p].reshape(-1, 1),                             sigma_dp_values[p].reshape(-1, 1),                             sigma_aa_values[p].reshape(-1, 1),                             X_values[p].reshape(-1, 1),                             v20_values[p].reshape(-1, 1),                             beta_values[p].reshape(-1, 1)                            ))

        X_all = np.vstack((X_all, X_data))

    return X_all


# In[ ]:


def generateNeuralNetwork_Popoola(X_shape, Y_shape, lr):
    
    model = Sequential()
    
    model.add(BatchNormalization(input_shape = (X_shape,)))
    
    model.add(Dense(7, activation = 'relu',name = "dense1"))
    
    model.add(Dense(Y_shape, activation = 'softmax', name = "dense2"))
    
    optA = Adam(learning_rate = lr)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = optA, metrics = ['accuracy'])
    
    return model


# <hr>
# 
# ## “Modulation classification based on statistical features and artificial neural network”
# 
# ### A. Alarbi and O. Alkishriwo

# In[ ]:


def calcFeatures_Alarbi (X, idx, mods, X_mods, lbl, usedSnr):
      
    n_train = len(idx)
    n_snr = len(usedSnr)
    
    var_amp_values = np.zeros((len(mods)*n_snr, n_train))
    var_phase_values = np.zeros((len(mods)*n_snr, n_train))
    kurtosis_amp_values = np.zeros((len(mods)*n_snr, n_train))
    kurtosis_phase_values = np.zeros((len(mods)*n_snr, n_train))

    entropy_amp_values = np.zeros((len(mods)*n_snr, n_train))
    entropy_phase_values = np.zeros((len(mods)*n_snr, n_train))
    skewness_values = np.zeros((len(mods)*n_snr, n_train))
      
    for i,mod in enumerate (mods): 
        idx_sameMod = np.where(np.array(X_mods) == mod)[0]
        X_sameMod = X[idx_sameMod]  
        
        for j,snr in enumerate (usedSnr):  
            snr_sameMod = np.array(list(map(lambda x: lbl[x][1], idx_sameMod)))
            idx_sameMod_snr = np.where(np.array(snr_sameMod) == snr)[0]
            X_sameMod_snr = X_sameMod[idx_sameMod_snr]   
            
            X_idx = X_sameMod_snr[idx]

            for ex in range(0, X_idx.shape[0]):
                
                X_0 = X_idx[ex][0]
                X_1 = X_idx[ex][1]
                

                signal = X_0 + X_1 * 1j
                
                # The next normalization is the one implemented on the public dataset.
                # We implement it in the same way to normalize the cases when the signals are interfered
                ener = sum(abs(signal))
                signal = signal / np.sqrt(ener)
                
                ampli = abs(signal) 
                
                X_0 = np.real(signal)
                X_1 = np.imag(signal)
                
                ma = ampli.mean()             
                
                #-------------- 
                
                var_amp_values[j + i*n_snr][ex] = ampli.mean()

                #------------------
                 
                angle = np.angle(signal)
                
                var_phase_values[j + i*n_snr][ex] = angle.var()
                
                #-------------- 
                
                acn = ampli - ma
                
                kurtosis_amp_values[j + i*n_snr][ex] = ( acn**4 /  ampli.std()**4 ).mean()

                #------------------
                
                mp = angle.mean()
                
                pcn = angle - mp
                
                kurtosis_phase_values[j + i*n_snr][ex] = ( pcn**4 /  angle.std()**4 ).mean()
                
                #------------------------------
                
                p_ampli, bins = np.histogram(ampli,len(ampli),density = True);

                p_ampli = p_ampli[p_ampli != 0]

                entropy_amp_values[j + i*n_snr][ex] = - sum(p_ampli * np.log2(p_ampli))
                
                #-----------------
                
                p_phase, bins = np.histogram(angle,len(angle),density = True);

                p_phase = p_phase[p_phase != 0]

                entropy_phase_values[j + i*n_snr][ex] = - sum(p_phase * np.log2(p_phase))
                
                #----------------
                num = ((ampli - ma)**3).mean()
                
                skewness_values[j + i*n_snr][ex] = num / ampli.std()**3
                
    return var_amp_values, var_phase_values, kurtosis_amp_values,         kurtosis_phase_values, entropy_amp_values, entropy_phase_values,         skewness_values


# In[ ]:


def generateAllDataArray_Alarbi(X, idx, mods, X_mods, lbl, usedSnr):
 

    var_amp_values,var_phase_values, kurtosis_amp_values,     kurtosis_phase_values, entropy_amp_values, entropy_phase_values,     skewness_values = calcFeatures_Alarbi (X, idx, mods, X_mods, lbl, usedSnr)  
    
    X_all = np.hstack((                         var_amp_values[0].reshape(-1, 1),                         var_phase_values[0].reshape(-1, 1),                         kurtosis_amp_values[0].reshape(-1, 1),                         kurtosis_phase_values[0].reshape(-1, 1),                         entropy_amp_values[0].reshape(-1, 1),                         entropy_phase_values[0].reshape(-1, 1),                         skewness_values[0].reshape(-1, 1)                      ))


    for p in range(1, len(usedSnr) * len(mods)):
        X_data = np.hstack((                        var_amp_values[p].reshape(-1, 1),                         var_phase_values[p].reshape(-1, 1),                         kurtosis_amp_values[p].reshape(-1, 1),                         kurtosis_phase_values[p].reshape(-1, 1),                         entropy_amp_values[p].reshape(-1, 1),                         entropy_phase_values[p].reshape(-1, 1),                         skewness_values[p].reshape(-1, 1)                            ))

        X_all = np.vstack((X_all, X_data))

    return X_all


# In[ ]:


def generateNeuralNetwork_Alarbi(X_shape, Y_shape, lr):
    
    model = Sequential()
    
    model.add(BatchNormalization(input_shape = (X_shape,)))
    
    model.add(Dense(48, activation = 'tanh',name = "dense1"))
    
    model.add(Dense(32, activation = 'tanh',name = "dense2"))
    
    model.add(Dense(Y_shape, activation = 'softmax', name = "dense3"))
    
    optA = Adam(learning_rate = lr)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = optA, metrics = ['accuracy'])
    
    return model


# <hr>
# 
# ## “Application of artificial neural networks in classification of digital modulations for software defined radio”
# 
# ### M. M. Roganovic, A. M. Neskovic, and N. J. Neskovic

# In[ ]:


def calcFeatures_Roganovic (X, idx, mods, X_mods, lbl, usedSnr):
      
    n_train = len(idx)
    n_snr = len(usedSnr)
    
    C1_values = np.zeros((len(mods) * n_snr, n_train))
    C7_values = np.zeros((len(mods) * n_snr, n_train))
    C12_values = np.zeros((len(mods) * n_snr, n_train))
    gamma_max_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_dp_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_af_values = np.zeros((len(mods) * n_snr, n_train))
      
    for i,mod in enumerate (mods): 
        idx_sameMod = np.where(np.array(X_mods) == mod)[0]
        X_sameMod = X[idx_sameMod]  
        
        for j,snr in enumerate (usedSnr):  
            snr_sameMod = np.array(list(map(lambda x: lbl[x][1], idx_sameMod)))
            idx_sameMod_snr = np.where(np.array(snr_sameMod) == snr)[0]
            X_sameMod_snr = X_sameMod[idx_sameMod_snr]   
            
            X_idx = X_sameMod_snr[idx]

            for ex in range(0, X_idx.shape[0]):
                
                X_0 = X_idx[ex][0]
                X_1 = X_idx[ex][1]
                

                signal = X_0 + X_1 * 1j
                
                # The next normalization is the one implemented on the public dataset.
                # We implement it in the same way to normalize the cases when the signals are interfered
                ener = sum(abs(signal))
                signal = signal / np.sqrt(ener)
        
                ampli = abs(signal) 
                
                X_0 = np.real(signal)
                X_1 = np.imag(signal)
                
                ma = ampli.mean()             
                
                acn = ampli / ma - 1
                
                #-------------- 
                
                angle = np.angle(signal)
                
                sigma_dp_values[j + i*n_snr][ex] = angle.std()
                
                #-----------------------
                
                frec = 1 / (2 * np.pi) * 1 / (1 + (X_1 / X_0)**2)
                 
                mf = frec.mean()
                
                frec = frec / mf - 1
                
                frec2 = abs(frec)
                
                sigma_af_values[j + i*n_snr][ex] = frec2.std()
                
                #--------------------------------
                
                N = len(ampli)
                
                dft = fft(acn)
                
                gamma_max_values[j + i*n_snr][ex] = np.max(np.abs(dft)**2) / N
                
                #---------------
                
                C1_values[j + i*n_snr][ex] = abs((angle**2).mean())
                
                
                M3 = (frec**4).mean()
                
                C7_values[j + i*n_snr][ex] = abs(M3)
                
                
                M20 = (frec**2).mean()
                
                
                M40 = (frec**4).mean()
                
                
                C12_value = M40 - 3 * M20**2
                
                C12_values[j + i*n_snr][ex] = abs(C12_value)
                
    return C1_values, C7_values, C12_values,    gamma_max_values, sigma_dp_values, sigma_af_values 


# In[ ]:


def generateAllDataArray_Roganovic(X, idx, mods, X_mods, lbl, usedSnr):
 
    C1_values, C7_values, C12_values,    gamma_max_values, sigma_dp_values,     sigma_af_values  = calcFeatures_Roganovic (X, idx, mods, X_mods, lbl, usedSnr)  
    
    X_all = np.hstack((                         C1_values[0].reshape(-1, 1),                         C7_values[0].reshape(-1, 1),                         C12_values[0].reshape(-1, 1),                         gamma_max_values[0].reshape(-1, 1),                         sigma_dp_values[0].reshape(-1, 1),                         sigma_af_values[0].reshape(-1, 1)                     ))


    for p in range(1, len(usedSnr) * len(mods)):
        X_data = np.hstack((                        C1_values[p].reshape(-1, 1),                         C7_values[p].reshape(-1, 1),                         C12_values[p].reshape(-1, 1),                         gamma_max_values[p].reshape(-1, 1),                         sigma_dp_values[p].reshape(-1, 1),                         sigma_af_values[p].reshape(-1, 1)                           ))

        X_all = np.vstack((X_all, X_data))

    return X_all


# In[ ]:


def generateNeuralNetwork_Roganovic(X_shape, Y_shape, lr):
    
    model = Sequential()
    
    model.add(BatchNormalization(input_shape = (X_shape,)))
    
    model.add(Dense(48, activation = 'tanh',name = "dense1"))
    
    model.add(Dense(32, activation = 'tanh',name = "dense2"))
    
    model.add(Dense(Y_shape, activation = 'softmax', name = "dense3"))
    
    optA = Adam(learning_rate = lr)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = optA, metrics = ['accuracy'])
    
    return model


# <hr>
# 
# ## “Artificial neural network based automatic modulation classification over a software defined radio testbed”
# 
# ### J. Jagannath, N. Polosky, D. O’Connor, L. N. Theagarajan, B. Sheaffer, S. Foulke, and P. K. Varshney

# In[ ]:


def calcFeatures_Jagannath (X, idx, mods, X_mods, lbl, usedSnr):
      
    n_train = len(idx)
    n_snr = len(usedSnr)
    
    var_amp_values = np.zeros((len(mods) * n_snr, n_train))
    gamma_max_values = np.zeros((len(mods) * n_snr, n_train))
    C_ratio_values = np.zeros((len(mods) * n_snr, n_train))
    var_f_values = np.zeros((len(mods) * n_snr, n_train))
    var_deviation_values = np.zeros((len(mods) * n_snr, n_train))
          
    for i,mod in enumerate (mods): 
        idx_sameMod = np.where(np.array(X_mods) == mod)[0]
        X_sameMod = X[idx_sameMod]  
        
        for j,snr in enumerate (usedSnr):  
            snr_sameMod = np.array(list(map(lambda x: lbl[x][1], idx_sameMod)))
            idx_sameMod_snr = np.where(np.array(snr_sameMod) == snr)[0]
            X_sameMod_snr = X_sameMod[idx_sameMod_snr]   
            
            X_idx = X_sameMod_snr[idx]

            for ex in range(0, X_idx.shape[0]):
                
                X_0 = X_idx[ex][0]
                X_1 = X_idx[ex][1]

                signal = X_0 + X_1*1j
                
                # The next normalization is the one implemented on the public dataset.
                # We implement it in the same way to normalize the cases when the signals are interfered
                ener = sum(abs(signal))
                signal = signal / np.sqrt(ener)
                
                max_val = max(max(np.abs(signal.real)), max(np.abs(signal.imag)))
                signal = signal / max_val
                
                ampli = abs(signal) 
                
                X_0 = np.real(signal)
                X_1 = np.imag(signal)
                
                ma = ampli.mean()             
                
                #------------------ 
                
                var_amp_values[j + i*n_snr][ex] = ampli.var()

                #------------------
                
                N = len(ampli)
                
                acn = ampli / ma -1
                
                dft = fft(acn)
                
                gamma_max_values[j + i*n_snr][ex] = np.max(np.abs(dft)**2) / N
                
                #-------------------
                
                M20 = (signal**2).mean()
                M21 = (signal * np.conj(signal)).mean()
                M40 = (signal**4).mean()
                M42 = (signal**2 * np.conj(signal)**2).mean()
                
                C40_value = M40 - 3 * M20**2
                
                C42_value = M42 -  abs(M20)**2 - 2 * M21**2
                
                C_ratio_values[j+i*n_snr][ex] = abs(C42_value / C40_value)
                
                #-------------------
                
                dft = np.abs(np.fft.fftshift(np.fft.fft(signal, n = 64)))

                n = len(dft)           
                dx = 1 / 1e6            
                frec_s = np.fft.fftshift(np.fft.fftfreq(n, dx))

                F = []
                for fi in range(len(frec_s) - 1):
                       F.append((dft[fi + 1] - dft[fi])/(abs(frec_s[fi + 1] - frec_s[fi])))

                var_f_values[j + i*n_snr][ex] = np.var(F)
                
                #-------------------
                
                var_deviation_values[j + i*n_snr][ex] = np.var(abs(signal) / (abs(signal)).mean() - 1)
                
                #--------------------
                
                
    return var_amp_values, gamma_max_values,     C_ratio_values, var_f_values, var_deviation_values


# In[ ]:


def generateAllDataArray_Jagannath(X, idx, mods, X_mods, lbl, usedSnr):
 

    var_amp_values, gamma_max_values,     C_ratio_values, var_f_values,     var_deviation_values = calcFeaturesPolosky(X, idx, mods, X_mods, lbl, usedSnr)  
    
    X_all = np.hstack((                         var_amp_values[0].reshape(-1, 1),                         gamma_max_values[0].reshape(-1, 1),                         C_ratio_values[0].reshape(-1, 1),                         var_f_values[0].reshape(-1, 1),                         var_deviation_values[0].reshape(-1, 1)                     ))


    for p in range(1, len(usedSnr) * len(mods)):
        X_data = np.hstack((                        var_amp_values[p].reshape(-1, 1),                         gamma_max_values[p].reshape(-1, 1),                         C_ratio_values[p].reshape(-1, 1),                         var_f_values[p].reshape(-1, 1),                         var_deviation_values[p].reshape(-1, 1)                           ))

        X_all = np.vstack((X_all, X_data))

    return X_all


# In[ ]:


def generateNeuralNetwork_Jagannath(X_shape, Y_shape, lr, n_layers):
    
    model = Sequential()
    
    model.add(BatchNormalization(input_shape = (X_shape,)))
    
    model.add(Dense(50, activation = 'sigmoid',name = "dense1"))
    
    if n_layers != 1:
    
        model.add(Dense(25, activation = 'sigmoid',name = "dense2"))
    
    model.add(Dense(Y_shape, activation = 'softmax', name = "dense3"))
    
    optA = Adam(learning_rate = lr)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = optA, metrics = ['accuracy'])
    
    return model


# <hr>
# 
# ## “Artificial intelligence-driven real-time automatic modulation classification scheme for next-generation cellular networks”
# 
# ### Z. Kaleem, M. Ali, I. Ahmad, W. Khalid, A. Alkhayyat, and A. Jamalipour

# In[ ]:


def calcFeatures_Kaleem (X, idx, mods, X_mods, lbl, usedSnr):
      
    n_train = len(idx)
    n_snr = len(usedSnr)
    
    gamma_max_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_ap_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_dp_values = np.zeros((len(mods) * n_snr, n_train))
    P_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_aa_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_af_values = np.zeros((len(mods) * n_snr, n_train))
    sigma_a_values = np.zeros((len(mods) * n_snr, n_train))
    kurtosis_values = np.zeros((len(mods) * n_snr, n_train))
    kurtosis_f_values = np.zeros((len(mods) * n_snr, n_train))
          
    for i,mod in enumerate (mods): 
        idx_sameMod = np.where(np.array(X_mods) == mod)[0]
        X_sameMod = X[idx_sameMod]  
        
        for j,snr in enumerate (usedSnr):  
            snr_sameMod = np.array(list(map(lambda x: lbl[x][1], idx_sameMod)))
            idx_sameMod_snr = np.where(np.array(snr_sameMod) == snr)[0]
            X_sameMod_snr = X_sameMod[idx_sameMod_snr]   
            
            X_idx = X_sameMod_snr[idx]

            for ex in range(0, X_idx.shape[0]):
                
                X_0 = X_idx[ex][0]
                X_1 = X_idx[ex][1]
                
                signal = X_0 + X_1 * 1j
                
                # The next normalization is the one implemented on the public dataset.
                # We implement it in the same way to normalize the cases when the signals are interfered
                ener = sum(abs(signal))
                signal = signal / np.sqrt(ener)
                
                ampli = abs(signal) 
                
                X_0 = np.real(signal)
                X_1 = np.imag(signal)
                
                ma = ampli.mean()             
                
                #------------------ 
            
                N = len(ampli)
                
                acn = ampli / ma -1
                
                dft = fft(acn)
                
                gamma_max_values[j + i*n_snr][ex] = np.max(np.abs(dft)**2) / N
                
                #-------------------
                
                angle = np.angle(signal)
                
                sigma_dp_values[j + i*n_snr][ex] = angle.std()
                
                #-------------------
                
                angle2 = abs(angle)
                
                sigma_ap_values[j + i*n_snr][ex] = angle2.std()
                
                #-------------------
                
                dft = np.fft.fftshift(fft(signal))
                
                PL = sum(abs(dft[0 : int(len(signal) / 2)])**2)
                
                PU = sum(abs(dft[int(len(signal)/2 + 1):-1])**2)
                
                P_values[j + i*n_snr][ex] = (PL - PU) / (PL + PU)
                
                #------------------
                
                acn2 = abs(acn)
                
                sigma_aa_values[j + i*n_snr][ex] = acn2.std()
                
                #------------------
                
                frec = 1 / (2 * np.pi) * 1 / (1 + (X_1 / X_0)**2)
                
                mf = frec.mean()
                
                frec = frec / mf - 1
                
                frec = abs(frec)
                
                sigma_af_values[j + i*n_snr][ex] = frec.std()
                
                #------------------
                
                sigma_a_values[j + i*n_snr][ex] = acn.std()
                
                #------------------
                
                acn = ampli / ma - 1
                
                kurtosis_values[j + i*n_snr][ex] = (acn**4).mean() /  (acn**2).mean()

                #-------------------
                
                frec = 1 / (2 * np.pi) * 1 / (1 + (X_1 / X_0)**2)
                
                mf = frec.mean()
                sf = frec.std()
                
                frec = frec / mf - 1
                
                kurtosis_f_values[j + i*n_snr][ex] = (frec**4).mean() /  (frec**2).mean()
                
                #------------------
                
    return gamma_max_values, sigma_ap_values, sigma_dp_values,     P_values, sigma_aa_values, sigma_af_values, sigma_a_values,     kurtosis_values, kurtosis_f_values


# In[ ]:


def generateAllDataArray_Kaleem(X, idx, mods, X_mods, lbl, usedSnr):
 

    gamma_max_values, sigma_ap_values, sigma_dp_values,     P_values, sigma_aa_values, sigma_af_values, sigma_a_values,     kurtosis_values, kurtosis_f_values = calcFeatures_Kaleem(X, idx, mods, X_mods, lbl, usedSnr)  
    
    X_all = np.hstack((                         gamma_max_values[0].reshape(-1, 1),                         sigma_ap_values[0].reshape(-1, 1),                         sigma_dp_values[0].reshape(-1, 1),                         P_values[0].reshape(-1, 1),                         sigma_aa_values[0].reshape(-1, 1),                         sigma_af_values[0].reshape(-1, 1),                         sigma_a_values[0].reshape(-1, 1),                         kurtosis_values[0].reshape(-1, 1),                         kurtosis_f_values[0].reshape(-1, 1)                      ))


    for p in range(1, len(usedSnr) * len(mods)):
        X_data = np.hstack((                        gamma_max_values[p].reshape(-1, 1),                         sigma_ap_values[p].reshape(-1, 1),                         sigma_dp_values[p].reshape(-1, 1),                         P_values[p].reshape(-1, 1),                         sigma_aa_values[p].reshape(-1, 1),                         sigma_af_values[p].reshape(-1, 1),                         sigma_a_values[p].reshape(-1, 1),                         kurtosis_values[p].reshape(-1, 1),                         kurtosis_f_values[p].reshape(-1, 1)                            ))

        X_all = np.vstack((X_all, X_data))

    return X_all


# In[1]:


def generateNeuralNetwork_Kaleem(X_shape, Y_shape, lr):
    
    model = Sequential()
    
    model.add(BatchNormalization(input_shape = (X_shape,)))
    
    model.add(Dense(25, activation = 'sigmoid',name = "dense1"))
    
    model.add(Dense(12, activation = 'sigmoid',name = "dense2"))
    
    model.add(Dense(Y_shape, activation = 'softmax', name = "dense3"))
    
    optA = Adam(learning_rate = lr)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = optA, metrics = ['accuracy'])
    
    return model


# In[ ]:




