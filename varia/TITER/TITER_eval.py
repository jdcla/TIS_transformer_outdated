# code largely imported from https://github.com/zhangsaithu/titer

import sys
import pandas as pd
import numpy as np
import h5py
import scipy.io
import theano
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from sklearn.metrics import average_precision_score, roc_auc_score
import theano.ifelse

def unbinarize_DNA(dna_bin, img=False, quick=False, custom_dict=None):
    if len(dna_bin.shape) == 1:
        dna_bin= dna_bin.reshape(-1,1)
    if custom_dict is None:
        img_dict = {1: {1: {1: {1: 'N', 0: 'H'}, 0: {1: 'D', 0: 'W'}},
                        0: {1: {1: 'V', 0: 'M'}, 0: {1: 'R', 0: 'A'}}},
                    0: {1: {1: {1: 'B', 0: 'Y'}, 0: {1: 'K', 0: 'T'}},
                        0: {1: {1: 'S', 0: 'C'}, 0: {1: 'G', 0: 'N'}}}}
        img_dict_simple = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
        int_dict = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'R', 5: 'Y',
                    6: 'S', 7: 'W', 8: 'K', 9: 'M', 10: 'B', 11: 'D',
                    12: 'H', 13: 'V', 14: 'N'}
    else:
        img_dict = img_dict_simple = int_dict = custom_dict
        
    if img:
        dna_str = np.full(dna_bin.shape[0], 'N', dtype='|S1')
        dna_bin_stripped = dna_bin[:, :4]
        if quick:
            for idx, nt_img in enumerate(dna_bin_stripped):
                dna_str[idx] = img_dict_simple[nt_img.argmax()]
        else:
            for idx, nt_img in enumerate(dna_bin_stripped):
                value = img_dict[nt_img[0]][nt_img[1]][nt_img[2]][nt_img[3]]
                dna_str[idx] = value
    else:
        dna_str = np.full(dna_bin.shape[0], 'N', dtype='|S1')
        for idx, nt_img in enumerate(dna_bin):
            dna_str[idx] = int_dict[nt_img[0]]

    return dna_str.tostring().decode('utf-8')

def seq_matrix(seq_list,label):
    tensor = np.zeros((len(seq_list),203,8), dtype=np.int8)
    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A' and (j<100 or j>102):
                tensor[i][j] = [1,0,0,0,0,0,0,0]
            if s == 'T' and (j<100 or j>102):
                tensor[i][j] = [0,1,0,0,0,0,0,0]
            if s == 'C' and (j<100 or j>102):
                tensor[i][j] = [0,0,1,0,0,0,0,0]
            if s == 'G' and (j<100 or j>102):
                tensor[i][j] = [0,0,0,1,0,0,0,0]
            if s == '$':
                tensor[i][j] = [0,0,0,0,0,0,0,0]
            if s == 'A' and (j>=100 and j<=102):
                tensor[i][j] = [0,0,0,0,1,0,0,0]
            if s == 'T' and (j>=100 and j<=102):
                tensor[i][j] = [0,0,0,0,0,1,0,0]
            if s == 'C' and (j>=100 and j<=102):
                tensor[i][j] = [0,0,0,0,0,0,1,0]
            if s == 'G' and (j>=100 and j<=102):
                tensor[i][j] = [0,0,0,0,0,0,0,1]
            j += 1
        if label == 1:
            y = np.ones((len(seq_list),1))
        else:
            y = np.zeros((len(seq_list),1))
        return tensor, y

###### main function ######
codon_tis_prior = np.load('dict_piror_front_Gaotrain.npy', allow_pickle=True)
codon_tis_prior = codon_tis_prior.item()
codon_tis_prior['CAT'] = -10
codon_tis_prior['TAT'] = -10
codon_tis_prior['TGT'] = -10
codon_tis_prior['TGC'] = -10
codon_tis_prior['TGG'] = -10
codon_tis_prior['TGA'] = -10

codon_list = []
for c in codon_tis_prior.keys():
    if codon_tis_prior[c]!='never' and codon_tis_prior[c] >= 1:
        codon_list.append(c)
###
codon_list = list(codon_tis_prior.keys())


### PREPARE CUSTOM DATA SET

data = np.load('../../data/GRCh38p13/chr4.npy', allow_pickle=True)

buff = np.zeros((100,2), dtype=np.int)
buff[:,0] = 5 
cust_dict = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'N', 5: '$'}
pos_samples = []
neg_samples = []
print('Generating input data')
for idx, s in enumerate(data):
    tr = np.concatenate((buff, s[0], buff))
    tr_str = unbinarize_DNA(tr, custom_dict=cust_dict)
    for i in np.arange(len(tr_str)-203):
        if tr[i+100,1] == 0:
            neg_samples.append(tr_str[i:i+203])
        else:
            pos_samples.append(tr_str[i:i+203])

pos_seq_test = pos_samples
neg_seq_test = neg_samples
pos_codon = []
neg_codon = []
for s in pos_seq_test:
    s = s
    if s[100:103] in codon_list:
        pos_codon.append(codon_tis_prior[s[100:103]])
for s in neg_seq_test:
    s = s
    if s[100:103] in codon_list:
        neg_codon.append(codon_tis_prior[s[100:103]])

        
pos_codon = np.array(pos_codon)
neg_codon = np.array(neg_codon)
codon = np.concatenate((pos_codon,neg_codon)).reshape((len(pos_codon)+len(neg_codon),1))

pos_seq_test1 = []
neg_seq_test1 = []
for s in pos_seq_test:
    s = s
    if s[100:103] in codon_list:
        pos_seq_test1.append(s)
for s in neg_seq_test:
    s = s
    if s[100:103] in codon_list:
        neg_seq_test1.append(s)

pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test1, label=1)
neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test1, label=0)
X_test = np.concatenate((pos_test_X,neg_test_X), axis=0)
y_test = np.concatenate((pos_test_y,neg_test_y), axis=0)


### DEFINE THE MODEL

model = Sequential()
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        input_dim=8,
                        input_length=203,
                        border_mode='valid',
                        W_constraint = maxnorm(3),
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=3))
model.add(Dropout(p=0.21370950078747658))
model.add(LSTM(output_dim=256,
               return_sequences=True))
model.add(Dropout(p=0.7238091317104384))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

y_test_pred_n = np.zeros((len(y_test),1))
y_test_pred_p = np.zeros((len(y_test),1))


### IMPUTE LABELS

models = 31
for i in np.arange(models):
    model.load_weights('TITER_models/bestmodel_'+str(i)+'.hdf5')
    y_test_pred = model.predict(X_test,batch_size=1000,verbose=1)
    y_test_pred_n += y_test_pred
    y_test_pred_p += y_test_pred*codon
    

y_test_pred_n = y_test_pred_n/models
y_test_pred_p = y_test_pred_p/models

print(f'Perf without prior, AUC: {str(roc_auc_score(y_test, y_test_pred_n))}')
print(f'Perf without prior, AUPR: {str(average_precision_score(y_test, y_test_pred_n))}')
print(f'Perf with prior, AUC: {str(roc_auc_score(y_test, y_test_pred_p))}')
print(f'Perf with prior, AUPR: {str(average_precision_score(y_test, y_test_pred_p))}')

out = [f'Perf without prior, AUC: {str(roc_auc_score(y_test, y_test_pred_n))}',
       f'Perf without prior, AUPR: {str(average_precision_score(y_test, y_test_pred_n))}',
       f'Perf with prior, AUC: {str(roc_auc_score(y_test, y_test_pred_p))}',
       f'Perf with prior, AUPR: {str(average_precision_score(y_test, y_test_pred_p))}']
np.save('results_out.npy', out)