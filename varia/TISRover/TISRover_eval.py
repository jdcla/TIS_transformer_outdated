#!/usr/bin/env python
# build network

# Dependencies: 
# - theano v0.8.2
# - numpy v1.14
# - lasagne v0.2dev

import os
#os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32,optimizer=None,cxx='
import sys

tmp = sys.stderr
sys.stderr = open(os.devnull, 'w') # to suppress import warning
import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne.nonlinearities as LN
sys.stderr = tmp # /to suppress import warning

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
beforeAndAfterSpliceSite = 5

from sklearn.metrics import average_precision_score, roc_auc_score

def binarize_DNA(dna_seq, img=False, dtype=np.uint8, custom_dict=None):
    if custom_dict is None:
        dict_img = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'R': [0, 3],
                    'Y': [1, 2], 'S': [2, 3], 'W': [0, 1], 'K': [1, 3],
                    'M': [0, 2], 'B': [1, 2, 3], 'D': [0, 1, 3],
                    'H': [0, 1, 2], 'V': [0, 2, 3], 'N': [0, 1, 2, 3]}
        seq_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'R': 4,
                    'Y': 5, 'S': 6, 'W': 7, 'K': 8, 'M': 9, 'B': 10,
                    'D': 11, 'H': 12, 'V': 13, 'N': 14}
    elif img:
        dict_img = custom_dict
    else:
        seq_dict = custom_dict
    if img:
        dna = np.zeros((len(dna_seq), 4), dtype=dtype)
        for idx in np.arange(len(dna_seq)):
            dna[idx, dict_img[dna_seq[idx]]] = 1
    else:
        dna = np.zeros((len(dna_seq), 1), dtype=dtype)
        for idx in np.arange(len(dna_seq)):
            dna[idx] = seq_dict[dna_seq[idx]]

    return dna


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


def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath("translation_initation_sites.params")))
    return os.path.join(base_path, relative_path)


def buildNetwork(parameter_file, xWidth, task_is_translation):
    X_input = T.imatrix()
    model = getEmbeddingModel()
    if task_is_translation:
        nn = L.InputLayer(shape=(None,xWidth), input_var = X_input)
        nn = L.EmbeddingLayer(nn,5,4,W=model)
        nn = L.ReshapeLayer(nn,([0],1,xWidth,4))
        nn = L.Conv2DLayer(nn,filter_size=(7,4),num_filters=70,pad='valid')
        nn = L.MaxPool2DLayer(nn,(3,1))
        nn = L.DropoutLayer(nn,0.2)
        nn = L.Conv2DLayer(nn,filter_size=(3,1),num_filters=100,pad='valid')
        nn = L.MaxPool2DLayer(nn,(3,1))
        nn = L.DropoutLayer(nn,0.2)
        nn = L.Conv2DLayer(nn,filter_size=(3,1),num_filters=150,pad='valid')
        nn = L.MaxPool2DLayer(nn,(3,1))
        nn = L.DropoutLayer(nn,0.2)
        nn = L.FlattenLayer(nn,outdim=2)
        nn = L.DenseLayer(nn,512)
        nn = L.DropoutLayer(nn,0.2)
        nn = L.DenseLayer(nn,2,nonlinearity=LN.softmax)
    else:
        nn = L.InputLayer(shape=(None,xWidth), input_var = X_input)
        nn = L.EmbeddingLayer(nn,5,4,W=model)
        nn = L.ReshapeLayer(nn,([0],1,xWidth,4))
        nn = L.Conv2DLayer(nn,filter_size=(9,4),num_filters=70,pad='valid')
        nn = L.DropoutLayer(nn,0.2)
        nn = L.Conv2DLayer(nn,filter_size=(7,1),num_filters=100,pad='valid')
        nn = L.DropoutLayer(nn,0.2)
        nn = L.Conv2DLayer(nn,filter_size=(7,1),num_filters=100,pad='valid')
        nn = L.MaxPool2DLayer(nn,(3,1))
        nn = L.DropoutLayer(nn,0.2)
        nn = L.Conv2DLayer(nn,filter_size=(7,1),num_filters=200,pad='valid')
        nn = L.MaxPool2DLayer(nn,(4,1))
        nn = L.DropoutLayer(nn,0.2)
        nn = L.Conv2DLayer(nn,filter_size=(7,1),num_filters=250,pad='valid')
        nn = L.MaxPool2DLayer(nn,(4,1))
        nn = L.DropoutLayer(nn,0.2)
        nn = L.FlattenLayer(nn,outdim=2)
        nn = L.DenseLayer(nn,512)
        nn = L.DropoutLayer(nn,0.2)
        nn = L.DenseLayer(nn,2,nonlinearity=LN.softmax)
    params = np.load(resource_path(parameter_file), allow_pickle=True, encoding='latin1')
    L.set_all_param_values(nn, params)
    return nn, X_input

def getEmbeddingModel():
    m = np.zeros((5,4),dtype=np.float32)
    m[0][0] = 1
    m[1][1] = 1
    m[2][2] = 1
    m[3][3] = 1
    m[4][0] = 0.25
    m[4][1] = 0.25
    m[4][2] = 0.25
    m[4][3] = 0.25
    return m

data = np.load('../../data/GRCh38p13/chr4.npy', allow_pickle=True)

cust_dict = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'N'}
cust_dict_b = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}
samples = []
labels = []
for idx, s in enumerate(data):
    #tr = np.concatenate((buff, s[0], buff))
    tr = s[0]
    tr_str = unbinarize_DNA(tr, custom_dict=cust_dict)
    for i in np.arange(60,len(tr_str)-143):
        if tr_str[i:i+3] == 'ATG':
            samples.append(binarize_DNA(tr_str[i-60:i+143], custom_dict=cust_dict_b).ravel())
            labels.append(tr[i,1])
samples = np.array(samples).astype(np.int32)

xWidth = 203
length_upstream = 60
length_downstream = 140

model_string = 'translation_initiation_sites'
parameterFileName = model_string
# parameter # filenames should be like human_donors.params
if not parameterFileName.endswith('.params'):
    parameterFileName += '.params'

nn, X_input = buildNetwork(parameter_file=parameterFileName,xWidth=xWidth,task_is_translation=True)

test_prediction = L.get_output(nn, deterministic=True)
predict_f = theano.function([X_input], test_prediction)
predictions = []
print('Generating input data')
for i in range(0,len(samples), 1000):
    predictions.append(predict_f(samples[i:i+1000])[:,1])
predictions = np.hstack(predictions)

out = [f'Perf AUC: {str(roc_auc_score(labels, predictions))}',
       f'Perf AUPR: {str(average_precision_score(labels, predictions))}']
np.save('results_out.npy', out)