# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:02:33 2020

@author: JF LIU
"""
#%% data import
import torch

import numpy as np
import random 
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from torchtext import data 
#from torchtext import datasets

import os
os.chdir("C:/Users/JF LIU/Desktop/DLproject/para")
# BATCH_SIZE = 64
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
#     (train_data, valid_data, test_data), 
#     batch_size = BATCH_SIZE, 
#     device = device)

#model

class nlp_cnn(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)

        # print(embedded.shape)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


INPUT_DIM =  20574 #len(TEXT.vocab)
EMBEDDING_DIM = 25
N_FILTERS = 25
FILTER_SIZES = [3,4]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = 0#TEXT.vocab.stoi[TEXT.pad_token]

# model = nlp_cnn(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

# print(model)
# pretrained_embeddings = TEXT.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)


# UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

# model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
# model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model_2 = nlp_cnn(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

model_2.load_state_dict(torch.load('Model_1_State_Dict'))

params = []
for name, param in model_2.named_parameters():
  if param.requires_grad:
    params.append((name, param.data.numpy()))
#print(params)

X = torch.load("Model_1_Intermediates_batch_0_input_embed")
X.shape
X = np.array(X)

cnn_output = torch.load("Model_1_Intermediates_batch_0_fc")
cnn_output = np.array(cnn_output)

weight1 = params[1][1].squeeze(1)
bias1 = params[2][1]
weight2 = params[3][1].squeeze(1)
bias2 = params[4][1]
weight_fc = params[5][1]
bias_fc = params[6][1]
#%% MIP 
import gurobipy as gp
from gurobipy import GRB
import numpy as np


#X = X0
#W = weight1[0]
#b = bias1[0]
#cnn_output = cnn_output[0]

def minimize_l2(X, W1, b1, W2, b2, weight_fc, bias_fc, cnn_output, K=3, kernel_size=(3,25)):
    global m
    weight1=W1 
    bias1=b1
    weight2=W2
    bias2=b2
    """
    Parameters
    ----------
    X : TYPE 2-dimension( height, channels=100)
        node values in cnn 
    W : TYPE 3-dimension(out_channels, kernel[0], kernel[1])
        weights in cnn
    b : TYPE 1-dimension(channels, )
        bias in cnn
    K : # of layers
    kernel_size : a tuple (3,100) for example, the same as the size in CNN 
    objective function: minimize manhattan distance between original input X
                        and new input Y                   
    variables: new input Y
    """
    # code for one batch one channel
    #m = gp.Model()
    #y = m.addVars(64,X.shape[1],100, name="y")
    y = m.addVars(X.shape[0], X.shape[1], lb=-GRB.INFINITY, name="y")
    output = m.addVar(lb=-GRB.INFINITY, name="output")
    tmp = m.addVar(lb=-GRB.INFINITY, name="tmp")
    m.update()
    lb = X*(1-0.05)
    ub = X*(1+0.05)
    d = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v1 = lb[i,j]
            v2 = ub[i,j]
            m.addConstr( y[i,j] >= v1 )
            m.addConstr( y[i,j] <= v2 )
            d = d + (y[i,j] - X[i,j]) * (y[i,j] - X[i,j])
    dist = m.addVar()
    m.update()
    m.setObjective( dist , GRB.MINIMIZE)
    m.addConstr(dist == d)
    m.update()
    m.setParam("NonConvex", 2)    
    #m.setObjective( L2(y, X) , GRB.MINIMIZE)
    #m.addGenConstrMax(z[0,2], [y[0,2,1], 0.0], name="maxconstr")
    def conv2vec(y, X, weight, bias, w_fc, b_fc, kernel_size):
        """
        x is the input with size (batch, height, channels=100)
        y : the gurobi variables
        weight_c, bias_c : weight, bias for each channel
        """
        global m
        nonlocal tmp, output
        z = m.addVars(X.shape[0], lb=-GRB.INFINITY, vtype=GRB.BINARY, name="z")
        a = m.addVars(X.shape[0], lb=0.0, name="a")
        c = m.addVars(X.shape[0], lb=0.0, name="c")
        #batch_size = X.shape[0]
        #kernel_size = (3,50)
        length = X.shape[0] - kernel_size[0] + 1 
        k_size = kernel_size[0]
        bias_c = bias
        #bias_c = 0.23
        weight_c = weight
        #vec = np.zeros((batch_size, length, 1))
        #dist = {}
        for j in range(length):
            # elementwise multipilication for convolutional layer
            tp1=bias_c
            for k in range(k_size):
                for n in range(kernel_size[1]):
                    tp1 = tp1 + weight_c[k,n] * y[j+k,n]  
                    #print(tp1)
                    #m.addGenConstrMax(z[i,j], [tp1, 0.0])
                    #print(tp1)
                    #a[0,1243]
            m.addConstr(a[j]-c[j] ==  tp1 )
            m.addGenConstrIndicator(z[j], True, a[j] <= 0.0)
            m.addGenConstrIndicator(z[j], False, c[j] <= 0.0)           
        m.update()
        u=m.addVar(name="u")
        tmp = tmp + w_fc * u 
        m.addGenConstrMax(u, [a[i] for i in range(X.shape[0])])
        m.update()    
        #m.feasRelax()
    for i in range(weight1.shape[0]):
        conv2vec(y, X, weight1[i], bias1[i], weight_fc[0,i], bias_fc, kernel_size)
    for i in range(weight2.shape[0]):
        conv2vec(y, X, weight2[i], bias2[i], weight_fc[0,25+i], bias_fc, kernel_size = (4,25))
    tmp = tmp + bias_fc
    print(tmp)
    m.addConstr(output == tmp)
    m.addConstr( tmp * cnn_output <= 0.0 )
    m.update()
    m.optimize()
    if m.status == GRB.INFEASIBLE:
        m.feasRelaxS(1, False, False, True)
        m.optimize()
    vals = m.getVars() 
    return (vals)
#%%
# time 1h4m
tic = time.time()

def extract_y(new_example, X):
    dim1 = X.shape[0]
    dim2 = X.shape[1]
    yvalue = [0] * (dim1*dim2)
    for i in range(dim1*dim2):
        yvalue[i] = new_example[i].X
    yvalue = np.reshape(yvalue, (dim1*dim2))
    return (yvalue)    
new_y = np.zeros((64,25,25))   
new_output = np.zeros((64,1))  
for i in range(64):
    example = X[i,:,:]
    m = gp.Model()
    new_example = minimize_l2(example, weight1, bias1, weight2, bias2, weight_fc, \
                          bias_fc[0], cnn_output[i][0],\
                          K=3, kernel_size=(3,25))
    new_y[i,:,:] = extract_y(new_example, example).reshape((25,25))
    new_output[i,:] = new_example[626].X

toc = time.time() - tic
print("*************time**************")
print(toc)
#%%
m = gp.Model()
#m.setParam("TimeLimit", 1200)
X0 = X[0,:,:]
new_example0 = minimize_l2(X0, weight1, bias1, weight2, bias2, weight_fc, \
                          bias_fc[0], cnn_output[0][0],\
                          K=3, kernel_size=(3,25))
#%%
X4 = X[4,:,:]
m = gp.Model()
new_example4 = minimize_l2(X4, weight1, bias1, weight2, bias2, weight_fc, \
                          bias_fc[0], cnn_output[4][0],\
                          K=3, kernel_size=(3,25))      
#%%
new_example = new_example0
temp = np.zeros((1,50))
num=0
for i in range(7931): 
    if new_example[i].VarName[0] == "u":
        temp[0,num] =  new_example[i].X
        num = num + 1
np.sum(temp * weight_fc) + bias_fc
new_example0[626]
