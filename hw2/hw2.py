
# coding: utf-8

# In[38]:


import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt


# In[39]:


def load_data(file):
    with open(file, 'r') as f:
        y_train = []
        x_train = []
        for line in f:
            line = line.split()
            x_train.append([float(line[i]) for i in range(len(line)-1)])
            y_train.append(float(line[-1]))
    return np.array(x_train), np.array(y_train).reshape(-1,1)


# In[40]:


def rbf(gamma,x,x_):
    row, column = x.shape
    row_, column_ = x_.shape
    #print(row,column, row_, column_)
    K = np.zeros((row, row_))
    for l in range(row_):
        K[:,l] = np.sum((x-x_[l:l+1,:])**2, 1)
    return np.exp(-gamma*K)
    


# In[41]:


x, y = load_data("hw2_lssvm_all.dat.txt")
print(x.shape, y.shape)


# In[42]:


#Q11
x_train, y_train = x[:400], y[:400]
x_test, y_test = x[400:], y[400:]

gamma = [32,2,0.125]
lamb = [0.001,1,1000]

eins = []
eouts = []
for g in gamma:
    K = rbf(g, x_train, x_train)
    K_test = rbf(g, x_train, x_test)
    for l in lamb:
        r,c = K.shape
        B = np.dot(np.linalg.inv(l * np.eye(r) + K), y_train)
        y_pred = np.sign(np.dot(K, B))
        E_in = np.mean(y_train != y_pred)
        eins.append(E_in)
        #print("Gamma: {}, Lambda: {}, E_in:{}".format(g,l,E_in))
        
        y_pred_test = np.sign(np.dot(K_test.T, B))
        E_out = np.mean(y_test != y_pred_test)
        eouts.append(E_out)
        #print("Gamma: {}, Lambda: {}, E_out:{}".format(g,l,E_out))
count = 0
for g in gamma:
    for l in lamb:
        print("Gamma: {}, Lambda: {}, E_in:{}".format(g,l,eins[count]))
        count += 1
count = 0
for g in gamma:
    for l in lamb:
        print("Gamma: {}, Lambda: {}, E_out:{}".format(g,l,eouts[count]))
        count += 1


# In[43]:


#Q13
lamb = [0.01,0.1,1,10,100]
row, column = x_train.shape
row_t, column_t = x_test.shape
print(row, column)
x0 = np.ones((row,1))
x0_test = np.ones((row_t,1))
x_train_lssvm = np.concatenate((x0, x_train),axis=1)
x_test_lssvm = np.concatenate((x0_test, x_test), axis=1)
print(x_train.shape)

eins = []
eouts = []
for l in lamb:
    w = np.linalg.inv(l*np.eye(x_train_lssvm.shape[1])+np.dot(x_train_lssvm.T, x_train_lssvm)).dot(x_train_lssvm.T).dot(y_train)
    y_pred = np.sign(np.dot(x_train_lssvm, w))
    E_in = np.mean(y_pred != y_train)
    eins.append(E_in)
    
    y_pred_test = np.sign(np.dot(x_test_lssvm, w))
    E_out = np.mean(y_pred_test != y_test)
    eouts.append(E_out)
    print(E_in, E_out)
    
for l in range(len(lamb)):
    print("Lambda: {}, Ein: {}".format(lamb[l], eins[l]))

for l in range(len(lamb)):
    print("Lambda: {}, Eout: {}".format(lamb[l], eouts[l]))


# In[46]:


#Q15
lamb = [0.01,0.1,1,10,100]
    

for l in lamb:
    votes = np.zeros((len(y_train),1))
    votes_test = np.zeros((len(y_test),1))
    E_in = 0
    for i in range(250):
        idx = np.random.choice(range(0,400),400)
        w = np.linalg.inv(l*np.eye(x_train_lssvm[idx].shape[1])+np.dot(x_train_lssvm[idx].T, x_train_lssvm[idx])).dot(x_train_lssvm[idx].T).dot(y_train[idx])
        y_pred = np.sign(np.dot(x_train_lssvm[idx], w))
        for _id in range(len(idx)):
            if y_pred[_id] == 1:
                #print('a')
                votes[idx[_id]] += 1
            else:
                #print('b')
                votes[idx[_id]] -= 1
        E_in = np.mean(y_pred != y_train[idx])
        
        eins.append(E_in)

        y_pred_test = np.sign(np.dot(x_test_lssvm, w))
        votes_test += y_pred_test
        E_out = np.mean(y_pred_test != y_test)
        eouts.append(E_out)
        #print(E_in, E_out)
    votes = np.sign(votes)
    votes_test = np.sign(votes_test)
    bagging_ein = np.mean(y_train != votes)
    bagging_eout = np.mean(y_test != votes_test)
    print("lambda:", l, "E_in:", bagging_ein, "E_out:", bagging_eout)
"""
for l in range(len(lamb)):
    print("Lambda: {}, Ein: {}".format(lamb[l], eins[l]))

for l in range(len(lamb)):
    print("Lambda: {}, Eout: {}".format(lamb[l], eouts[l]))
"""


# In[45]:




