
# coding: utf-8

# In[1]:


import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm


# In[2]:


def load_data():
    y_train = []
    x_train = []
    x_test = []
    y_test = []
    with open("features.train.txt", "r") as f:
        for data in f:
            data = data.split()
            y_train.append(float(data[0]))
            x_train.append([float(data[1]), float(data[2])])
            
    with open("features.test.txt", "r") as f:
        for data in f:
            data = data.split()
            y_test.append(float(data[0]))
            x_test.append([float(data[1]), float(data[2])])
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)      



# In[3]:


x_train, x_test, y_, y_t = load_data()


# In[4]:



row = x_train.shape[0]
y_train = []
for y in y_:
    if y == 0.0:
        y_train.append(1)
    else:
        y_train.append(-1)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)
C = [1e-5,1e-3, 1e-1, 10, 1000]
w = []
for c in C:
    clf = svm.SVC(C = c, kernel = "linear")
    clf.fit(x_train, y_train)
    print("C:",c, "weight:", clf.coef_, "weight_norm", np.linalg.norm(clf.coef_))
    w.append(np.linalg.norm(clf.coef_))

plt.xlabel("log(C)")
plt.ylabel("w_norm")
plt.xticks([-5,-3,-1,1,3])
plt.plot([-5,-3,-1,1,3], w)
plt.savefig("11.png")
plt.show()


# In[5]:


y_train = []
for y in y_:
    if y == 8.0:
        y_train.append(1)
    else:
        y_train.append(-1)
E_ins = []
sum_sv = []
for c in C:
    clf = svm.SVC(C=c, kernel="poly", degree = 2, gamma = 1, coef0 = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    E_in = np.mean(y_pred != y_train)
    print("C:",c, "Ein:", E_in, "num_sv:", sum(clf.n_support_))
    E_ins.append(E_in)
    sum_sv.append(sum(clf.n_support_))
plt.xlabel("log(C)")
plt.ylabel("E_in")
plt.xticks([-5,-3,-1,1,3])
plt.plot([-5,-3,-1,1,3], E_ins)
plt.savefig("12.png")
plt.show()
        
plt.xlabel("log(C)")
plt.ylabel("# of support vectors")
plt.xticks([-5,-3,-1,1,3])
plt.plot([-5,-3,-1,1,3], sum_sv)
plt.savefig("13.png")
plt.show()


# In[6]:


y_train = []
for y in y_:
    if y == 0.0:
        y_train.append(1)
    else:
        y_train.append(-1)
y_train = np.array(y_train)
C = [1e-3,1e-2, 1e-1, 1, 10]
w = []
for c in C:
    clf = svm.SVC(C = c, kernel = "rbf", gamma = 80)
    clf.fit(x_train, y_train)
    weight = np.dot(clf.dual_coef_,clf.support_vectors_)
    print("C:", c, "Weight:", weight, "Norm:", 1/np.linalg.norm(weight))
    w.append(1/np.linalg.norm(weight))
    
plt.xlabel("log(C)")
plt.ylabel("Distance")
plt.xticks([0,1,2,3,4])
plt.plot([0,1,2,3,4], w)
plt.savefig("14.png")
plt.show()


# In[7]:


y_train = []
y_test = []
for y in y_t:
    if y == 0.0:
        y_test.append(1)
    else:
        y_test.append(-1)
for y in y_:
    if y == 0.0:
        y_train.append(1)
    else:
        y_train.append(-1)
y_train = np.array(y_train)
C = [1e-3,1e-2, 1e-1, 1, 10]
gammas = [1, 10, 100, 1000, 10000]
w = []
E_outs = []

for r in gammas:
    clf = svm.SVC(C=0.1, kernel = "rbf", gamma = r)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    E_out = np.mean(y_pred != y_test)
    E_outs.append(E_out)
    print("Gamma:", r, "E_out:", E_out)


    
plt.xlabel("log(r)")
plt.ylabel("E_out")
plt.xticks([0,1,2,3,4])
plt.plot([0,1,2,3,4], E_outs)
plt.savefig("15.png")
plt.show()


# In[21]:


y_train = []
for y in y_:
    if y == 0.0:
        y_train.append(1)
    else:
        y_train.append(-1)
y_train = np.array(y_train)
nums = [0,0,0,0,0]
gammas = [1e-1,1,10,100,1000]
for i in range(100):
    idx = np.random.permutation(row)
    x_val = x_train[idx[:1000],:]
    y_val = y_train[idx[:1000]]
    x_tr = x_train[idx[1000:],:]
    y_tr = y_train[idx[1000:]]
    val = 10
    best_r = -2
    #print(i)
    for r in range(len(gammas)):
        clf = svm.SVC(C = 0.1, kernel = "rbf", gamma = gammas[r])
        clf.fit(x_tr, y_tr)
        y_pred = clf.predict(x_val)
        E_val = np.mean(y_pred != y_val)
        #print(E_val)
        if val > E_val:
            val = E_val
            best_r = r
    #print(best_r)
    nums[best_r] += 1
a = []
for i in range(len(nums)):
    for j in range(nums[i]):
        a.append(i)
plt.hist(a, bins = [-1.5,-0.5,0.5,1.5,2.5,3.5])
plt.xticks([-1,0,1,2,3,4])
plt.xlabel("Gamma")
plt.ylabel("Number")
plt.savefig("16.png")
plt.show()

