#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import mode
from scipy.stats import multivariate_normal
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
import cvxpy as cp
from cvxopt import matrix, solvers
from tqdm import tqdm
from scipy import optimize
from numpy import linalg
import cvxopt
# import cvxopt.solvers


# 1. Fit PCA or MDA to the entire data
# 2. Then Split

# #### Implement MDA from Scratch

# In[2]:


#Computing the within group variance - SIgma_W

#Compute MDA

def fit_MDA(data , labels, mda_components):
    
    
    #Scale the data
    #data_scaled = StandardScaler().fit_transform(data)

    #defining the data
    df = pd.DataFrame(data)
    df['labels'] = labels
    cols = list(df.columns)
    
    #val_count
    label_dist = df['labels'].value_counts()

    #Create the dict of Cov within
    sigma_w = np.zeros((data.shape[1],data.shape[1]))

    #Create the dict of Cov betweem
    sigma_b = np.zeros((data.shape[1],data.shape[1]))

    #val_count
    label_dist = df['labels'].value_counts()

    #total N
    N = data.shape[0]

    #Defining the group_mean
    df_agg = df.groupby(['labels']).mean().reset_index()[cols]
    mu0 = np.array(df_agg.drop('labels',axis=1)).mean(axis=0).reshape(-1,1)


    for lab in list(np.unique(labels)):

        #Computing the Within group variance
        df_sub = df[df['labels'] == lab].drop('labels',axis=1)
        cov = np.array(df_sub.cov())
        prior = label_dist[lab]/N * 1.0
        sigma_w = sigma_w +  prior * cov


        #Computing the Between group variance
        mu_cl = np.array(df_agg[df_agg['labels'] == lab].drop('labels',axis=1)).transpose()
        mu_diff = mu_cl - mu0
        sigma_b = sigma_b + np.dot(mu_diff, mu_diff.transpose()) * prior
        

    #Computing pseudo inv
    sigma_w_inv = np.linalg.pinv(sigma_w)
    mda_mat = np.dot(sigma_w_inv,sigma_b)

    #Eigen value deocomposition
    eigen_values , eigen_vectors = LA.eigh(mda_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    eigenvalues_sorted = eigen_values[sorted_index]
    eigenvectors_sorted = eigen_vectors[:,sorted_index]

    #subsetting the mda componnets
    eigenvectors_subset = eigenvectors_sorted[:,0:mda_components]

    #getting the variance explained
    exp_var = sum(eigenvalues_sorted[0:mda_components])/ sum(eigen_values) * 100.0

    # #final pc components
    data_reduced = np.dot(eigenvectors_subset.transpose() , data.transpose()).transpose()
    
    return data_reduced


# #### Implement PCA from scratch

# In[3]:


#PCA IMplementation

def fit_PCA(data , pca_components):

    #Scale
    data_scaled = StandardScaler(with_std=False).fit_transform(data)

    #Covariance compute
    features_data = data_scaled.transpose()
    cov_matrix = np.cov(features_data)

    #Eiegen value decomp
    eigen_values , eigen_vectors = np.linalg.eigh(cov_matrix)

    #Subset the eigen vec
    sorted_index = np.argsort(eigen_values)[::-1]
    eigenvalues_sorted = eigen_values[sorted_index]
    eigenvectors_sorted = eigen_vectors[:,sorted_index]

    #sort the top k comp
    eigenvectors_subset = eigenvectors_sorted[:,0:pca_components]
    

    #getting the variance explained
    exp_var = sum(eigenvalues_sorted[0:pca_components])/ sum(eigen_values) * 100.0


    #final pc components
    data_reduced = np.dot(eigenvectors_subset.transpose() , data_scaled.transpose()).transpose()

    return data_reduced, exp_var


# #### Dataset Preparation

# In[4]:


def data_preprocessing(inp_path,seed,task,transform,components):
    
    #load the data
    data = scipy.io.loadmat(inp_path)
    temp = data['face']
    face_data = temp.reshape(-1,temp.shape[2]).transpose()
    
    
    #create the labels --- sub
    if task == 'subject' :
        labels = np.arange(1,201).repeat(3)
        face_data1 = face_data
        labels1 = labels
        
    
    elif task == 'expression' :
        #'neutral = 0','facial =1','illumination=2'
        labels = np.array([0,1,2]*200)
        
        #removing the illumination class
        face_data1 = face_data[labels !=2]
        labels1 = labels[labels !=2]
        
    
    
    #check for transform
    
    #if PCA
    if transform == 'PCA':
        data_reduced, exp_var = fit_PCA(face_data1 , components)
        new_data = data_reduced
        

    #if MDA
    elif transform == 'MDA':
        #use labels
        #data_reduced, exp_var = fit_PCA(data , pca_components)
        new_data = fit_MDA(face_data1 , labels1, components)
    
    #If no transformatiom
    else :
        new_data = face_data1
    
    
    if task == 'subject' :
        
        #splitting the train and test data
        x_train, x_test, y_train, y_test = train_test_split(new_data, labels1, 
                                                            test_size = 200,
                                                            stratify = labels1,
                                                            random_state=seed)   
    else :
        
        
        #splitting the train and test data
        x_train, x_test, y_train, y_test = train_test_split(new_data, labels1, 
                                                            test_size = 0.2,
                                                            stratify = labels1,
                                                            random_state=seed)   
        
    
    
    #return the test and train data
    return x_train, x_test, y_train, y_test


# In[5]:


components  = 2
transform = 'MDA'
x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                       seed=1,
                                                      task = 'expression',
                                                      transform = transform,
                                                      components = components)

df = pd.DataFrame(x_train)
df.columns = ['X1','X2']
df['labels'] = y_train

import seaborn as sns
fig = plt.figure()
sns.scatterplot(data=df, x="X1", y="X2", hue="labels")
plt.title('MDA_Visualization')
fig.savefig('MDA_Visualization.png')


# In[7]:


components  = 2
transform = 'PCA'
x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                       seed=1,
                                                      task = 'expression',
                                                      transform = transform,
                                                      components = components)

df = pd.DataFrame(x_train)
df.columns = ['X1','X2']
df['labels'] = y_train

import seaborn as sns
fig = plt.figure()
sns.scatterplot(data=df, x="X1", y="X2", hue="labels")
plt.title('PCA_Visualization')
fig.savefig('PCA_Visualization.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




