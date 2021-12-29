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


# In[ ]:





# #### KNN Implementation

# In[5]:


#KNN

#function to predict
def predict_knn(x_train, y_train , x_test, y_test,k):

    #test_predictions
    y_pred = []

    
    #Loop through the Test data
    for test_data in x_test: 

        #Array to store distances
        distance_matrix = []

        #Loop through each training Data
        for i in range(0,len(x_train)): 

            #compute distances
            distances = np.linalg.norm(x_train[i]- test_data)

            #Calculating the distance
            distance_matrix.append(distances)

        #converting to an array
        dist_mat_arr = np.array(distance_matrix) 

        #Sorting the array while preserving the index
        sorted_distance = np.argsort(dist_mat_arr)[:k] 
        
        #Labels of the K datapoints from above
        labels = y_train[sorted_distance]

        #Majority voting predictions
        pred = mode(labels).mode[0]
        y_pred.append(pred)
    

    
    #compute the score for test data
    score = accuracy_score(y_test, y_pred)
    
    
    return y_pred, score


# #### Sample Experiments

# In[12]:


components = 2
transform = 'MDA'


# In[13]:


x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                       seed=0,
                                                      task='expression',
                                                      transform = transform,
                                                      components = components)


# In[15]:


#Compute Multiple
scores = []
for k in tqdm(range(1,50)) :
    
    y_pred, score = predict_knn(x_train, y_train , x_test, y_test, k= k)
    scores.append(score)


# In[17]:


fig = plt.figure()
plt.plot(scores)
plt.xlabel('No of nearest neighbours,k')
plt.ylabel('Test accuracy')
plt.title('KNN with ' + str(transform) + ' ' + str(components) + ' components')
fig.savefig('knn_test_accuracy.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




