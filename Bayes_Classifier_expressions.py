#!/usr/bin/env python
# coding: utf-8

# In[9]:


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

# In[10]:


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

# In[11]:


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

# In[12]:


def data_preprocessing(inp_path,seed,task,transform,components):
    
    #load the data
    data = scipy.io.loadmat(inp_path)
    temp = data['face']
    face_data = temp.reshape(-1,temp.shape[2]).transpose()
    
    
    #create the labels --- sub
    if task == 'subject' :
        labels = np.arange(0,201).repeat(3)
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
        
#     if transform == 'PCA_sklearn':
#         pca = PCA(n_components=components)
#         new_data = pca.fit_transform(face_data)
        

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


# #### Bayes Classifier

# In[13]:


def fit_bayes_classifier(x_train, y_train):
    
    
    #defining the variance-cov matrix dictionary
    mean_vec = {}
    cov_mat = {}
    
    #defining no of classes
    labels = np.unique(y_train)

    for lab in list(np.unique(labels)):
        
        #subsetting the data on labels
        X_class = x_train[y_train==lab]

        mu_class = np.mean(X_class, axis=0)
        
        #saving the mean _vec dict
        mean_vec[lab] = mu_class
        
        #saving the cov matrix dict
        cov_class = np.cov(X_class, rowvar=False)
        cov_mat[lab] = cov_class
    
    
    #return the two dictionaries
    return mean_vec, cov_mat
        


# In[14]:


def predict_bayes_classifier(x_train, y_train, x, task,lamda=0.01):
    
    
    #getting the dict
    mean_vec, cov_mat = fit_bayes_classifier(x_train, y_train)
    
    #defining the prob
    labels = np.unique(y_train)
    
    #defining the prior
    prior_dict = Counter(y_train)
    
    #prob_dict
    prob_vec = []
    
    #regularization matrix
    if len(cov_mat[labels[0]].shape)> 0 :
        size = cov_mat[labels[0]].shape[0]
        
    else :
        size = 1
    
    reg_mat = lamda * np.eye(size,size)
    
    
    #Making the prediction
    for lab in list(np.unique(labels)):

        #define the gaussian
        mvn = multivariate_normal(mean= mean_vec[lab], cov=cov_mat[lab] + reg_mat)
        
        
        # We use uniform priors
        prior = prior_dict[lab]/len(y_train)
        
        #prior prob
        prob_val = prior*mvn.pdf(x)

        prob_vec.append(prob_val)
        

    return np.array(prob_vec)


# In[15]:


def get_bayes_score(x_train, y_train, x_test, y_test,task,lamda=0.01):
    
    #get the prediction
    y_pred = []

    for x in tqdm(x_test) : 
        prob = predict_bayes_classifier(x_train, y_train, x, task,lamda=0.01)
        pred = np.argmax(prob)

        y_pred.append(pred)

    
    #test score
    score = accuracy_score(y_test, y_pred)
    
    #return predictions and test score
    return y_pred, score


# #### Sample Experiments

components = 20
transform = 'PCA'


x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                       seed=0,
                                                      task='expression',
                                                      transform = transform,
                                                      components = components)




y_pred, score_te = get_bayes_score(x_train, y_train, x_test, y_test,'expression',lamda=0.1)




print ('The test accuracy is ' + str(score_te) + ' for ' + str(components) + ' components ' +  str(transform) )




y_pred, score_tr = get_bayes_score(x_train, y_train, x_train, y_train,'expression',lamda=0.1)




print ('The train accuracy is ' + str(score_tr) + ' for ' + str(components) + ' components ' +  str(transform) )




# change the transform to if you need to see for MDA
transform = 'PCA'


# In[84]:


bayes_scores_te = []
list_te = [10,20,30,40,50,80,100,150,200]

for components in tqdm(list_te) :
    
    #subset the data
    x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                           seed=0,
                                                           task='expression',
                                                           transform = transform,
                                                           components = components)

    y_pred, score = get_bayes_score(x_train, y_train, x_test, y_test,'expression',lamda=0.1)
    bayes_scores_te.append(score)


# In[85]:


# test_scores[7]
print ('The max Test accuracy score is ' + str(max(bayes_scores_te)))

print ('The optimal PC component: ' + str(list_te[bayes_scores_te.index(max(bayes_scores_te))]))


# In[92]:


fig = plt.figure()
plt.plot(list_te,bayes_scores_te)
plt.xlabel('No of Principal Components')
plt.ylabel('Bayes Test Accuracy')
plt.title ('Bayes Classifier Expressions Data')
fig.savefig('Bayes Classifier Test accuracy.png')


# In[ ]:




