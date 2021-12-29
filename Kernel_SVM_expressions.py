#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# import cvxopt.solvers


# 1. Fit PCA or MDA to the entire data
# 2. Then Split

# #### Implement MDA from Scratch

# In[ ]:


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

# In[ ]:


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

# In[ ]:


def data_preprocessing(inp_path,seed,task,transform,components):
    
    #load the data
    data = scipy.io.loadmat(inp_path)
    temp = data['face']
    face_data = temp.reshape(-1,temp.shape[2]).transpose()
    
    
    #create the labels --- sub
    if task == 'subject' :
        labels = np.arange(0,200).repeat(3)
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


# In[ ]:





# #### SVM

# In[ ]:


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, p=3):
    return (1 + np.dot(x1, x2)) ** p

def gaussian_kernel(x1, x2, sigma= 1):
    return np.exp(-linalg.norm(x1 - x2)**2 / ( (sigma ** 2)))


# In[ ]:



class SVM(object):
    
    def __init__(self, kernel):
        self.kernel = kernel
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
        
        #Making it symmetric
        #K = np.tril(K) + np.triu(K.T, 1)
        
        
        #Formulating the opt problem
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt_matrix(-np.ones((n_samples, 1)))
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))
        G = cvxopt_matrix(np.vstack((np.eye(n_samples)*-1,np.eye(n_samples))))
        h = cvxopt_matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples))))


        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b, maxiters=2)
        
        
        #Setting solver parameters (change default to decrease tolerance)
        cvxopt_solvers.options['show_progress'] = False

        #Lagrange multipliers
        a = np.ravel(solution['x'])
        

        # Support vectors have non zero lagrange multipliers
        
        if self.kernel == polynomial_kernel :
            sv = a > 1e-20
        else :
            sv = a > 1e-5
            
        #print (sv)
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        #print('{0} support vectors out of {1} points'.format(len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        #print (len(self.a))
        self.b /= len(self.a) 

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
    
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


# In[ ]:


def get_svm_data(x_train,y_train) :
    
    X = x_train
    Y = y_train
    y = np.array([-1 if i==0 else 1 for i in Y])
    
    return X,y


# #### Experiments with single train-test split

# In[ ]:


x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                       seed=10,
                                                       task='expression',
                                                       transform = 'MDA',
                                                       components = 1)


# In[ ]:


#Process data
X,y = get_svm_data(x_train,y_train)

# # #Predict
svm = SVM(kernel = gaussian_kernel)
svm.fit(X, y.ravel().astype(float))


pred = svm.predict(x_test)

y_pred = np.array([0 if i==-1 else 1 for i in pred])
score_te = accuracy_score(y_test, y_pred)


# In[ ]:


print ('The optimal test accuracy is ' + str(score_te*100))


# In[ ]:





# #### Experiments with Cross Validation

# In[ ]:


#Select the no of components and transformation

components = 2
transform = 'MDA'      ### either 'MDA' or 'PCA'
kernel_type = 'rbf'    ### either 'rbf' or 'poly'


# In[ ]:


# #Implement Cross-Validation


#Save the cross-val score
dict_score = {}
    

for seed in tqdm(np.arange(1, 10)) : 
    

    #Choose the data
    x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                       seed=seed,
                                                       task='expression',
                                                       transform = transform,
                                                       components = components)
    
    
    #input the data
    X = x_train
    Y = y_train
    
    #convert the data in SVM format
    y = np.array([-1 if i==0 else 1 for i in Y])
    
    #define the list
    acc_score = []
    
    
    #Predict
    for parameter in np.arange(1,20,2) :
        
        if kernel_type == 'rbf':

            def gaussian_kernel(x1, x2, sigma= parameter):
                return np.exp(-linalg.norm(x1 - x2)**2 / ( (sigma ** 2)))
            kernel = gaussian_kernel
           
            
        elif kernel_type == 'poly':
            def polynomial_kernel(x1, x2, p = parameter ):
                return (1 + np.dot(x1, x2)) ** p 
            kernel = polynomial_kernel
        
       
        #define SVM
        svm = SVM(kernel = kernel)
        svm.fit(X, y.ravel().astype(float))
        
        #Predict using SVM
        pred = svm.predict(x_test)
        #pred = svm.predict(X)
        
        
        #make it in correct format
        y_pred = np.array([0 if i==-1 else 1 for i in pred])
        
        #compute accuracy
        acc_score.append(accuracy_score(y_test, y_pred))
        #acc_score.append(accuracy_score(y, pred))
    
    
    #Get the dataset's pred
    dict_score[seed] = acc_score


# In[ ]:


#returns the dictionary of 
dict_score


# In[ ]:


#Get the average value
val_arr = np.zeros((len(dict_score[1])))
for key in list(dict_score.keys()) :
    val_arr = val_arr + np.array(dict_score[key])
#get the avg cv vals
val_arr1 = val_arr/len(list(dict_score.keys()))

#plot
fig = plt.figure()
plt.plot(np.arange(1,20,2), val_arr1)
plt.xlabel('Different values of parameter')
plt.ylabel('Avg Cross Validation Accuracy Score')
plt.title('Kernel SVM ' + str(kernel_type) + ' ' + str(components) + str(transform) + ' components')
fig.savefig('Kernel SVM Classifier CV Test accuracy.png')


# In[ ]:





# In[ ]:





# In[ ]:




