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
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from cvxopt import solvers as cvxopt_solvers
import PIL  

# import cvxopt.solvers


# 1. Fit PCA or MDA to the entire data
# 2. Then Split

# In[2]:


# data = scipy.io.loadmat('data.mat')
# face_data = data['face']
# face_data = face_data.reshape(-1,face_data.shape[2]).transpose()
# labels = list(np.arange(1,201).repeat(3))
# len(labels)


# #### Implement MDA from Scratch

# In[3]:


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

# In[4]:


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

# In[5]:


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

# In[6]:



def linear_kernel(x1, x2):
    return np.dot(x1, x2)


# In[7]:



class Lin_SVM(object):
    
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
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))


        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b, maxiters=2)
        cvxopt_solvers.options['show_progress'] = False
        
        #Setting solver parameters (change default to decrease tolerance) 
#         cvxopt_solvers.options['abstol'] = 1e-10
#         cvxopt_solvers.options['reltol'] = 1e-10
#         cvxopt_solvers.options['feastol'] = 1e-10
        cvxopt_solvers.options['show_progress'] = False

        #Lagrange multipliers
        a = np.ravel(solution['x'])
        

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-20
        
#         if self.kernel == polynomial_kernel :
#             sv = a > 1e-20
#         else :
#             sv = a > 1e-5
            
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


cvxopt_solvers.options['show_progress'] = False


# In[8]:


# from sklearn.svm import SVC
# clf = SVC(kernel = 'rbf', gamma=0.11)
# clf.fit(X, y.ravel()) 

# pred = clf.predict(x_test)
# y_pred = np.array([0 if i==-1 else 1 for i in pred])
# accuracy_score(y_test, y_pred)


# In[ ]:





# #### Boosted SVM

# In[9]:


#Computes weighted error

def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)


# In[10]:


#Weight to combine weak classifiers for the final classifier

def compute_alpha(error):
    return np.log((1 - error) / error + 1e-8)


# In[11]:


#update the weights based on the errors

def update_weights(w_i, alpha, y, y_pred):
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))


# In[12]:


#sampling the datapoints based on weighted method

def get_weighted_sampling(X,y,w_i):
    
    #dataframe
    df = pd.DataFrame(X)
    df['labels'] = y
    df['freq'] = w_i
    
    #sample
    df_sampled = df.sample(weights = df.freq, frac = 1, random_state =0)
    
    #return
    y_new = df_sampled['labels'].values
    X_new = df_sampled.drop(['labels','freq'], axis=1)
    X_new = np.array(X_new)
    
    return X_new, y_new
    


# In[13]:


# Fit model
# ab = Boosted_SVM()
# ab.fit(X, y, M = 10)


# In[14]:


# Define AdaBoost class
class SVMBoost:
    
    
    def __init__(self):
        
        #Initializing the parameters
        self.alphas = []
        self.Boost_Model = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, M):
        
        '''
        X: feature variables
        y: response variable
        M: number of boosting
        '''
        
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.M = M

        # Iterate over M weak classifiers
        for m in tqdm(range(0, M)):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred)
                
            
            
            #Sample X based on weighted distribution
            X, y = get_weighted_sampling(X,y,w_i)
            
            
            #Fit weak SVM and predict labels
            svm_model = Lin_SVM(kernel = linear_kernel)  
            svm_model.fit(X, y.ravel().astype(float))
            y_pred = svm_model.predict(X)
            
            self.Boost_Model.append(svm_model) 

            #Compute error based on model
            error_m = compute_error(y, y_pred, w_i)
            
           
            self.training_errors.append(error_m)
            
            # (c) Compute alpha
            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)
            

        assert len(self.Boost_Model) == len(self.alphas)


    def predict(self, X):

        # Initialise with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.Boost_Model[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # Estimate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
      
    def error_rates(self, X, y):
        
        self.prediction_errors = [] 
        
        # Predict class label for each weak classifier
        for m in range(self.M):
            y_pred_m = self.Boost_Model[m].predict(X)          
            error_m = compute_error(y = y, y_pred = y_pred_m, w_i = np.ones(len(y)))
            self.prediction_errors.append(error_m)


# In[15]:


def get_svm_data(x_train,y_train) :
    
    X = x_train
    Y = y_train
    y = np.array([-1 if i==0 else 1 for i in Y])
    
    return X,y


# #### Split the Data

# In[34]:


components = 2
transform = 'MDA'


# In[35]:


x_train, x_test, y_train, y_test = data_preprocessing('data.mat',
                                                       seed=2,
                                                       task='expression',
                                                       transform = 'MDA',
                                                       components = components)


# In[36]:


#Get the data in SVM format


# In[37]:


# Fit Boosting model
def get_boosted_score(M, x_train, x_test, y_train, y_test):
    
    #Data format
    X,y = get_svm_data(x_train,y_train)
    
    #fit model
    ab = SVMBoost()
    ab.fit(X, y, M = M)

    #Get prediction
    y_pred_ad = ab.predict(x_test)

    #convert in proper format
    y_pred = np.array([0 if i==-1 else 1 for i in y_pred_ad])
    return accuracy_score(y_test, y_pred)


# In[38]:


#get_boosted_score(70, x_train, x_test, y_train, y_test)


# In[39]:


list_iters = [2, 5, 10, 15, 20, 25, 30]


# In[40]:


#Append scores
test_scores = []

for mods in list_iters :
    
    test_scores.append(get_boosted_score(mods, x_train, x_test, y_train, y_test))


# In[41]:


# test_scores[7]
print ('The max Test accuracy score is ' + str(max(test_scores)))

print ('The number of models required: ' + str(list_iters[test_scores.index(max(test_scores))]))


# In[42]:


fig = plt.figure()
plt.plot(list_iters,test_scores)
plt.xlabel('No of Weak Classifiers (SVMs)')
plt.ylabel('Boosted Test Accuracy')
plt.title ('Boosted Linear SVM with ' + str(transform) + ' ' + str(components) + ' components')
fig.savefig('boosted_svm.png')


# In[ ]:





# In[ ]:




