#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:05:50 2019

@author: Kartik
"""

import numpy as np
import pandas as pd
import os
import time

#Reading the data in
os.getcwd()
os.chdir('/Users/apple/Documents/ML_Data')
data = pd.read_csv('uci-secom.csv')
print(data.head(5))

#Removing Time and target variable
data.drop(columns = ['Time','Pass/Fail'], inplace = True)
#Replacing all the null values with columns mean
data.fillna(data.mean(), inplace = True)
print('Number of null values in the dataframe: ',sum(data.isna().sum()))

    
    


def PCA(data,threshold):
    '''Takes in a dataset and a specified threshold value to return a low dimensional representation
    of the original dataset containing threshold % variance of the original dataset'''
    start = time.time()
    data2 = np.matrix(data)
    #Compute mean
    mu = data2.mean(axis = 0)
    #center the data
    data2 = data2 - mu
    #compute covariance matrix
    cov = (1/data2.shape[1])*(data2.transpose()*data2)
    #Compute eigenvalues 
    eig_val, eig_vec = np.linalg.eig(cov)
    #Sort eigenvalue, eigenvector pairs
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_val_sort = list(np.flip(np.sort(eig_val)))
    #Number of principal components needed
    for i in range(len(eig_val)):
        check = sum(eig_val_sort[0:i])/sum(eig_val_sort)
        if check >= threshold:
            print('Number of principal components needed: ', i)
            number_pc = i
            break
        else:
            i +=1
    #Ratio of variance explained 
    var_explained = [i[0]/sum(eig_val) for i in eig_pairs[0:number_pc]]
    #Basis of the selected PC's
    basis = [i[1] for i in eig_pairs[0:number_pc]]
    #Basis Matrix of the selected subspace
    reduced_basis_mat = np.vstack((basis[0].reshape(len(eig_val),1)))
    for j in range(1, number_pc):
        reduced_basis_mat = np.hstack((reduced_basis_mat, basis[j].reshape(len(eig_val),1)))
    #Reduced Matrix
    reduced_mat = (reduced_basis_mat.transpose()*data2.transpose()).transpose()
    finish = time.time()
    print("Run Time: ", round(finish - start,2), "seconds")
    return(reduced_mat, var_explained)

    
reduced_mat, var_explained = PCA(data, 0.90)
print(var_explained)
print(sum(var_explained))
print(reduced_mat.shape)


