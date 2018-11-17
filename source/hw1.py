# -*- coding: utf-8 -*-
"""
Politecnico di Torino - Machine learning and Artificial intelligence

Homework 1

@author: Valerio Paolicelli (matr. 253054)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def normalize(data):
    ''' 
    Normalize all features of each sample (image): ( x - mean)
                                                   -----------
                                                      std
    Receive the dataset, performe the mean, the std and so the normalization
    Return the tuple (data normalized, all computed means, all computed stds)
    '''
    i=0
    means= []
    devs= []
    result= []
    for entry in data: #loop to standardize (mean 0 - std 1)
        image= np.array(entry).copy()
        means.append(np.mean(image))
        devs.append(np.std(image))
        image_std= (image - means[i]) / devs[i]
        result.append(image_std)
        i += 1
    return np.array(result), np.array(means), np.array(devs)

def normalizeEig(data, means, devs):
    ''' 
    Normalize all features of each sample (image): ( x - mean)
                                                   -----------
                                                      std
    Receive the dataset, performe the mean, the std and so the normalization
    Return the tuple (data normalized, all computed means, all computed stds)
    '''
    i=0
    result= []
    for entry in data: #loop to standardize (mean 0 - std 1)
        image= np.array(entry).copy()
        image_std= (image - means[i]) / devs[i]
        result.append(image_std)
        i += 1
    return np.array(result)


def unnormalize(data, means, devs):
    '''
    Unnormalize all features of each sample (image): (x * std) + mean
    Receive the dataset (normalized), the set of means and stds corresponding to each 
        sample
    Return the unnormalized dataset
    '''
    
    dataset= np.array(data).copy()
    result= []
    i=0
    for entry in dataset:
        result.append((entry*devs[i])+means[i])
        i+=1
    return np.array(np.uint8(result))

def doPCA(x, means, devs, n_comp, img_to_print, flag, name):
    '''
    Receive the dataset standardized, with its own means and stds,
        the image to plot after the PCA algorithm and a flag.
    The flag is usefull (flag= 0) to print the image after PCA, 
        or (flag= 1) only to return the re-projected data onto the new base.
    Return something if and only if the flag is setted to 1.
    '''
    pca= PCA(n_components= n_comp)
    x_t= pca.fit_transform(x)
    x_t_i= pca.inverse_transform(x_t)
    x_t_i_un= unnormalize(x_t_i, means, devs)
    if flag == 0:
        plt.xlabel(str(n_comp)+'PCs')
        plt.imshow(np.reshape(x_t_i_un[img_to_print],(227,227,3)))
        plt.savefig('images/' + name + str(n_comp) + '.jpg')
        plt.show()
    else:
        return x_t_i_un

def doPCAManually(x, means, devs, comp_from, comp_to, img_to_print, flag, name):
    '''
    Receive the dataset standardized, with its own means and stds,
        the image to plot after the PCA algorithm and a flag.
    The flag is usefull (flag= 0) to print the image after PCA, 
        or (flag= 1) only to return the re-projected data onto the new base
    Return something if and only if the flag is setted to 1.
    '''
    
    pca= PCA()
    pca.fit(x)
    if comp_from > 0 and comp_to > 0:
        string= 'From the ' + str(comp_from) + 'PC to the ' + str(comp_to) + 'PC'
        comp_from-=1
        n_eigenvect= pca.components_[comp_from:comp_to,:]
    elif comp_from < 0 and comp_to == 0:
        string= 'Last ' + str(comp_from*-1) + 'PCs'
        n_eigenvect= pca.components_[comp_from:,:]
        
    # transform
    x_trasformed= np.dot(np.array(x), np.array(n_eigenvect).T)
    
    #inverse -> re-project
    x_restored= np.dot(x_trasformed, np.array(n_eigenvect))
    
    x_t_i_un= unnormalize(np.array(x_restored), means, devs)
            
    if flag == 0:
        plt.xlabel(string)
        plt.imshow(np.reshape(x_t_i_un[img_to_print],(227,227,3)))
        if comp_from < 0:
            plt.savefig('images/' + name + 'Last' + str(comp_from*-1) + '.jpg')
        else:
            plt.savefig('images/' + name + str(comp_from) + 'And' + str(comp_to) + '.jpg')
        plt.show()
    else:
        return x_t_i_un

def plotData(x1, x2, x3, x4, from_comp, to_comp):
    if from_comp == 0:
        pca= PCA(n_components= to_comp)
        x_t1= pca.fit_transform(x1)
        x_t2= pca.fit_transform(x2)
        x_t3= pca.fit_transform(x3)
        x_t4= pca.fit_transform(x4)    
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    
    else: # for the cases 3th + 4th PCs and 10th + 11th PCs
        #it's possible to adapt the code for all cases of input such as -2 and -1 PCs
        #but in the text, for this point, it is not required
        from_comp-=1
        
        pca= PCA()
        pca.fit(x1)
        n_eigenvect1= pca.components_[from_comp:to_comp,:]
        x_t1= np.dot(np.array(x1), np.array(n_eigenvect1).T)
        
        pca= PCA()
        pca.fit(x2)
        n_eigenvect2= pca.components_[from_comp:to_comp,:]
        x_t2= np.dot(np.array(x2), np.array(n_eigenvect2).T)
        
        pca= PCA()
        pca.fit(x3)
        n_eigenvect3= pca.components_[from_comp:to_comp,:]
        x_t3= np.dot(np.array(x3), np.array(n_eigenvect3).T)
        
        pca= PCA()
        pca.fit(x4)
        n_eigenvect4= pca.components_[from_comp:to_comp,:]
        x_t4= np.dot(np.array(x4), np.array(n_eigenvect4).T)
        
        plt.xlabel('PC'+str(from_comp+1))
        plt.ylabel('PC'+str(to_comp))
    
    line1= plt.scatter(x_t1[:,0], x_t1[:,1], c='y')
    line2= plt.scatter(x_t2[:,0], x_t2[:,1], c='r')
    line3= plt.scatter(x_t3[:,0], x_t3[:,1], c='m')
    line4= plt.scatter(x_t4[:,0], x_t4[:,1], c='b')
    plt.legend((line1, line2, line3, line4), ('dog', 'guitar', 'house', 'person'))
    if from_comp == 0:
        plt.savefig('images/scatterFirst'+ str(from_comp+1) + '.jpg')
    else:
        plt.savefig('images/scatterFirst'+ str(from_comp) + 'and' + str(to_comp) + '.jpg')
    plt.show()
    
def classifierGNB(data, means, devs, labels, from_comp, to_comp, name):
    if from_comp == 0 and to_comp == 0:
        x= np.array(data).copy()
    if from_comp == 0 and to_comp > 0:
        x= doPCA(data, means, devs, 2, 0, 1, '')
    if from_comp > 0 and to_comp > 0:
        x= doPCAManually(data, means, devs, from_comp, to_comp, 0, 1, name)
        
    x_train, x_test, lab_train, lab_test= train_test_split(x, labels, test_size=0.2)
    classifier= GaussianNB()
    classifier.fit(x_train,lab_train)
    classifier.predict(x_test)
    score= classifier.score(x_test, lab_test)
    score *= 100
    print('Accuracy: %.2f' %(score) + str('%'))
    

def readPath(path):
    '''
    Receive the path
    For each data convert the 3D array in 154587-dimensional vector
    Return the dataset
    '''
    x= []
    i= 0
    for root, dirs, files in os.walk(path):
        files.sort()
        for file in files:
            file_name= root + str('/') + file
            img_data= np.asarray(Image.open(file_name)) #3D array
            images= img_data.ravel() #convert in 154587-dimensional vector
            x.append(images) #collect each image into the N x 154587 matrix
            i+=1
    return np.array(x)

def readPathAllTogether(path):
    '''
    Receive the path
    For each data convert the 3D array in 154587-dimensional vector and store the label
    Return the tuple (dataset, labels)
    '''
    x= []
    labels= []
    i= 0
    for root, dirs, files in os.walk(path):
        files.sort()
        for file in files:
            file_name= root + str('/') + file
            img_data= np.asarray(Image.open(file_name)) #3D array
            labels.append(root.split('/')[2])
            images= img_data.ravel() #convert in 154587-dimensional vector
            x.append(images) #collect each image into the N x 154587 matrix
            i+=1
    return np.array(x), np.array(labels)

############### main ##########################################################
data_path= '../PACS_homework/' 
x_dog= np.array([])
x_guitar= np.array([])
x_house= np.array([])
x_person= np.array([])

# Read distict class for PCA
for root, dirs, files in os.walk(data_path):
    if root.split('/')[2] == 'dog':
        x_dog= readPath(root)
    elif root.split('/')[2] == 'guitar':
        x_guitar= readPath(root)
    elif root.split('/')[2] == 'house':
        x_house= readPath(root)
    elif root.split('/')[2] == 'person':
        x_person= readPath(root)

###############################################################################
# 1. PCA
# Read data grouped by class and apply for each of them the PCA

# PCA for dog
x_dog_norm, dog_mean, dog_std= normalize(x_dog)

doPCA(x_dog_norm, dog_mean, dog_std, 2, 0, 0, 'dog') # dog 0 - with the first 2PCs
doPCA(x_dog_norm, dog_mean, dog_std, 6, 0, 0, 'dog') # dog 0 - with the first 6PCs
doPCA(x_dog_norm, dog_mean, dog_std, 60, 0, 0, 'dog') # dog 0 - with the first 60PCs
doPCAManually(x_dog_norm, dog_mean, dog_std, -6, 0, 0, 0, 'dog') # dog 0 - with the last 6PCs

# PCA for guitar
x_guitar_norm, guitar_mean, guitar_std= normalize(x_guitar)

doPCA(x_guitar_norm, guitar_mean, guitar_std, 2, 0, 0, 'guitar') # guitar 0 - with the first 2PCs
doPCA(x_guitar_norm, guitar_mean, guitar_std, 6, 0, 0, 'guitar') # guitar 0 - with the first 6PCs
doPCA(x_guitar_norm, guitar_mean, guitar_std, 60, 0, 0, 'guitar') # guitar 0 - with the first 60PCs
doPCAManually(x_guitar_norm, guitar_mean, guitar_std, -6, 0, 0, 0, 'guitar') # guitar 0 - with the last 6PCs

# PCA for house
x_house_norm, house_mean, house_std= normalize(x_house)

doPCA(x_house_norm, house_mean, house_std, 2, 0, 0, 'house') # house 0 - with the first 2PCs
doPCA(x_house_norm, house_mean, house_std, 6, 0, 0, 'house') # house 0 - with the first 6PCs
doPCA(x_house_norm, house_mean, house_std, 60, 0, 0, 'house') # house 0 - with the first 60PCs
doPCAManually(x_house_norm, house_mean, house_std, -6, 0, 0, 0, 'house') # house 0 - with the last 6PCs

# PCA for person
x_person_norm, person_mean, person_std= normalize(x_person)

doPCA(x_person_norm, person_mean, person_std, 2, 309, 0, 'person') # person 309 - with the first 2PCs
doPCA(x_person_norm, person_mean, person_std, 6, 309, 0, 'person') # person 309 - with the first 6PCs
doPCA(x_person_norm, person_mean, person_std, 60, 309, 0, 'person') # person 309 - with the first 60PCs
doPCAManually(x_person_norm, person_mean, person_std, -6, 0, 0, 0, 'person') # person 309 - with the last 6PCs

# PCA for entire dataset (unbalanced dataset)
x_all, labels= readPathAllTogether(data_path)
x_all_norm, means_all, devs_all= normalize(x_all)

doPCA(x_all_norm, means_all, devs_all, 2, 0, 0, 'dogFromAll')
doPCA(x_all_norm, means_all, devs_all, 6, 0, 0, 'dogFromAll')
doPCA(x_all_norm, means_all, devs_all, 60, 0, 0, 'dogFromAll')
doPCAManually(x_all_norm, means_all, devs_all, -6, 0, 0, 0, 'dog')

# Scatter Plot
plotData(x_dog_norm, x_guitar_norm, x_house_norm, x_person_norm, 0, 2) # first 2PCs
plotData(x_dog_norm, x_guitar_norm, x_house_norm, x_person_norm, 3, 4) # 3thPC and 4thPC
plotData(x_dog_norm, x_guitar_norm, x_house_norm, x_person_norm, 10, 11) # 10thPC and 11thPC


###############################################################################
# 2. GaussianNB Classifier

# Read all together for GaussianNB classifier
x_class= np.array([])
labels= np.array([])
x_class, labels= readPathAllTogether(data_path)

# Normalize
x_class_norm, means_class, devs_class= normalize(x_class)
print('Classifier on entire dataset: ')
classifierGNB(x_class_norm, means_class, devs_class, labels, 0, 0, '')
print('Classifier on first 2PCs: ')
classifierGNB(x_class_norm, means_class, devs_class, labels, 0, 2, '')
print('Classifier on 3th and 4th PCs: ')
classifierGNB(x_class_norm, means_class, devs_class, labels, 3, 4, '')