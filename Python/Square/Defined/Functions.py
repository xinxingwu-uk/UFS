#----------------------------Reproducible----------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import random as rn
import os

seed=0
os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
rn.seed(seed)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

#tf.set_random_seed(seed)
tf.compat.v1.set_random_seed(seed)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

K.set_session(sess)
#----------------------------Reproducible----------------------------------------------------------------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--------------------------------------------------------------------------------------------------------------------------------
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Activation, Dropout, Layer
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras import optimizers,initializers,constraints,regularizers
from keras import backend as K
from keras.callbacks import LambdaCallback,ModelCheckpoint
from keras.utils import plot_model

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

#--------------------------------------------------------------------------------------------------------------------------------
def show_data_figures(p_data,w=20,h=20,columns = 20):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(p_data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(p_data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(p_data[i,:].reshape((w, h)),plt.cm.gray)#
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------
def show_data_figures_with_keyfeature(p_data,p_key_feature_catch,w=20,h=20,columns = 20):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(p_data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(p_data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(p_data[i,:].reshape((w, h)),plt.cm.gray)
        for key_feature_catch_i in np.arange(len(p_key_feature_catch)):
            plt.scatter(p_key_feature_catch[key_feature_catch_i][1],p_key_feature_catch[key_feature_catch_i][0],s=0.5,color='r')
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------
def top_k_keep(p_arr_,p_top_k_):
    top_k_idx=p_arr_.argsort()[::-1][0:p_top_k_]
    top_k_value=p_arr_[top_k_idx]
    return np.where(p_arr_<top_k_value[-1],0,p_arr_)

#--------------------------------------------------------------------------------------------------------------------------------
def show_feature_selection(p_file_name,p_test_data,p_sample_number=40,p_key_number=36):
    file = h5py.File(p_file_name,'r') 
    data = file['feature_selection']['feature_selection']['kernel:0']
    weight_top_k=top_k_keep(np.array(data),p_key_number)
    show_data_figures(np.dot(p_test_data[0:p_sample_number],np.diag(weight_top_k)))
    file.close()

#--------------------------------------------------------------------------------------------------------------------------------
'''
def top_k_keepWeights_1(p_arr_,p_top_k_):
    top_k_idx=p_arr_.argsort()[::-1][0:p_top_k_]
    top_k_value=p_arr_[top_k_idx]
    if np.sum(p_arr_>0)>p_top_k_:
        p_arr_=np.where(p_arr_<top_k_value[-1],0,1)
    else:
        p_arr_=np.where(p_arr_<=0,0,1) 
    return p_arr_
'''

def top_k_keepWeights_1(p_arr_,p_top_k_,p_ignore_equal=True):
    top_k_idx=p_arr_.argsort()[::-1][0:p_top_k_]
    top_k_value=p_arr_[top_k_idx]
    if np.sum(p_arr_>0)>p_top_k_:
        if p_ignore_equal:
            p_arr_=np.where(p_arr_<top_k_value[-1],0,1)
        else:
            p_arr_=np.where(p_arr_<=top_k_value[-1],0,1)
    else:
        p_arr_=np.where(p_arr_<=0,0,1) 
    return p_arr_

#--------------------------------------------------------------------------------------------------------------------------------
def hierarchy_top_k_keep(p_arr_,p_choose_top_k_,p_selection_hierarchy):

    if p_selection_hierarchy==1:
        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value=p_arr_[top_k_idx]        
        return np.where(p_arr_<top_k_value[-1],0,p_arr_)
    elif p_selection_hierarchy>1:
        top_k_idx=p_arr_.argsort()[::-1][0:(p_selection_hierarchy-1)*p_choose_top_k_]
        top_k_value_1=p_arr_[top_k_idx]
        p_arr_=np.where(p_arr_<top_k_value_1[-1],p_arr_,0)
        
        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value_2=p_arr_[top_k_idx]
        return np.where(p_arr_<top_k_value_2[-1],0,p_arr_)      

#--------------------------------------------------------------------------------------------------------------------------------
def show_hierarchy_feature_selection(p_file_name,p_test_data,p_selection_hierarchy=1,p_sample_number=40,p_key_number=36):
    file = h5py.File(p_file_name,'r') 
    data = file['feature_selection']['feature_selection']['kernel:0']
    weight_top_k=hierarchy_top_k_keep(np.array(data),p_key_number,p_selection_hierarchy)
    show_data_figures(np.dot(p_test_data[0:p_sample_number],np.diag(weight_top_k)))
    file.close()
    
#--------------------------------------------------------------------------------------------------------------------------------
def hierarchy_top_k_keepWeights_1(p_arr_,p_choose_top_k_,p_selection_hierarchy):

    if p_selection_hierarchy==1:
        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value=p_arr_[top_k_idx]        
        if np.sum(p_arr_>0)>p_choose_top_k_:
            p_arr_=np.where(p_arr_<top_k_value[-1],0,1)
        else:
            p_arr_=np.where(p_arr_<=0,0,1) 
        return p_arr_   

    elif p_selection_hierarchy>1:
        top_k_idx=p_arr_.argsort()[::-1][0:(p_selection_hierarchy-1)*p_choose_top_k_]
        top_k_value_1=p_arr_[top_k_idx]  
        
        p_arr_=np.where(p_arr_<top_k_value_1[-1],p_arr_,0)

        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value_2=p_arr_[top_k_idx]
        
        if np.sum(p_arr_>0)>p_choose_top_k_:
            p_arr_=np.where(p_arr_<top_k_value_2[-1],0,1)
        else:
            p_arr_=np.where(p_arr_<=0,0,1) 
        return p_arr_ 

#--------------------------------------------------------------------------------------------------------------------------------
def show_data_figures_with_hierarchy_keyfeature(p_data,p_key_feature_catch,w=20,h=20,columns = 20):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(p_data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(p_data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(p_data[i,:].reshape((w, h)),plt.cm.gray)
        p_key_feature_catch_i=p_key_feature_catch[i]
        for key_feature_catch_i in np.arange(len(p_key_feature_catch_i)):
            plt.scatter(p_key_feature_catch_i[key_feature_catch_i][1],p_key_feature_catch_i[key_feature_catch_i][0],s=10,color='r')
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------
def ETree(p_train_feature,p_train_label,p_test_feature,p_test_label,p_seed,p_n_estimators=50):
    clf = ExtraTreesClassifier(n_estimators=p_n_estimators, random_state=p_seed)
    
    # Training
    clf.fit(p_train_feature, p_train_label)
    
    # Training accuracy
    print('Training accuracy：',clf.score(p_train_feature, np.array(p_train_label)))
    print('Training accuracy：',accuracy_score(np.array(p_train_label),clf.predict(p_train_feature)))
    #print('Training accuracy：',np.sum(clf.predict(p_train_feature)==np.array(p_train_label))/p_train_label.shape[0])

    # Testing accuracy
    print('Testing accuracy：',clf.score(p_test_feature, np.array(p_test_label)))
    print('Testing accuracy：',accuracy_score(np.array(p_test_label),clf.predict(p_test_feature)))
    #print('Testing accuracy：',np.sum(clf.predict(p_test_feature)==np.array(p_test_label))/p_test_label.shape[0])
    
#--------------------------------------------------------------------------------------------------------------------------------   
def compress_zero(p_data_matrix,p_key_feture_number):
    p_data_matrix_Results=[]
    for p_data_matrix_i in p_data_matrix:
        p_data_matrix_Results_i=[]
        for ele_i in p_data_matrix_i:
            if ele_i!=0:
                p_data_matrix_Results_i.append(ele_i)
        if len(p_data_matrix_Results_i)<p_key_feture_number:
            for add_i in np.arange(p_key_feture_number-len(p_data_matrix_Results_i)):
                p_data_matrix_Results_i.append(0)
        p_data_matrix_Results.append(p_data_matrix_Results_i)
    return np.array(p_data_matrix_Results)

#--------------------------------------------------------------------------------------------------------------------------------
def compress_zero_withkeystructure(p_data_matrix,p_selected_position):
    p_data_matrix_Results=[]
    for p_data_matrix_i in p_data_matrix:
        p_data_matrix_Results_i=[]
        for selection_j in p_selected_position:
            p_data_matrix_Results_i.append(p_data_matrix_i[selection_j])
        p_data_matrix_Results.append(p_data_matrix_Results_i)
    return np.array(p_data_matrix_Results)

#--------------------------------------------------------------------------------------------------------------------------------
def k_index_argsort_1d(data,k):
    top_k_idx=data.argsort()[::-1][0:k]
    return top_k_idx


#--------------------------------------------------------------------------------------------------------------------------------
def write_to_csv(p_data,p_path):
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a',header=False,index=False,sep=',')
