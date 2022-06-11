#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sys
import math
import copy


# In[ ]:


def main():
    print("Welcome to the feature search Algorithm")
    data=pd.read_fwf('/Users/ghannesh59/Downloads/CS205_SP_2022_Largetestdata__10.txt',header=None)
    (rows,columns)=data.shape
    algorithm=int(input('Enter the algorithm you want to run:\n'
                     '\n1. forward_selection'
                     '\n2. backward_elimination\n'))
    print("This dataset has {} features, with {} instances".format(columns-1,rows))
    if algorithm==1:
        return forward_selection(data)
    if algorithm==2:
        return backward_elimination(data)


# In[ ]:


def forward_selection(data):
    (rows,columns)=data.shape
    current_features=[]
    dict1={}
    for i in range(1,columns):
        print("for the {} level of search tree".format(i))
        feature_set=[]
        best_accuracy=0
        for j in range(1,columns):
            if j not in current_features:
                print("consider adding the feature {}".format(j))
                feature_set=copy.deepcopy(current_features)
                feature_set.append(j)
                accuracy=leave_one_out_cross_validation(data,feature_set)
                print("using feature {} the accuracy is {}".format(feature_set,accuracy))
                if accuracy>best_accuracy:
                    best_accuracy=accuracy
                    feature_to_add_at_this_level=j
        current_features.append(feature_to_add_at_this_level)
        current_features_copy=copy.deepcopy(current_features)
        dict1[best_accuracy]=current_features_copy
        print('\n')
        print("on level {}  the feature set {} has best accuracy {} ".format(i,current_features,best_accuracy))
    max1=max(dict1.keys())
    best_subset=dict1[max1]
    print('\n')
    print("The search is completed.The best feature subset is {} with accuracy of {}".format(best_subset,max1))


# In[ ]:


def leave_one_out_cross_validation(data,feature_set):
    copy_data=copy.deepcopy(data)
    (rows,columns)=data.shape
    for k in range(1,columns):
        if k not in feature_set:
            copy_data.iloc[:,k]=0.0       
    no_of_correctly_classified=0
    for i in range(rows):
        object_to_classify=copy_data.iloc[i,1:columns]
        label_of_object_to_classify=copy_data.iloc[i,0]
        nearest_neighbor_distance=sys.maxsize
        nearest_neighbor_location=sys.maxsize
        for j in range(rows):
            distance=0
            if j!=i:
                distance=math.dist(object_to_classify,copy_data.iloc[j,1:])
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance=distance
                    nearest_neighbor_location=j
                    nearest_neighbor_label=copy_data.iloc[nearest_neighbor_location,0]
        if label_of_object_to_classify==nearest_neighbor_label:
            no_of_correctly_classified=no_of_correctly_classified+1
    accuracy=no_of_correctly_classified/rows
    return accuracy
                
    


# In[ ]:


def backward_elimination(data):
    (rows,columns)=data.shape
    current_features=[]
    dict1={}
    for i in range(1,columns):
        current_features.append(i)
    feature_set=[]
    feature_set=copy.deepcopy(current_features)
    accuracy=leave_one_out_cross_validation(data,feature_set)
    print("The entire feature set {} has accuracy of {} ".format(current_features,accuracy))
    print('\n')
    for i in range(2,columns):
        best_accuracy=0
        feature_set=[]
        for j in range(1,columns):
            if j in current_features:
                feature_set=copy.deepcopy(current_features)
                print("consider removing the feature {} from {}".format(j,feature_set))
                feature_set.remove(j)
                accuracy=leave_one_out_cross_validation(data,feature_set)
                print("using feature {} the accuracy is {}".format(feature_set,accuracy))
                if accuracy>best_accuracy:
                    best_accuracy=accuracy
                    feature_to_remove_in_this_level=j
        current_features.remove(feature_to_remove_in_this_level)
        current_features_copy=copy.deepcopy(current_features)
        dict1[best_accuracy]=current_features_copy
        print('\n')
        print("on level {}  the feature set {} has best accuracy {} ".format(i,current_features,best_accuracy))
    max1=max(dict1.keys())
    best_subset=dict1[max1]
    print('\n')
    print("The search is completed.The best feature subset is {} with accuracy of {}".format(best_subset,max1))
        


# In[ ]:


if __name__ == "__main__":
    main()

