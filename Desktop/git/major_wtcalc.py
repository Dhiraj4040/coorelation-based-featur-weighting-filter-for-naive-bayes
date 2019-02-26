# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:05:48 2019

@author: hp
"""
import numpy as np
import pandas as pd
import math as m

#IMPORT DATASET
data = pd.read_csv('diabetes.csv')
data.columns=['A','B','C','D','E','F','G','H','CLASS']



"""normalized mutual information between all attributes and class"""
NIAC=[] #NIAC in Normalized Mutual Information between each Attribute and Class
IAC=[]  #IAC is Mutual Information between each Attribute and Class
# mutual_info_score  calculates the mutual information between two columns of a data matrix

from sklearn.metrics.cluster import mutual_info_score
for i in range(0,8):
    mi_att_class=mutual_info_score(data.iloc[:,i],data['CLASS'])
    IAC.append(mi_att_class)
    
NIAC=IAC/(np.sum(IAC)/8)   #np.sum(IAC)  returns sum of all elements of array IAC
#print("Normalized Mutual Information between Attributes and class")
#print(NIAC)

 
#mutual information between feature and feature
"""normalized mutual information between attrubutes"""
NIAA=[] #NIAA in Normalized Mutual Information between each Attribute with other attribute
IAA=[]  #IAA in Normalized Mutual Information between each Attribute with other attribute
                                    
for x in range(0,8):
    for y in range(0,8):
        if x!=y :
            mi_att_att=mutual_info_score(data.iloc[:,x],data.iloc[:,y])
            IAA.append(mi_att_att)
    
    
NIAA=IAA/(np.sum(IAA)/56) #np.sum(IAA)  returns sum of all elements of array IAA
#print("length of NIAA=",len(NIAA))
#print("Normalized Mutual Information between Attributes ")
#print(NIAA)



avg_redundancy=[]
#print("sum of every 8 elements")
avg_redundancy=np.add.reduceat(NIAA,np.arange(0,len(NIAA),7)) #calculating sum of every 3 elements
#print(avg_redundancy)

#calculating Di and Wi
# D is the difference between the feature-class correlation and the average feature-feature correlation

D=NIAC-avg_redundancy
print("calculated value of D are")
for i in range(0,len(D)):
    print(""+D[i])

#print("calculation of D=",D)
#calculating the weight matrix W
W=[]
for j in range(0,8):
      W.append(1/(1+m.exp(-D[j])))

print("calculated weights are:")
for i in range(0,len(W)):
    print(W[i])






