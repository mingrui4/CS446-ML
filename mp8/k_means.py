from copy import deepcopy
import numpy as np
import pandas as pd
import sys
from sklearn.cluster import KMeans


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
df = pd.read_csv('data/iris.data')
data= df.iloc[:,[0,1,2,3]]
classifier = df.iloc[:,[4]]
data = np.array(data)
classifier = np.array(classifier)

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

def k_means(C):
    # Write your code here!
    C_i = {}
    C_i[0] = C
    j = 1
    Cluster_FLAG = True
    while Cluster_FLAG:
        C_test = deepcopy(C_i[j-1])
        classf = {}
        classf[0] = []
        classf[1] = []
        classf[2] = []
        length = len(data)
        for i in range(length):
            dis1 = np.sqrt(np.sum(np.square(data[i] - C_test[0])))
            dis2 = np.sqrt(np.sum(np.square(data[i] - C_test[1])))
            dis3 = np.sqrt(np.sum(np.square(data[i] - C_test[2])))
            dis = [dis1,dis2,dis3]
            min_i = np.argmin(dis)
            classf[min_i].append(data[i])
        C_final_0 = np.mean(classf[0],axis=0)
        C_final_1 = np.mean(classf[1],axis=0)
        C_final_2 = np.mean(classf[2],axis=0)
        C_final = np.array([C_final_0,C_final_1,C_final_2])

        C_i[j] = C_final
        if (C_i[j] == C_i[j - 1]).all():
            Cluster_FLAG = False
        j = j + 1
    return C_final

print(k_means(C))
#
# est = KMeans(n_clusters = 3,max_iter=6,init=C,n_init=1)  # 3 clusters
# est.fit(data)
# y_kmeans = est.predict(data)
# centroids = est.cluster_centers_
# print(centroids)








