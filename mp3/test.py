import numpy as np
# file = open('data/trainset' + '/' + 'indexing.txt', 'r')
# T = []
# A = []
# data = file.readline()
# line = data.split()
# sample = open('data/trainset' + '/' + line[1].strip('\n'))
# x = sample.read().split()
# A.append(x)
# T.append(line[0])
# while data:
#     data = file.readline()
#     line = data.split()
#     try:
#         sample = open('data/trainset' + '/' + line[1].strip('\n'))
#         x = sample.read().split()
#         A.append(x)
#         T.append(line[0])
#     except:
#         pass
#
# A0 = np.array(A)
# A1 = np.ones(A0.shape[0])
# A = np.c_[A1, A0]
# T = np.array(T)
# print(np.random.uniform(size=(2, 10)))
T=[1,1,1,1,1,1,1]
T =np.array(T)
Q=[]
for i in range(len(T)):
    if T[i] == -1:
        T[i] = 0
    elif T[i] == 1:
        Q.append([1])
print(Q)