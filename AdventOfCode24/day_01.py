import numpy as np

pairs = np.loadtxt("input.dat",dtype=int)
leftlist = np.sort(pairs[:,0])
rightlist = np.sort(pairs[:,1])
#diff = rightlist-leftlist
#print(np.sum(np.abs(diff)))
similarity_index = 0
for i in range(len(leftlist)):
    similarity_index += leftlist[i]*np.count_nonzero(rightlist == leftlist[i])
print(similarity_index)