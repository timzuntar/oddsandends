import numpy as np

onedim = np.loadtxt("input.dat",dtype="U140")
twodim = np.full((len(onedim)+2,len(onedim)+2),".",dtype="|U1")
twodim_idx = np.zeros((len(onedim)+2,len(onedim)+2),dtype=int)
for i in range(len(onedim)):
    twodim[i+1,1:-1] = np.asarray(list(onedim[i]))

""" query = np.asarray(["X","M","A","S"])
num_queries = 0

for j in range(1,len(onedim)+1):
    for k in range(1,len(onedim)+1):
        if twodim[j,k] == "X":
            #inefficient hackery:
            #right
            try:
                if np.array_equal(twodim[j:j+4,k],query):
                    num_queries += 1
                    twodim_idx[j:j+4,k] = 1
            except:
                pass
            #left
            try:
                if np.array_equal(twodim[j:j-4:-1,k],query):
                    num_queries += 1
                    twodim_idx[j:j-4:-1,k] = 1
            except:
                pass
            #down
            try:
                if np.array_equal(twodim[j,k:k+4],query):
                    num_queries += 1
                    twodim_idx[j,k:k+4] = 1
            except:
                pass
            #up
            try:
                if np.array_equal(twodim[j,k:k-4:-1],query):
                    num_queries += 1
                    twodim_idx[j,k:k-4:-1] = 1
            except:
                pass
            #right down
            try:
                if np.array_equal(np.diagonal(twodim[j:j+4,k:k+4]),query):
                    num_queries += 1
                    twodim_idx[j,k] = 1
                    twodim_idx[j+1,k+1] = 1
                    twodim_idx[j+2,k+2] = 1
                    twodim_idx[j+3,k+3] = 1
            except:
                pass
            #left up
            try:
                if np.array_equal(np.diagonal(np.fliplr(np.flipud(twodim[j-3:j+1,k-3:k+1]))),query):
                    num_queries += 1
                    twodim_idx[j,k] = 1
                    twodim_idx[j-1,k-1] = 1
                    twodim_idx[j-2,k-2] = 1
                    twodim_idx[j-3,k-3] = 1
            except:
                pass
            #right up
            try:
                if np.array_equal(np.diagonal(np.flipud(twodim[j-3:j+1,k:k+4])),query):
                    num_queries += 1
                    twodim_idx[j,k] = 1
                    twodim_idx[j-1,k+1] = 1
                    twodim_idx[j-2,k+2] = 1
                    twodim_idx[j-3,k+3] = 1
            except:
                pass
            #left down
            try:
                if np.array_equal(np.diagonal(np.fliplr(twodim[j:j+4,k-3:k+1])),query):
                    num_queries += 1
                    twodim_idx[j,k] = 1
                    twodim_idx[j+1,k-1] = 1
                    twodim_idx[j+2,k-2] = 1
                    twodim_idx[j+3,k-3] = 1
            except:
                pass     

print(num_queries)
print(twodim)
print(twodim_idx) """

#############
# second part
#############
num_queries = 0

for j in range(2,len(onedim)):
    for k in range(2,len(onedim)):
        if twodim[j,k] == "A":
            #inefficient hackery:
            #vertical/horizontal
            try:
                if (twodim[j-1,k] == "M" and twodim[j+1,k] == "S" or twodim[j-1,k] == "S" and twodim[j+1,k] == "M"):
                    if (twodim[j,k-1] == "M" and twodim[j,k+1] == "S" or twodim[j,k-1] == "S" and twodim[j,k+1] == "M"):
                        num_queries += 0
            except:
                pass
            #diagonals
            try:
                if (twodim[j-1,k-1] == "M" and twodim[j+1,k+1] == "S" or twodim[j-1,k-1] == "S" and twodim[j+1,k+1] == "M"):
                    if (twodim[j-1,k+1] == "M" and twodim[j+1,k-1] == "S" or twodim[j-1,k+1] == "S" and twodim[j+1,k-1] == "M"):
                        num_queries += 1
            except:
                pass
print(num_queries)