import numpy as np
import pandas as pd

reports = pd.read_csv("input.dat",sep=" ",names=list(range(12))).dropna(axis='columns', how='all')
num_safe = 0
idx_safe = []
for i in range(1000):
    line = reports[i:i+1].to_numpy()[0]
    cleaned = line[~np.isnan(line)].astype(int)
    diffs = cleaned[1:]-cleaned[:-1]

    if(np.all(((diffs>=1)&(diffs<=3))) or np.all(((diffs>=-3)&(diffs<=-1)))):
            num_safe += 1
            idx_safe.append(i)
    else:
        for j in range(0,len(cleaned)):
            cleaned_removed = np.delete(cleaned,j)
            diffs_removed = cleaned_removed[1:]-cleaned_removed[:-1]
            if(np.all(((diffs_removed>=1)&(diffs_removed<=3))) or np.all(((diffs_removed>=-3)&(diffs_removed<=-1)))):
                num_safe += 1
                idx_safe.append(i)
                print(diffs,diffs_removed)
                break
print(num_safe)