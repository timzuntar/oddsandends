import numpy as np
import pandas as pd

rules = np.loadtxt("rules.dat",delimiter="|",dtype=int)
before = rules[:,0]
after = rules[:,1]

sequences = pd.read_csv("input.dat",sep=",",names=list(range(30))).dropna(axis='columns', how='all')
middle_sum_ordered = 0
middle_sum_unordered = 0

for i in range(sequences.shape[0]):
    line = sequences[i:i+1].to_numpy()[0]
    cleaned = line[~np.isnan(line)].astype(int)
    ordered = True
    for rule_idx in range(len(before)):
        try:
            before_idx = np.argwhere(cleaned == before[rule_idx])[0][0]
        except:
            before_idx = -1
        try:
            after_idx = np.argwhere(cleaned == after[rule_idx])[0][0]
        except:
            after_idx = 100
        if not ((after_idx - before_idx) > 0):
            ordered = False
            break
    if ordered:
        middle_idx = len(cleaned)//2
        middle_sum_ordered += cleaned[middle_idx]
    else:
        for cleanednum in cleaned:
            indices = np.where(before == cleanednum)
            before_shortened = before[indices]
            after_shortened = after[indices]
            for rule_idx in range(len(before_shortened)):
                try:
                    before_idx = np.argwhere(cleaned == before_shortened[rule_idx])[0][0]
                    after_idx = np.argwhere(cleaned == after_shortened[rule_idx])[0][0]
                    if not ((after_idx - before_idx) > 0):
                        cleaned[before_idx],cleaned[after_idx] = cleaned[after_idx],cleaned[before_idx]
                except:
                    pass
        middle_idx = len(cleaned)//2
        middle_sum_unordered += cleaned[middle_idx]

print(middle_sum_ordered)
print(middle_sum_unordered)