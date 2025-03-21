import numpy as np
import pandas as pd
import itertools

sequences = pd.read_csv("input.dat",sep=":| ",names=list(range(30))).dropna(axis='columns', how='all')

total_cal_result = 0

for i in range(len(sequences)):
    line = sequences[i:i+1].to_numpy()[0]
    cleaned = np.asarray(line[~np.isnan(line)].astype(int))
    num_operators = len(cleaned)-2
    
    #let's consider + == 0 and * == 1
    oplist = itertools.product([0,1,2],repeat=num_operators)
    for op in oplist:
        #cleaned = np.asarray(line[~np.isnan(line)].astype(int))
        #multiplication blocks need to be dealt with first
        #catblocks = []
        #block = []
        #for j in range(len(op)):
        #   if op[j] == 2:
        #       block.append(j)
        #   elif (op[j] == 0 or op[j] == 1):
        #        if len(block) > 0:
        #            catblocks.append(block)
        #        block = []
        #    if (j == (len(op)-1) and len(block) > 0):
        #        catblocks.append(block)
        #nums = list(itertools.chain(*catblocks))

        #test operation for one operator order
        #indices_to_delete = []
        #for block in catblocks:
        #    string = ""
        #    for element in range(block[0]+1,block[-1]+3):
        #        string = string + str(cleaned[element])
        #    for element in range(len(block)):
        #        indices_to_delete.append(block[element])
        #    #replace the first value with the final number
        #    cleaned[block[0]+1] = int(string)
        #delete the rest from the number and operator arrays
        #if len(indices_to_delete) > 0:
        #    op = np.delete(op,np.asarray(indices_to_delete))
        #    cleaned = np.delete(cleaned,np.asarray(indices_to_delete)+2)

        result = cleaned[1]
        for j in range(len(op)):
            if (op[j] == 0):
                result += cleaned[j+2]
            elif (op[j] == 1):
                result *= cleaned[j+2]
            elif (op[j] == 2):
                result = int(str(result) + str(cleaned[j+2]))

        if result == cleaned[0]:
            total_cal_result += result
            print(result)
            break

print(total_cal_result)