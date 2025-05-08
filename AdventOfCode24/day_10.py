 
import numpy as np

def follow_branch(array,init_pos,columnlen):
    """
    Follows a branch up to its end or the next branching point (for recursion purposes).
    """
    val = array[init_pos[0],init_pos[1]]
    
    pos = init_pos
    num_branches = 0
    branch_positions = []
    #ugly
    if (0 < pos[1] <= columnlen-1):
        if (array[pos[0],pos[1]-1] == val+1):
            num_branches += 1
            branch_positions.append([pos[0],pos[1]-1])
    if (0 <= pos[1] < columnlen-1):
        if (array[pos[0],pos[1]+1] == val+1):
            num_branches += 1
            branch_positions.append([pos[0],pos[1]+1])
    if (0 < pos[0] <= columnlen-1):
        if (array[pos[0]-1,pos[1]] == val+1):
            num_branches += 1
            branch_positions.append([pos[0]-1,pos[1]])
    if (0 <= pos[0] < columnlen-1):
        if (array[pos[0]+1,pos[1]] == val+1):
            num_branches += 1
            branch_positions.append([pos[0]+1,pos[1]])

    return num_branches,val+1,branch_positions


dirs = np.asarray(["up","right","down","left"])

onedim = np.loadtxt("day_10/input.dat",dtype="U53",comments=None)

twodim = np.empty((len(onedim),len(onedim)),dtype=int)
for i in range(len(onedim)):
    twodim[i,:] = np.asarray(list(onedim[i]))

total_strength = 0
columnlen = np.shape(twodim)[1]

for x in range(len(onedim)):
    for y in range(len(onedim)):
        if twodim[x,y] == 0:
            #start searching for trail
            init_pos = [x,y]
            positions = [init_pos]
            val = 0
            # Recursive search for trails from trailhead
            while val < 9:
                current_pos_all = []
                for pos in positions:
                    num_branches, value, current_positions = follow_branch(twodim,pos,columnlen)
                    
                    #print("Branches at value %d: %d" % (value,num_branches))
                    current_pos_all = current_pos_all + current_positions
                    val = value
                positions = current_pos_all
            #print(np.unique(positions,axis=0))
            #print(np.shape(np.unique(positions,axis=0))[0])
            #total_strength += np.shape(np.unique(positions,axis=0))[0]

            #for Part 2
            total_strength += np.shape(positions)[0]
        else:
            continue
print(total_strength)