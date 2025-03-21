import numpy as np
import itertools

onedim = np.loadtxt("input.dat",dtype="U50",comments=None)

twodim = np.full((len(onedim),len(onedim)),".",dtype="|U1")
for i in range(len(onedim)):
    twodim[i,:] = np.asarray(list(onedim[i]))

allchars,charcounts = np.unique(twodim,return_counts=True)
allchars = allchars[np.where(charcounts > 1)]
allchars = allchars[allchars != "."]    # those are all instances of antennae

nodelist = []
# We now go frequency by frequency...
for freq in allchars:
    # Find all antenna locations
    loclist = np.argwhere(twodim == freq)
    # Get all possible location pairs
    loc_combos = itertools.combinations(loclist,2)
    # Now start searching for antinodes
    
    for op in loc_combos:
        x1 = op[0][0]
        y1 = op[0][1]
        x2 = op[1][0]
        y2 = op[1][1]
        dx = x2-x1
        dy = y2-y1

        """
        # Outer ones
        loc1 = [x1-dx,y1-dy]
        loc2 = [x2+dx,y2+dy]
        if -1 < loc1[0] < 50 and -1 < loc1[1] < 50:
            nodelist.append(loc1)
        if -1 < loc2[0] < 50 and -1 < loc2[1] < 50:
            nodelist.append(loc2)
        # potential inner ones
        if dx % 3 == 0 and dy % 3 == 0:
            loc3 = [x1+dx//3,y1+dy//3]
            loc4 = [x2-dx//3,y2-dy//3]
            nodelist.append(loc3)
            nodelist.append(loc4) """

        # Original antenna positions
        nodelist.append(op[0])
        nodelist.append(op[1])

        # Outer ones
        harmonic = 1
        in_bounds = True
        while in_bounds:
            loc1 = [x1-harmonic*dx,y1-harmonic*dy]
            if -1 < loc1[0] < 50 and -1 < loc1[1] < 50:
                nodelist.append(loc1)
                harmonic += 1
                continue
            else:
                in_bounds = False

        harmonic = 1
        in_bounds = True
        while in_bounds:
            loc2 = [x2+harmonic*dx,y2+harmonic*dy]
            if -1 < loc2[0] < 50 and -1 < loc2[1] < 50:
                nodelist.append(loc2)
                harmonic += 1
                continue
            else:
                in_bounds = False
        
        # potential inner ones - this shouldn't be needed anymore
        #if dx % 3 == 0 and dy % 3 == 0:
        #    loc3 = [x1+dx//3,y1+dy//3]
        #    loc4 = [x2-dx//3,y2-dy//3]
        #    nodelist.append(loc3)
        #    nodelist.append(loc4)

nodelist = np.asarray(nodelist)
global_nodenum = len(np.unique(nodelist,axis=0))
print(global_nodenum)