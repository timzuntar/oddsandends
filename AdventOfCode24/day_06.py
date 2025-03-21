import numpy as np

dirs = np.asarray(["up","right","down","left"])

onedim = np.loadtxt("input.dat",dtype="U140",comments=None)

twodim = np.full((len(onedim)+2,len(onedim)+2),"X",dtype="|U1")
twodim_idx = np.zeros((len(onedim)+2,len(onedim)+2),dtype=int)
for i in range(len(onedim)):
    twodim[i+1,1:-1] = np.asarray(list(onedim[i]))

location = tuple(np.argwhere(twodim == "^")[0])
starting_location = location
twodim[location] = "."
twodim_copy = np.copy(twodim)
locs = [location]

current_dir_idx = 0
out_of_bounds = False

while (out_of_bounds == False):

    current_dir = dirs[current_dir_idx%4]
    if current_dir == "left":
        nextloc = (location[0],location[1]-1)
    elif current_dir == "up":
        nextloc = (location[0]-1,location[1])
    elif current_dir == "right":
        nextloc = (location[0],location[1]+1)
    elif current_dir == "down":
        nextloc = (location[0]+1,location[1])

    if twodim[nextloc] == ".": #take step forward
        location = nextloc
        locs.append(location)
        continue
    elif twodim[nextloc] == "#":    # turn right
        current_dir_idx += 1
        continue
    elif twodim[nextloc] == "X":    # leave map
        out_of_bounds = True
        continue

locs_arr = np.asarray(locs)
unique_locs = np.unique(locs_arr,axis=0)
print(np.shape(unique_locs))

#restart

num_looping_positions = 0

for i in range(1,len(onedim)+1):
    for j in range(1,len(onedim)+1):
        twodim = np.copy(twodim_copy)
        current_dir_idx = 0
        out_of_bounds = False
        loopfound = False
        location = starting_location

        locs_obstacle = [location]
        dir_indices = [current_dir_idx]

        if ((i,j) != starting_location and twodim[i,j] == "."):
            twodim[i,j] = "#"
            print("placed obstacle on field [%d,%d]" % (i,j))

            ctr = 0
            while (out_of_bounds == False):
                current_dir = dirs[current_dir_idx%4]
                if current_dir == "left":
                    nextloc = (location[0],location[1]-1)
                elif current_dir == "up":
                    nextloc = (location[0]-1,location[1])
                elif current_dir == "right":
                    nextloc = (location[0],location[1]+1)
                elif current_dir == "down":
                    nextloc = (location[0]+1,location[1])

                if twodim[nextloc] == ".": #take step forward
                    location = nextloc
                    locs_obstacle.append(location)
                    dir_indices.append(current_dir_idx%4)
                    ctr += 1
                elif twodim[nextloc] == "#":    # turn right
                    current_dir_idx += 1
                elif twodim[nextloc] == "X":    # leave map
                    out_of_bounds = True 
                
                if ctr > 10000 and location in locs_obstacle[15:-2]:
                    locidx = []
                    start = 0
                    while True:
                        try:
                            index = locs_obstacle[2:-2].index(location,start)
                            locidx.append(index)
                            start = index + 1
                            if len(locidx) > 4:
                                break
                        except ValueError:
                            break
                    for idx in locidx:
                        if current_dir_idx%4 == dir_indices[idx+2]:
                            num_looping_positions += 1
                            loopfound = True
                            break


                if loopfound:
                    break

print(num_looping_positions)