import numpy as np

with open("day_09/input.dat","r") as file:
    megastring = file.read().rstrip('\n')

num_blocks = len(megastring)

if (num_blocks % 2 == 0):
    print("Are you sure this is the correct input string? The number of elements must be odd.")

# Now the IDs are assigned to a combined array to track their positions
# Iterate, then concatenate via numpy

all_indices = []

for i in range(num_blocks):
    if (i%2 == 0):  # File block
        idx = i//2
        all_indices.append(np.full(int(megastring[i]),idx,dtype=int))
    elif (i%2 == 1):    # Empty block
        all_indices.append(np.full(int(megastring[i]),-1,dtype=int))

all_indices_flattened = np.hstack(all_indices)
all_indices_final = np.copy(all_indices_flattened)

# Part 1
# Now iteratively find the first occurence of a free space and slot the files into it in reverse order (permutation?)

""" block_positions = np.argwhere(all_indices_flattened != -1).flatten()
free_space_positions = np.argwhere(all_indices_flattened == -1).flatten()

flipped_block_positions = np.flip(block_positions)

for j in range(len(free_space_positions)):
    if free_space_positions[j] < flipped_block_positions[j]:
        all_indices_final[free_space_positions[j]] = all_indices_final[flipped_block_positions[j]]
        all_indices_final[flipped_block_positions[j]] = -1

all_indices_final = all_indices_final[all_indices_final != -1]

checksum  = np.inner(all_indices_final,np.arange(len(all_indices_final)))
print(checksum) """

# Part 2
# Determine the lengths of contiguous regions
block_starts = []
free_space_starts = []
block_lengths = []
free_space_lengths = []
block_idx = []

runnning_idx = all_indices_final[0]
blocklen = 1
for idx,i in enumerate(all_indices_final[1:]):
    if i == runnning_idx:
        blocklen += 1
        if idx < len(all_indices_final)-1:  #last element
            continue
        else:
            if runnning_idx == -1:
                free_space_starts.append(idx-blocklen+2)
                free_space_lengths.append(blocklen)
            else:
                block_starts.append(idx-blocklen+2)
                block_lengths.append(blocklen)
                block_idx.append(runnning_idx)
    else:
        if runnning_idx == -1:
            free_space_starts.append(idx-blocklen+1)
            free_space_lengths.append(blocklen)
        else:
            block_starts.append(idx-blocklen+1)
            block_lengths.append(blocklen)
            block_idx.append(runnning_idx)
        runnning_idx = i
        blocklen = 1


free_space_starts = np.asarray(free_space_starts)
free_space_lengths = np.asarray(free_space_lengths)
block_starts = np.asarray(block_starts)
block_lengths = np.asarray(block_lengths)
block_idx = np.asarray(block_idx)

flipped_block_lengths = np.flip(block_lengths)
flipped_block_starts = np.flip(block_starts)
flipped_block_idx = np.flip(block_idx)

for i in range(len(flipped_block_lengths)):
    for j in range(len(free_space_lengths)):
        if (free_space_lengths[j] >= flipped_block_lengths[i]):
            # move file and shorten free space
            #print("Free space length = %d, block length = %d, moving block." % (free_space_lengths[j],flipped_block_lengths[i]))
            
            all_indices_final[flipped_block_starts[i]:flipped_block_starts[i]+flipped_block_lengths[i]] = -1
            flipped_block_starts[i] = free_space_starts[j]
            free_space_starts[j] += flipped_block_lengths[i]
            free_space_lengths[j] -= flipped_block_lengths[i]
            # flip indices
            all_indices_final[flipped_block_starts[i]:flipped_block_starts[i]+flipped_block_lengths[i]] = flipped_block_idx[i]
            break
        else:
            continue
        
    #print(all_indices_final)

all_indices_final[all_indices_final == -1] = 0
checksum  = np.inner(all_indices_final,np.arange(len(all_indices_final)))
print(checksum)