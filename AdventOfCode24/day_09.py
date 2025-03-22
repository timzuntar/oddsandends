import numpy as np

with open("input.dat","r") as file:
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

# Now iteratively find the first occurence of a free space and slot the files into it in reverse order (permutation?)

block_positions = np.argwhere(all_indices_flattened != -1).flatten()
free_space_positions = np.argwhere(all_indices_flattened == -1).flatten()

flipped_block_positions = np.flip(block_positions)

for j in range(len(free_space_positions)):
    if free_space_positions[j] < flipped_block_positions[j]:
        all_indices_final[free_space_positions[j]] = all_indices_final[flipped_block_positions[j]]
        all_indices_final[flipped_block_positions[j]] = -1

all_indices_final = all_indices_final[all_indices_final != -1]

checksum  = np.inner(all_indices_final,np.arange(len(all_indices_final)))
print(checksum)