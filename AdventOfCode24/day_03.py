import re

with open("input.dat","r") as file:
    megastring = file.read().rstrip('\n')
full_megastring = "do()"+megastring+"don't()"

dos = "do[(][)](.*?)don't[(][)]"
matching_dos = re.findall(dos,full_megastring)

pattern = "mul[(](\\d+[,]\\d+)[)]"
sum = 0
if len(matching_dos) > 0:
    for dostring in matching_dos:
        print(dostring+"\n")
        matches = re.findall(pattern,dostring)
        for match in matches:
            #print(match)
            strs = match.split(",")
            sum += int(strs[0])*int(strs[1])
print(sum)