#!/usr/bin/python


import collections
counter_dict = collections.defaultdict(int)

handle = open("structmap.txt", 'r')
output_file = "strcuter.txt"

seen_before = []

cnt=0

for line in handle:
    elems = line.rstrip().split()
    acc = elems[0]
    species = elems[1]
    if '[' in species:
        if ']' in species:
           species = species[1:-1]
    if counter_dict[species] > 0:
        print("IM HERE!!!!!!")
        print(counter_dict[species])
        new_species = f"{species}_{counter_dict[species]}"
    else:
        new_species = species
    counter_dict[species]+=1
    seen_before.append(f"{acc}\t{new_species}")



    
with open(output_file, "w") as outfile:
    for line in seen_before:
        outfile.write(line+"\n")

