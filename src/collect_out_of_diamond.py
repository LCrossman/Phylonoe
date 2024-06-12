#!/usr/bin/python


import sys

handle = open(sys.argv[1], 'r')

flag = 2
seq = ''

for line in handle:
    if line.startswith('         <Hsp_hseq>'):
        flag = 0
    if flag == 0:
        seq = line.rstrip().replace('         <Hsp_hseq>','')
        seq2 = seq.replace('*</Hsp_hseq>','')
    if '*</Hsp_hseq>' in line:
        flag = 1
   

print(">{}\n{}".format(sys.argv[1].replace("_genomic.gbff.faa.db.dmnd.xml",""),seq2))
