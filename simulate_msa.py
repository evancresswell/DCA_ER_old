
from random import choice, shuffle

AAs = 'ACDEFGHIKLMNPQRSTVWY'

#MSA with 1000 sequences
nseqs = 1000

#I'll build up each column and then transpose to MSA
cols = []

# column #1: random
cols.append([choice(AAs) for i in range(nseqs)])

# column #2: random
cols.append([choice(AAs) for i in range(nseqs)])

# column #3: conserved
cols.append(['A']*nseqs)

# column #4: conserved
cols.append(['C']*nseqs)

# column #5: covarying with 6 (only two AAs)
cols.append(['D']*(int(nseqs/2)) + ['K']*(int(nseqs/2)))

# column #6: covarying with 5 (only two AAs)
cols.append(['K']*(int(nseqs/2)) + ['D']*(int(nseqs/2)))

# column #7: covarying with 8 (over all AAs)
cols.append([choice(AAs) for i in range(nseqs)])

# column #8: covarying with 7 (over all AAs)
# col 7 was random, want to generate a dict that uniqly maps col 7's AAs to another set of AAs
AA_dict_values = list(AAs)
shuffle(AA_dict_values)
C8dict = dict((AAs[i],AA_dict_values[i]) for i in range(len(AAs)))
cols.append([C8dict[r] for r in cols[-1]])

# transpose columns to get MSA
MSA = [''.join(s) for s in zip(*cols)]
