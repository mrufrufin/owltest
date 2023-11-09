import csv
import os
curdir = os.path.split(__file__)[0]
datafolder = os.path.join(curdir, 'macaulay', 'csv')
c_outfile = 'macaulay_cmn.txt'
s_outfile = 'macaulay_sci.txt'
wantfiles = [x for x in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, x)) == False]

c_name = set()
s_name = set()
for x in wantfiles:
    curpath = os.path.join(datafolder, x)
    with open(curpath) as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            cur_c = row['Common Name'].lower().strip()
            cur_s = row['Scientific Name'].lower().strip()
            c_name.add(cur_c)
            s_name.add(cur_s)

my_dict = {'common': list(c_name), 'sci': list(s_name)}

with open(c_outfile, 'w') as curf:
    for c in c_name:
        curf.write(c)
        curf.write("\n")

with open(s_outfile, 'w') as curf:
    for c in s_name:
        curf.write(c)
        curf.write("\n")










