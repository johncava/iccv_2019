import os
import glob
import numpy as np
import sys

directory = sys.argv[1]
csv_input_path='./' + directory + '/*.jpg'
files = glob.glob(csv_input_path)

p = []
for file_ in files:
    num = int(file_.split('/')[-1].split('.jpg')[0])
    p.append((num,file_))
files = sorted(p,key=lambda x: x[0])

i=1
for f in files:
    os.rename(f[1],  str(i) + '.jpg')
    i = i+1

