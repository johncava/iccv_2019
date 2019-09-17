import os
import glob
import numpy as np
import sys

directory = sys.argv[1]
csv_input_path='./' + directory + '/*.csv'
files = glob.glob(csv_input_path)

p = []
for file_ in files:
    num = int(file_.split('/')[-1].split('.csv')[0])
    p.append((num,file_))
files = sorted(p,key=lambda x: x[0])
# write in first action frame time



i=1
for f in files:
    #rename files in format hours_minutes_seconds_microseconds
    os.rename(f[1],  str(i) + '.csv')
    i = i+1

