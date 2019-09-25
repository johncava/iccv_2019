import glob
import random
import numpy as np

##
# Get the dataset with folders from ICCV_Dataset and filter
# based on at least two detected objects
##

short_dirs = glob.glob('./../ICCV_Dataset/*Short/*short/combined/*/')

#mid_dirs = glob.glob('*Mid/*mid/combined/*/')
#far_dirs = glob.glob('*Far/*far/combined/*/')
#dirs = short_dirs + mid_dirs + far_dirs

filter_short_dirs = []
for s in short_dirs:
    pic = s + s.split('/')[-2] + '.jpg'
    check = glob.glob(s + '*.jpg')
    if len(check) <= 2:
        continue
    filter_short_dirs.append(s)

random.shuffle(filter_short_dirs)
threshold = int(len(filter_short_dirs)*0.80)
training, test = filter_short_dirs[:threshold], filter_short_dirs[threshold:]
print(len(short_dirs),len(filter_short_dirs), len(training), len(test))
np.save('training.npy',training)
np.save('test.npy',test)
